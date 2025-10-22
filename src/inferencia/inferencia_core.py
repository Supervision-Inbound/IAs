# src/inferencia/inferencia_core.py
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from .features import ensure_ts, add_time_parts, dummies_and_reindex  # <- NO usamos add_lags_mas para llamadas
from .erlang import required_agents, schedule_agents
from .utils_io import write_daily_json, write_hourly_json

TIMEZONE = "America/Santiago"
PUBLIC_DIR = "public"

PLANNER_MODEL = "models/modelo_planner.keras"
PLANNER_SCALER = "models/scaler_planner.pkl"
PLANNER_COLS = "models/training_columns_planner.json"

TMO_MODEL = "models/modelo_tmo.keras"
TMO_SCALER = "models/scaler_tmo.pkl"
TMO_COLS = "models/training_columns_tmo.json"

TARGET_CALLS = "recibidos_nacional"
TARGET_TMO = "tmo_general"

# Ventana reciente para lags/MA (no afecta last_ts)
HIST_WINDOW_DAYS = 90

# ======= Guardrail Outliers (config) =======
ENABLE_OUTLIER_CAP = True   # <- ponlo en False si quieres desactivarlo
K_WEEKDAY = 6.0             # techos +K*MAD en lun-vie
K_WEEKEND = 7.0             # techos +K*MAD en sáb-dom


def _load_cols(path: str):
    with open(path, "r") as f:
        return json.load(f)

# ========= Helpers de feriados y ajustes =========
def _safe_ratio(num, den, fallback=1.0):
    num = float(num) if num is not None and not np.isnan(num) else np.nan
    den = float(den) if den is not None and not np.isnan(den) and den != 0 else np.nan
    if np.isnan(num) or np.isnan(den) or den == 0:
        return fallback
    return num / den


def _series_is_holiday(idx, holidays_set):
    tz = getattr(idx, "tz", None)
    idx_dates = idx.tz_convert(TIMEZONE).date if tz is not None else idx.date
    return pd.Series([d in holidays_set for d in idx_dates], index=idx, dtype=bool)


def compute_holiday_factors(df_hist, holidays_set,
                            col_calls=TARGET_CALLS, col_tmo=TARGET_TMO):
    cols = [col_calls]
    if col_tmo in df_hist.columns:
        cols.append(col_tmo)

    dfh = add_time_parts(df_hist[cols].copy())
    dfh["is_holiday"] = _series_is_holiday(dfh.index, holidays_set)

    med_hol_calls = dfh[dfh["is_holiday"]].groupby("hour")[col_calls].median()
    med_nor_calls = dfh[~dfh["is_holiday"]].groupby("hour")[col_calls].median()

    if col_tmo in dfh.columns:
        med_hol_tmo = dfh[dfh["is_holiday"]].groupby("hour")[col_tmo].median()
        med_nor_tmo = dfh[~dfh["is_holiday"]].groupby("hour")[col_tmo].median()
        g_hol_tmo = dfh[dfh["is_holiday"]][col_tmo].median()
        g_nor_tmo = dfh[~dfh["is_holiday"]][col_tmo].median()
        global_tmo_factor = _safe_ratio(g_hol_tmo, g_nor_tmo, fallback=1.00)
    else:
        global_tmo_factor = 1.00

    g_hol_calls = dfh[dfh["is_holiday"]][col_calls].median()
    g_nor_calls = dfh[~dfh["is_holiday"]][col_calls].median()
    global_calls_factor = _safe_ratio(g_hol_calls, g_nor_calls, fallback=0.75)

    factors_calls_by_hour = {
        int(h): _safe_ratio(med_hol_calls.get(h, np.nan),
                            med_nor_calls.get(h, np.nan),
                            fallback=global_calls_factor)
        for h in range(24)
    }

    if col_tmo in dfh.columns:
        factors_tmo_by_hour = {
            int(h): _safe_ratio(med_hol_tmo.get(h, np.nan),
                                med_nor_tmo.get(h, np.nan),
                                fallback=global_tmo_factor)
            for h in range(24)
        }
    else:
        factors_tmo_by_hour = {int(h): 1.0 for h in range(24)}

    factors_calls_by_hour = {h: float(np.clip(v, 0.10, 1.60)) for h, v in factors_calls_by_hour.items()}
    factors_tmo_by_hour   = {h: float(np.clip(v, 0.70, 1.50)) for h, v in factors_tmo_by_hour.items()}

    dfh = dfh.copy()
    dfh["is_post_hol"] = (~dfh["is_holiday"]) & (dfh["is_holiday"].shift(1).fillna(False))
    med_post_calls = dfh[dfh["is_post_hol"]].groupby("hour")[col_calls].median()
    post_calls_by_hour = {
        int(h): _safe_ratio(med_post_calls.get(h, np.nan),
                            med_nor_calls.get(h, np.nan),
                            fallback=1.05)
        for h in range(24)
    }
    post_calls_by_hour = {h: float(np.clip(v, 0.90, 1.80)) for h, v in post_calls_by_hour.items()}

    return (factors_calls_by_hour, factors_tmo_by_hour,
            global_calls_factor, global_tmo_factor, post_calls_by_hour)


def apply_holiday_adjustment(df_future, holidays_set,
                             factors_calls_by_hour, factors_tmo_by_hour,
                             col_calls_future="calls", col_tmo_future="tmo_s"):
    d = add_time_parts(df_future.copy())
    is_hol = _series_is_holiday(d.index, holidays_set)
    hours = d["hour"].astype(int).values
    call_f = np.array([factors_calls_by_hour.get(int(h), 1.0) for h in hours])
    tmo_f  = np.array([factors_tmo_by_hour.get(int(h), 1.0) for h in hours])
    out = df_future.copy()
    mask = is_hol.values
    out.loc[mask, col_calls_future] = np.round(out.loc[mask, col_calls_future].astype(float) * call_f[mask]).astype(int)
    out.loc[mask, col_tmo_future]   = np.round(out.loc[mask, col_tmo_future].astype(float)   * tmo_f[mask]).astype(int)
    return out


def apply_post_holiday_adjustment(df_future, holidays_set, post_calls_by_hour,
                                  col_calls_future="calls"):
    idx = df_future.index
    prev_idx = (idx - pd.Timedelta(days=1))
    try:
        prev_dates = prev_idx.tz_convert(TIMEZONE).date
        curr_dates = idx.tz_convert(TIMEZONE).date
    except Exception:
        prev_dates = prev_idx.date
        curr_dates = idx.date
    is_prev_hol = pd.Series([d in holidays_set for d in prev_dates], index=idx, dtype=bool)
    is_today_hol = pd.Series([d in holidays_set for d in curr_dates], index=idx, dtype=bool)
    is_post = (~is_today_hol) & (is_prev_hol)
    d = add_time_parts(df_future.copy())
    hours = d["hour"].astype(int).values
    ph_f = np.array([post_calls_by_hour.get(int(h), 1.0) for h in hours])
    out = df_future.copy()
    mask = is_post.values
    out.loc[mask, col_calls_future] = np.round(out.loc[mask, col_calls_future].astype(float) * ph_f[mask]).astype(int)
    return out

# ========= Guardrail outliers =========
def _baseline_median_mad(df_hist, col=TARGET_CALLS):
    d = add_time_parts(df_hist[[col]].copy())
    g = d.groupby(["dow", "hour"])[col]
    base = g.median().rename("med").to_frame()
    mad = g.apply(lambda x: np.median(np.abs(x - np.median(x)))).rename("mad")
    base = base.join(mad)
    if base["mad"].isna().all():
        base["mad"] = 0
    base["mad"] = base["mad"].replace(0, base["mad"].median() if not np.isnan(base["mad"].median()) else 1.0)
    return base.reset_index()


def apply_outlier_cap(df_future, base_median_mad, holidays_set,
                      col_calls_future="calls",
                      k_weekday=K_WEEKDAY, k_weekend=K_WEEKEND):
    if df_future.empty:
        return df_future
    d = add_time_parts(df_future.copy())
    prev_idx = (d.index - pd.Timedelta(days=1))
    try:
        curr_dates = d.index.tz_convert(TIMEZONE).date
        prev_dates = prev_idx.tz_convert(TIMEZONE).date
    except Exception:
        curr_dates = d.index.date
        prev_dates = prev_idx.date
    is_hol = pd.Series([dt in holidays_set for dt in curr_dates], index=d.index, dtype=bool) if holidays_set else pd.Series(False, index=d.index)
    is_prev_hol = pd.Series([dt in holidays_set for dt in prev_dates], index=d.index, dtype=bool) if holidays_set else pd.Series(False, index=d.index)
    is_post_hol = (~is_hol) & (is_prev_hol)

    capped = d.merge(base_median_mad, on=["dow","hour"], how="left")
    capped["mad"] = capped["mad"].fillna(capped["mad"].median() if not np.isnan(capped["mad"].median()) else 1.0)
    capped["med"] = capped["med"].fillna(capped["med"].median() if not np.isnan(capped["med"].median()) else 0.0)

    is_weekend = capped["dow"].isin([5,6]).values
    K = np.where(is_weekend, k_weekend, k_weekday).astype(float)
    upper = capped["med"].values + K * capped["mad"].values

    mask = (~is_hol.values) & (~is_post_hol.values) & (capped[col_calls_future].astype(float).values > upper)
    capped.loc[mask, col_calls_future] = np.round(upper[mask]).astype(int)

    out = df_future.copy()
    out[col_calls_future] = capped[col_calls_future].astype(int).values
    return out

# ========= Aux =========
def _is_holiday(ts, holidays_set: set) -> int:
    if not holidays_set:
        return 0
    try:
        d = ts.tz_convert(TIMEZONE).date()
    except Exception:
        d = ts.date()
    return 1 if d in holidays_set else 0


# =================== FORECAST HÍBRIDO: llamadas blindadas + TMO autorregresivo ===================
def forecast_120d(df_hist_joined: pd.DataFrame, df_hist_tmo_only: pd.DataFrame | None,
                  horizon_days: int = 120, holidays_set: set | None = None):
    """
    1) Llamadas: reconstrucción estricta de features según training_columns_planner.json (idénticas a la original).
       - Sin add_lags_mas ni add_time_parts (se replican a mano).
    2) TMO: autorregresivo (lags/MA de TMO) usando las llamadas ya predichas en (1).
    3) Ajustes por feriados/post-feriados y (opcional) cap de outliers.
    """

    # === Artefactos ===
    m_pl = tf.keras.models.load_model(PLANNER_MODEL, compile=False)
    sc_pl = joblib.load(PLANNER_SCALER)
    cols_pl = _load_cols(PLANNER_COLS)

    m_tmo = tf.keras.models.load_model(TMO_MODEL, compile=False)
    sc_tmo = joblib.load(TMO_SCALER)
    cols_tmo = _load_cols(TMO_COLS)

    # ============================================================
    # 1) LLAMADAS: pipeline original, RECONSTRUCCIÓN ESTRICTA
    # ============================================================
    df_calls_base = ensure_ts(df_hist_joined)
    if TARGET_CALLS not in df_calls_base.columns:
        raise ValueError(f"Falta columna {TARGET_CALLS} en historical_data.csv")

    # Mantener solo llamadas (+ feriados si existe), orden y tipos
    cols_calls = [TARGET_CALLS] + (["feriados"] if "feriados" in df_calls_base.columns else [])
    df_calls_base = df_calls_base[cols_calls].copy().sort_index()
    df_calls_base[TARGET_CALLS] = pd.to_numeric(df_calls_base[TARGET_CALLS], errors="coerce")
    df_calls_base = df_calls_base.dropna(subset=[TARGET_CALLS])

    last_ts_calls = df_calls_base.index.max()
    start_hist_calls = last_ts_calls - pd.Timedelta(days=HIST_WINDOW_DAYS)

    history_calls = df_calls_base.loc[df_calls_base.index >= start_hist_calls].copy()
    if history_calls.empty:
        history_calls = df_calls_base.copy()
    history_calls[TARGET_CALLS] = history_calls[TARGET_CALLS].ffill().fillna(0.0)
    if "feriados" in history_calls.columns:
        history_calls["feriados"] = pd.to_numeric(history_calls["feriados"], errors="coerce").ffill().fillna(0).astype(int)

    future_ts = pd.date_range(
        last_ts_calls + pd.Timedelta(hours=1),
        periods=horizon_days * 24,
        freq="h",
        tz=TIMEZONE
    )

    # Helper: construir features EXACTOS según cols_pl
    def _build_planner_row_features(history_df: pd.DataFrame, ts_pred: pd.Timestamp, cols_pl: list) -> pd.DataFrame:
        """
        history_df: DataFrame con índice temporal y columnas [TARGET_CALLS, (feriados?)]
        ts_pred:    Timestamp objetivo (tz-aware)
        cols_pl:    lista exacta de columnas del entrenamiento del planner
        """
        # 1) Identificar lags y MAs requeridas por nombre en cols_pl
        needed_lags = []
        needed_ma = []
        for c in cols_pl:
            if c.startswith("lag_"):
                try: needed_lags.append(int(c.split("_")[1]))
                except: pass
            elif c.startswith("ma_"):
                try: needed_ma.append(int(c.split("_")[1]))
                except: pass

        h = history_df.copy().sort_index()

        # 2) Generar lags de llamadas
        for L in sorted(set(needed_lags)):
            h[f"lag_{L}"] = h[TARGET_CALLS].shift(L)

        # 3) Generar medias móviles
        for W in sorted(set(needed_ma)):
            # IMPORTANTE:
            # Tu entrenamiento v7 (según script compartido) calculó: rolling(W, min_periods=1).mean() SIN shift(1).
            # Si tu entrenamiento REAL usó shift(1), activa la siguiente línea y comenta la anterior.
            h[f"ma_{W}"] = h[TARGET_CALLS].rolling(W, min_periods=1).mean()
            # h[f"ma_{W}"] = h[TARGET_CALLS].rolling(W, min_periods=1).mean().shift(1)

        # 4) Construir la fila objetivo
        local_ts = ts_pred.tz_convert(TIMEZONE) if getattr(ts_pred, "tz", None) is not None else ts_pred
        dow = int(local_ts.weekday())
        month = int(local_ts.month)
        hour = int(local_ts.hour)
        day = int(local_ts.day)
        sin_hour = float(np.sin(2 * np.pi * hour / 24))
        cos_hour = float(np.cos(2 * np.pi * hour / 24))
        sin_dow  = float(np.sin(2 * np.pi * dow / 7))
        cos_dow  = float(np.cos(2 * np.pi * dow / 7))

        # valor feriado (se sobreescribe abajo si corresponde)
        fer = 0.0

        last = h.iloc[-1]
        X_dict = {}
        for c in cols_pl:
            if c.startswith("lag_") or c.startswith("ma_"):
                X_dict[c] = float(last.get(c, np.nan))
            elif c == "sin_hour":
                X_dict[c] = sin_hour
            elif c == "cos_hour":
                X_dict[c] = cos_hour
            elif c == "sin_dow":
                X_dict[c] = sin_dow
            elif c == "cos_dow":
                X_dict[c] = cos_dow
            elif c == "feriados":
                X_dict[c] = fer
            elif c.startswith("dow_"):
                try: val = int(c.split("_")[1])
                except: val = -1
                X_dict[c] = 1.0 if val == dow else 0.0
            elif c.startswith("month_"):
                try: val = int(c.split("_")[1])
                except: val = -1
                X_dict[c] = 1.0 if val == month else 0.0
            elif c.startswith("hour_"):
                try: val = int(c.split("_")[1])
                except: val = -1
                X_dict[c] = 1.0 if val == hour else 0.0
            else:
                X_dict[c] = 0.0

        X_df = pd.DataFrame([[X_dict.get(c, 0.0) for c in cols_pl]], columns=cols_pl, index=[ts_pred])
        return X_df

    pred_calls_list = []
    for ts in future_ts:
        # set feriado en la historia como en la original
        if "feriados" in history_calls.columns:
            history_calls.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)

        # features exactos según training_columns_planner.json
        X_pl = _build_planner_row_features(history_calls, ts, cols_pl)

        # si feriados está en cols_pl, pisamos con el valor real del ts
        if "feriados" in history_calls.columns and "feriados" in cols_pl:
            X_pl.loc[ts, "feriados"] = float(history_calls.loc[ts, "feriados"])

        # predicción
        yhat = float(m_pl.predict(sc_pl.transform(X_pl.astype(np.float64)), verbose=0).flatten()[0])
        yhat = max(0.0, yhat)
        pred_calls_list.append(yhat)

        # actualizar historia (para próximos lags/MAs)
        history_calls.loc[ts, TARGET_CALLS] = yhat

    pred_calls = pd.Series(pred_calls_list, index=future_ts, dtype=float)

    # ============================================================
    # 2) TMO: autorregresivo (ya podemos tocar df para TMO)
    # ============================================================
    df = ensure_ts(df_hist_joined)
    if TARGET_TMO not in df.columns:
        df[TARGET_TMO] = np.nan

    # Si viene histórico TMO-only, usarlo para sobreescribir cols TMO y auxiliares
    if df_hist_tmo_only is not None and not df_hist_tmo_only.empty:
        try:
            df_tmo_pure = ensure_ts(df_hist_tmo_only).ffill()
            if not df_tmo_pure.empty:
                print("INFO: Sobrescribiendo features TMO con HISTORICO_TMO.csv")
                df.update(df_tmo_pure)
        except Exception as e:
            print(f"WARN: Error procesando df_hist_tmo_only ({e}), usando datos unidos).")

    # Auxiliares TMO
    for aux in [TARGET_TMO, "feriados", "es_dia_de_pago",
                "tmo_comercial", "tmo_tecnico", "proporcion_comercial", "proporcion_tecnica"]:
        if aux in df.columns:
            df[aux] = pd.to_numeric(df[aux], errors="coerce").ffill()

    last_ts = df.index.max()
    start_hist = last_ts - pd.Timedelta(days=HIST_WINDOW_DAYS)
    df_recent = df.loc[df.index >= start_hist].copy()
    if df_recent.empty:
        df_recent = df.copy()

    # Estáticos TMO (mediana 14d “laboral” con fallback)
    tmo_static_features = {"proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico"}
    avail_static = [c for c in tmo_static_features if c in df.columns]
    static_tmo_cols_dict = {c: 0.0 for c in tmo_static_features}
    if avail_static:
        last_ts_features = df.index.max()
        mask = ((df.index >= last_ts_features - pd.Timedelta(days=14)) &
                (df.index.hour >= 8) & (df.index.hour <= 20))
        recent_data = df.loc[mask, avail_static]
        if not recent_data.empty and not recent_data[avail_static].isnull().all().all():
            med = recent_data[avail_static].apply(pd.to_numeric, errors="coerce").median()
            for c in avail_static:
                v = med.get(c)
                if pd.notna(v):
                    static_tmo_cols_dict[c] = float(v)
            print(f"INFO: Estáticos TMO (mediana 14d): {{k: static_tmo_cols_dict[k] for k in sorted(static_tmo_cols_dict)}}")
        else:
            last_row = df[avail_static].tail(1).apply(pd.to_numeric, errors="coerce")
            if not last_row.empty:
                for c in avail_static:
                    v = last_row.iloc[0].get(c)
                    if pd.notna(v):
                        static_tmo_cols_dict[c] = float(v)
                print(f"WARN: Usando últimos valores TMO estáticos: {{k: static_tmo_cols_dict[k] for k in sorted(static_tmo_cols_dict)}}")
            else:
                print("WARN: Sin datos TMO estáticos; usando ceros.")
    else:
        print("WARN: No existen columnas TMO estáticas; usando ceros.")

    # dfp_tmo lleva TMO + feriados + estáticos; llamadas se inyectan por hora
    cols_iter_tmo = [TARGET_TMO] + (["feriados"] if "feriados" in df_recent.columns else [])
    dfp_tmo = df_recent[cols_iter_tmo].copy().sort_index()
    dfp_tmo[TARGET_TMO] = pd.to_numeric(dfp_tmo[TARGET_TMO], errors="coerce").ffill().fillna(0.0)
    for c, v in static_tmo_cols_dict.items():
        dfp_tmo[c] = v

    for ts in future_ts:
        tmp = pd.concat([dfp_tmo, pd.DataFrame(index=[ts])]).sort_index()
        tmp[TARGET_TMO] = tmp[TARGET_TMO].ffill()
        for c in static_tmo_cols_dict.keys():
            tmp.loc[ts, c] = static_tmo_cols_dict[c]
        if "feriados" in tmp.columns:
            tmp.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)

        # inyectar llamadas predichas
        if TARGET_CALLS not in tmp.columns:
            tmp[TARGET_CALLS] = np.nan
        tmp[TARGET_CALLS] = tmp[TARGET_CALLS].ffill()
        tmp.loc[ts, TARGET_CALLS] = float(pred_calls.loc[ts])

        # lags/MA TMO (autorregresivo)
        for lag in [24, 48, 72, 168]:
            tmp[f'tmo_lag_{lag}'] = tmp[TARGET_TMO].shift(lag)
        for window in [24, 72, 168]:
            tmp[f'tmo_ma_{window}'] = tmp[TARGET_TMO].rolling(window, min_periods=1).mean()

        tmp = add_time_parts(tmp)
        X_tmo = dummies_and_reindex(tmp.tail(1), cols_tmo)
        yhat_tmo = float(m_tmo.predict(sc_tmo.transform(X_tmo.astype(np.float64)), verbose=0).flatten()[0])
        dfp_tmo.loc[ts, TARGET_TMO] = max(0.0, yhat_tmo)
        if "feriados" in dfp_tmo.columns:
            dfp_tmo.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)

    # ===== Curva base =====
    df_hourly = pd.DataFrame(index=future_ts)
    df_hourly["calls"] = np.round(pred_calls).astype(int)
    df_hourly["tmo_s"] = np.round(dfp_tmo.loc[future_ts, TARGET_TMO]).astype(int)

    # Ajustes feriados/post
    if holidays_set and len(holidays_set) > 0:
        (f_calls_by_hour, f_tmo_by_hour,
         _, _, post_calls_by_hour) = compute_holiday_factors(df, holidays_set)
        df_hourly = apply_holiday_adjustment(df_hourly, holidays_set,
                                             f_calls_by_hour, f_tmo_by_hour,
                                             col_calls_future="calls", col_tmo_future="tmo_s")
        df_hourly = apply_post_holiday_adjustment(df_hourly, holidays_set, post_calls_by_hour,
                                                  col_calls_future="calls")

    # Cap de outliers (sobre llamadas)
    if ENABLE_OUTLIER_CAP:
        base_mad = _baseline_median_mad(df_calls_base, col=TARGET_CALLS)
        df_hourly = apply_outlier_cap(df_hourly, base_mad, holidays_set,
                                      col_calls_future="calls",
                                      k_weekday=K_WEEKDAY, k_weekend=K_WEEKEND)

    # Erlang
    df_hourly["agents_prod"] = 0
    for ts in df_hourly.index:
        a, _ = required_agents(float(df_hourly.at[ts, "calls"]), float(df_hourly.at[ts, "tmo_s"]))
        df_hourly.at[ts, "agents_prod"] = int(a)
    df_hourly["agents_sched"] = df_hourly["agents_prod"].apply(schedule_agents)

    # Salidas
    write_hourly_json(f"{PUBLIC_DIR}/prediccion_horaria.json",
                      df_hourly, "calls", "tmo_s", "agents_sched")
    write_daily_json(f"{PUBLIC_DIR}/prediccion_diaria.json",
                     df_hourly, "calls", "tmo_s")

    return df_hourly

