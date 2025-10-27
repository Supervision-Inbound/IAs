# src/inferencia/inferencia_core.py
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from .features import ensure_ts, add_time_parts, add_lags_mas, dummies_and_reindex
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

# ======= Ajuste clave: el volumen ahora es 'contestados' =======
TARGET_CALLS = "contestados"
TARGET_TMO = "tmo_general"

# Ventana de lookback para construir lags pro
HIST_WINDOW_DAYS = 90

# ======= NUEVO: Guardrail Outliers (config) =======
ENABLE_OUTLIER_CAP = True
K_WEEKDAY = 6.0
K_WEEKEND = 7.0

# ======= Artefactos TMO residual =======
TMO_BASELINE = "models/tmo_baseline_dow_hour.csv"
TMO_META     = "models/tmo_residual_meta.json"

def _load_cols(path: str):
    with open(path, "r") as f:
        return json.load(f)

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
        med_hol_tmo = med_nor_tmo = None
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

    if med_hol_tmo is not None:
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

    base = base_median_mad.copy()
    capped = d.merge(base, on=["dow","hour"], how="left")
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

def _is_holiday(ts, holidays_set: set) -> int:
    if not holidays_set:
        return 0
    try:
        d = ts.tz_convert(TIMEZONE).date()
    except Exception:
        d = ts.date()
    return 1 if d in holidays_set else 0

# ======= Residual TMO helpers =======
def _load_tmo_residual_artifacts():
    with open(TMO_META, "r", encoding="utf-8") as f:
        meta = json.load(f)
    base = pd.read_csv(TMO_BASELINE)
    base = base[["dow","hour","tmo_baseline"]].copy()
    base["dow"]  = base["dow"].astype(int)
    base["hour"] = base["hour"].astype(int)
    resid_mean = float(meta.get("resid_mean", 0.0))
    resid_std  = float(meta.get("resid_std",  1.0)) or 1.0
    return base, resid_mean, resid_std

def _add_tmo_resid_features(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in.copy()
    for lag in [1,2,3,6,12,24,48,72,168]:
        d[f"lag_resid_{lag}"] = d["tmo_resid"].shift(lag)
    r1 = d["tmo_resid"].shift(1)
    for w in [6,12,24,72,168]:
        d[f"ma_resid_{w}"] = r1.rolling(w, min_periods=1).mean()
    for span in [6,12,24]:
        d[f"ema_resid_{span}"] = r1.ewm(span=span, adjust=False, min_periods=1).mean()
    for w in [24,72]:
        d[f"std_resid_{w}"] = r1.rolling(w, min_periods=2).std()
        d[f"max_resid_{w}"] = r1.rolling(w, min_periods=1).max()
    d = add_time_parts(d)
    return d

def forecast_120d(df_hist_joined: pd.DataFrame, horizon_days: int = 120, holidays_set: set | None = None):
    """
    Versión Autorregresiva (v8)
    - Planner iterativo (volumen = contestados).
    - TMO residual iterativo usando baseline (dow,hour) + residuo desestandarizado.
    - Ajustes por FERIADOS y POST-FERIADOS.
    - CAP outliers por (dow,hour) con mediana+MAD.
    - Erlang C y salidas JSON.
    """
    # === Artefactos ===
    m_pl = tf.keras.models.load_model(PLANNER_MODEL, compile=False)
    sc_pl = joblib.load(PLANNER_SCALER)
    cols_pl = _load_cols(PLANNER_COLS)

    m_tmo = tf.keras.models.load_model(TMO_MODEL, compile=False)
    sc_tmo = joblib.load(TMO_SCALER)
    cols_tmo = _load_cols(TMO_COLS)
    tmo_base_table, resid_mean, resid_std = _load_tmo_residual_artifacts()

    # === Base histórica (para lags) ===
    df = ensure_ts(df_hist_joined)

    if TARGET_CALLS not in df.columns:
        raise ValueError(f"Falta columna {TARGET_CALLS} en historical_data.csv")
    if "feriados" not in df.columns:
        df["feriados"] = 0
    if "es_dia_de_pago" not in df.columns:
        df["es_dia_de_pago"] = 0

    # baseline histórica y residuo histórico (si hay TMO observado)
    keep_cols = [TARGET_CALLS, "feriados", "es_dia_de_pago"]
    if TARGET_TMO in df.columns:
        keep_cols.append(TARGET_TMO)
    df_tmp = add_time_parts(df[keep_cols].copy())
    df_tmp = df_tmp.merge(tmo_base_table, on=["dow","hour"], how="left")
    if TARGET_TMO in df_tmp.columns:
        df_tmp["tmo_resid"] = pd.to_numeric(df_tmp[TARGET_TMO], errors="coerce") - df_tmp["tmo_baseline"]
    else:
        df_tmp["tmo_resid"] = np.nan

    # ventana reciente para lags
    last_ts = df_tmp.index.max()
    dfp = df_tmp.loc[df_tmp.index >= last_ts - pd.Timedelta(days=HIST_WINDOW_DAYS)].copy()
    if dfp.empty:
        dfp = df_tmp.copy()

    dfp[TARGET_CALLS] = pd.to_numeric(dfp[TARGET_CALLS], errors="coerce").ffill().fillna(0.0)
    dfp["tmo_resid"] = pd.to_numeric(dfp["tmo_resid"], errors="coerce")
    if dfp["tmo_resid"].isna().all():
        dfp["tmo_resid"] = 0.0

    # ===== Horizonte futuro =====
    future_ts = pd.date_range(
        last_ts + pd.Timedelta(hours=1),
        periods=horizon_days * 24,
        freq="h",
        tz=TIMEZONE
    )

    # ===== Bucle Iterativo (contestados + TMO) =====
    print("Iniciando predicción iterativa (Contestados + TMO residual)...")
    for ts in future_ts:
        # 1) Volumen (contestados) con planner
        tmp_calls = dfp[[TARGET_CALLS, "feriados","es_dia_de_pago"]].copy()
        tmp_calls = add_lags_mas(tmp_calls, TARGET_CALLS)
        tmp_calls = add_time_parts(tmp_calls)
        X_pl = dummies_and_reindex(tmp_calls.tail(1), cols_pl)
        yhat_calls = float(m_pl.predict(sc_pl.transform(X_pl), verbose=0).flatten()[0])
        yhat_calls = max(0.0, yhat_calls)

        # 2) Baseline TMO para este ts
        dow = int(ts.tz_convert(TIMEZONE).weekday())
        hour = int(ts.tz_convert(TIMEZONE).hour)
        base_row = tmo_base_table[(tmo_base_table["dow"]==dow)&(tmo_base_table["hour"]==hour)]
        tmo_base = float(base_row["tmo_baseline"].iloc[0]) if not base_row.empty else float(dfp["tmo_baseline"].median())

        # 3) Features residuales TMO
        tmp_tmo = dfp[["tmo_resid","feriados","es_dia_de_pago"]].copy()
        tmp_tmo.loc[ts, ["tmo_resid","feriados","es_dia_de_pago"]] = [tmp_tmo["tmo_resid"].iloc[-1], 0, 0]
        tmp_tmo = _add_tmo_resid_features(tmp_tmo)
        X_tmo = dummies_and_reindex(tmp_tmo.tail(1), cols_tmo)

        # 4) Pred residuo z -> residuo sec -> TMO sec
        yhat_z = float(m_tmo.predict(sc_tmo.transform(X_tmo), verbose=0).flatten()[0])
        yhat_resid = yhat_z * resid_std + resid_mean
        yhat_tmo = max(0.0, tmo_base + yhat_resid)

        # 5) Actualizar series
        dfp.loc[ts, TARGET_CALLS] = yhat_calls
        dfp.loc[ts, "tmo_baseline"] = tmo_base
        dfp.loc[ts, "tmo_resid"] = yhat_tmo - tmo_base
        dfp.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)
        dfp.loc[ts, "es_dia_de_pago"] = 0

    print("Predicción iterativa completada.")

    # ===== Curva base =====
    df_hourly = pd.DataFrame(index=future_ts)
    # mantenemos el nombre 'calls' en la salida para compatibilidad con alertas y front,
    # pero representa 'contestados' predichos
    df_hourly["calls"] = np.round(dfp.loc[future_ts, TARGET_CALLS]).astype(int)
    df_hourly["tmo_s"] = np.round(dfp.loc[future_ts, "tmo_baseline"] + dfp.loc[future_ts, "tmo_resid"]).astype(int)

    # ===== Ajustes por feriados (opcional) =====
    if holidays_set and len(holidays_set) > 0:
        (f_calls_by_hour, f_tmo_by_hour,
         _g_calls, _g_tmo, post_calls_by_hour) = compute_holiday_factors(df, holidays_set, col_calls=TARGET_CALLS, col_tmo=TARGET_TMO)
        df_hourly = apply_holiday_adjustment(
            df_hourly, holidays_set,
            f_calls_by_hour, f_tmo_by_hour,
            col_calls_future="calls", col_tmo_future="tmo_s"
        )
        df_hourly = apply_post_holiday_adjustment(
            df_hourly, holidays_set, post_calls_by_hour,
            col_calls_future="calls"
        )

    # ===== (OPCIONAL) CAP de OUTLIERS en volumen =====
    if ENABLE_OUTLIER_CAP:
        base_mad = _baseline_median_mad(df, col=TARGET_CALLS)
        df_hourly = apply_outlier_cap(
            df_hourly, base_mad, holidays_set,
            col_calls_future="calls",
            k_weekday=K_WEEKDAY, k_weekend=K_WEEKEND
        )

    # ===== Erlang por hora =====
    df_hourly["agents_prod"] = 0
    for ts in df_hourly.index:
        a, _ = required_agents(float(df_hourly.at[ts, "calls"]), float(df_hourly.at[ts, "tmo_s"]))
        df_hourly.at[ts, "agents_prod"] = int(a)
    df_hourly["agents_sched"] = df_hourly["agents_prod"].apply(schedule_agents)

    # ===== Salidas =====
    write_hourly_json(f"{PUBLIC_DIR}/prediccion_horaria.json",
                      df_hourly, "calls", "tmo_s", "agents_sched")
    # PROMEDIO DIARIO PONDERADO por 'contestados' (en futuro usamos 'calls' que es contestados predicho)
    write_daily_json(f"{PUBLIC_DIR}/prediccion_diaria.json",
                     df_hourly, "calls", "tmo_s", weights_col="calls")

    return df_hourly
