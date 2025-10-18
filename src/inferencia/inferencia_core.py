# src/inferencia/inferencia_core.py
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from .features import ensure_ts, add_time_parts, add_lags_mas, dummies_and_reindex, dummies_and_reindex_with_scaler_means
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

HIST_WINDOW_DAYS = 90
ENABLE_OUTLIER_CAP = False  # si lo necesitas, puedes reactivarlo

def _load_cols(path: str):
    with open(path, "r") as f:
        return json.load(f)

def _safe_ratio(num, den, fallback=1.0):
    try:
        num = float(num)
        den = float(den)
        if den == 0 or np.isnan(num) or np.isnan(den):
            return fallback
        return num / den
    except Exception:
        return fallback

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

    # LÍMITES: SUBIMOS PISO DE TMO PARA NO "APLASTAR" SI YA VIENE BAJO
    factors_calls_by_hour = {h: float(np.clip(v, 0.10, 1.60)) for h, v in factors_calls_by_hour.items()}
    factors_tmo_by_hour   = {h: float(np.clip(v, 0.85, 1.50)) for h, v in factors_tmo_by_hour.items()}

    # Post-feriado (día siguiente)
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

def _is_holiday(ts, holidays_set: set) -> int:
    if not holidays_set:
        return 0
    try:
        d = ts.tz_convert(TIMEZONE).date()
    except Exception:
        d = ts.date()
    return 1 if d in holidays_set else 0

def forecast_120d(df_hist_calls: pd.DataFrame, horizon_days: int = 120, holidays_set: set | None = None):
    # Artefactos
    m_pl = tf.keras.models.load_model(PLANNER_MODEL, compile=False)
    sc_pl = joblib.load(PLANNER_SCALER)
    cols_pl = _load_cols(PLANNER_COLS)

    m_tmo = tf.keras.models.load_model(TMO_MODEL, compile=False)
    sc_tmo = joblib.load(TMO_SCALER)
    cols_tmo = _load_cols(TMO_COLS)

    # Base histórica
    df = ensure_ts(df_hist_calls)
    if TARGET_CALLS not in df.columns:
        raise ValueError(f"Falta columna {TARGET_CALLS} en historical_data.csv")
    df = df[[TARGET_CALLS, TARGET_TMO] if TARGET_TMO in df.columns else [TARGET_CALLS]].copy()
    df = df.dropna(subset=[TARGET_CALLS])

    for c in ["feriados","es_dia_de_pago","tmo_comercial","tmo_tecnico","proporcion_comercial","proporcion_tecnica"]:
        if c in df.columns:
            df[c] = df[c].ffill()

    last_ts = df.index.max()
    start_hist = last_ts - pd.Timedelta(days=HIST_WINDOW_DAYS)
    df_recent = df.loc[df.index >= start_hist].copy()
    if df_recent.empty:
        df_recent = df.copy()

    # Horizonte
    future_ts = pd.date_range(
        last_ts + pd.Timedelta(hours=1),
        periods=horizon_days * 24,
        freq="h",
        tz=TIMEZONE
    )

    # Planner iterativo (con feriados futuro)
    if "feriados" in df_recent.columns:
        dfp = df_recent[[TARGET_CALLS, "feriados"]].copy()
    else:
        dfp = df_recent[[TARGET_CALLS]].copy()
    dfp[TARGET_CALLS] = pd.to_numeric(dfp[TARGET_CALLS], errors="coerce").ffill().fillna(0.0)

    for ts in future_ts:
        tmp = pd.concat([dfp, pd.DataFrame(index=[ts])])
        tmp[TARGET_CALLS] = tmp[TARGET_CALLS].ffill()
        if "feriados" in tmp.columns:
            tmp.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)
        tmp = add_lags_mas(tmp, TARGET_CALLS)
        tmp = add_time_parts(tmp)
        X = dummies_and_reindex(tmp.tail(1), cols_pl)
        yhat = float(m_pl.predict(sc_pl.transform(X), verbose=0).flatten()[0])
        dfp.loc[ts, TARGET_CALLS] = max(0.0, yhat)
        if "feriados" in dfp.columns:
            dfp.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)

    pred_calls = dfp.loc[future_ts, TARGET_CALLS]

    # TMO por hora
    base_tmo = pd.DataFrame(index=future_ts)
    base_tmo[TARGET_CALLS] = pred_calls.values

    # Si llegaron desde merge (main) se usan; si no, valores neutrales
    if "proporcion_comercial" not in df.columns:
        base_tmo["proporcion_comercial"] = 0.5
    else:
        base_tmo["proporcion_comercial"] = df["proporcion_comercial"].ffill().iloc[-1]
    base_tmo["proporcion_tecnica"] = 1.0 - base_tmo["proporcion_comercial"]

    if "tmo_comercial" in df.columns and df["tmo_comercial"].notna().any():
        last_tmo_c = float(df["tmo_comercial"].ffill().iloc[-1])
    else:
        last_tmo_c = float(df[TARGET_TMO].ffill().iloc[-1]) if TARGET_TMO in df.columns and df[TARGET_TMO].notna().any() else 180.0
    if "tmo_tecnico" in df.columns and df["tmo_tecnico"].notna().any():
        last_tmo_t = float(df["tmo_tecnico"].ffill().iloc[-1])
    else:
        last_tmo_t = last_tmo_c

    base_tmo["tmo_comercial"] = last_tmo_c
    base_tmo["tmo_tecnico"]   = last_tmo_t

    if "feriados" in df.columns:
        base_tmo["feriados"] = [ _is_holiday(ts, holidays_set) for ts in base_tmo.index ]

    base_tmo = add_time_parts(base_tmo)
    # USAR MEDIAS DEL SCALER EN FEATURES FALTANTES (solo TMO)
    Xt = dummies_and_reindex_with_scaler_means(base_tmo, cols_tmo, sc_tmo)
    y_tmo = m_tmo.predict(sc_tmo.transform(Xt), verbose=0).flatten()
    y_tmo = np.maximum(0, y_tmo)

    # Curva base
    df_hourly = pd.DataFrame(index=future_ts)
    df_hourly["calls"] = np.round(pred_calls).astype(int)
    df_hourly["tmo_s"] = np.round(y_tmo).astype(int)

    # Ajustes feriados
    if holidays_set and len(holidays_set) > 0:
        (f_calls_by_hour, f_tmo_by_hour,
         g_calls, g_tmo, post_calls_by_hour) = compute_holiday_factors(df, holidays_set)

        df_hourly = apply_holiday_adjustment(
            df_hourly, holidays_set,
            f_calls_by_hour, f_tmo_by_hour,
            col_calls_future="calls", col_tmo_future="tmo_s"
        )
        df_hourly = apply_post_holiday_adjustment(
            df_hourly, holidays_set, post_calls_by_hour,
            col_calls_future="calls"
        )

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
