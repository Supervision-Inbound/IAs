# src/inferencia/inferencia_core.py
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from .features import (
    ensure_ts,
    add_time_parts,
    add_lags_mas,
    dummies_and_reindex,
    dummies_and_reindex_with_scaler_means,
)
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


def compute_holiday_factors(
    df_hist_calls, holidays_set, col_calls=TARGET_CALLS
):
    # Solo factores de llamadas por feriado/post-feriado (el TMO se maneja por su propio histórico)
    dfh = add_time_parts(df_hist_calls[[col_calls]].copy())
    dfh["is_holiday"] = _series_is_holiday(dfh.index, holidays_set)

    med_hol_calls = dfh[dfh["is_holiday"]].groupby("hour")[col_calls].median()
    med_nor_calls = dfh[~dfh["is_holiday"]].groupby("hour")[col_calls].median()

    g_hol_calls = dfh[dfh["is_holiday"]][col_calls].median()
    g_nor_calls = dfh[~dfh["is_holiday"]][col_calls].median()
    global_calls_factor = _safe_ratio(g_hol_calls, g_nor_calls, fallback=0.75)

    factors_calls_by_hour = {
        int(h): _safe_ratio(
            med_hol_calls.get(h, np.nan),
            med_nor_calls.get(h, np.nan),
            fallback=global_calls_factor,
        )
        for h in range(24)
    }
    factors_calls_by_hour = {h: float(np.clip(v, 0.10, 1.60)) for h, v in factors_calls_by_hour.items()}

    # post-feriado
    dfh = dfh.copy()
    dfh["is_post_hol"] = (~dfh["is_holiday"]) & (dfh["is_holiday"].shift(1).fillna(False))
    med_post_calls = dfh[dfh["is_post_hol"]].groupby("hour")[col_calls].median()
    post_calls_by_hour = {
        int(h): _safe_ratio(med_post_calls.get(h, np.nan), med_nor_calls.get(h, np.nan), fallback=1.05)
        for h in range(24)
    }
    post_calls_by_hour = {h: float(np.clip(v, 0.90, 1.80)) for h, v in post_calls_by_hour.items()}
    return factors_calls_by_hour, post_calls_by_hour


def apply_holiday_adjustment_calls(
    df_future, holidays_set, factors_calls_by_hour, col_calls_future="calls"
):
    d = add_time_parts(df_future.copy())
    is_hol = _series_is_holiday(d.index, holidays_set)
    hours = d["hour"].astype(int).values
    call_f = np.array([factors_calls_by_hour.get(int(h), 1.0) for h in hours])

    out = df_future.copy()
    mask = is_hol.values
    out.loc[mask, col_calls_future] = (
        np.round(out.loc[mask, col_calls_future].astype(float) * call_f[mask]).astype(int)
    )
    return out


def apply_post_holiday_adjustment_calls(
    df_future, holidays_set, post_calls_by_hour, col_calls_future="calls"
):
    idx = df_future.index
    prev_idx = idx - pd.Timedelta(days=1)
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
    out.loc[mask, col_calls_future] = (
        np.round(out.loc[mask, col_calls_future].astype(float) * ph_f[mask]).astype(int)
    )
    return out


def _is_holiday(ts, holidays_set: set) -> int:
    if not holidays_set:
        return 0
    try:
        d = ts.tz_convert(TIMEZONE).date()
    except Exception:
        d = ts.date()
    return 1 if d in holidays_set else 0


def forecast_120d(
    df_hist_calls: pd.DataFrame,
    df_tmo_hist: pd.DataFrame,
    horizon_days: int = 120,
    holidays_set: set | None = None,
):
    # ========= Modelos =========
    m_pl = tf.keras.models.load_model(PLANNER_MODEL, compile=False)
    sc_pl = joblib.load(PLANNER_SCALER)
    cols_pl = _load_cols(PLANNER_COLS)

    m_tmo = tf.keras.models.load_model(TMO_MODEL, compile=False)
    sc_tmo = joblib.load(TMO_SCALER)
    cols_tmo = _load_cols(TMO_COLS)

    # ========= Llamadas (historical_data) =========
    df_calls = ensure_ts(df_hist_calls)
    if TARGET_CALLS not in df_calls.columns:
        raise ValueError(f"Falta columna {TARGET_CALLS} en historical_data.csv")
    df_calls = df_calls[[TARGET_CALLS] + (["feriados"] if "feriados" in df_calls.columns else [])].copy()
    df_calls = df_calls.dropna(subset=[TARGET_CALLS])

    last_ts = df_calls.index.max()
    start_hist = last_ts - pd.Timedelta(days=HIST_WINDOW_DAYS)
    df_recent = df_calls.loc[df_calls.index >= start_hist].copy()
    if df_recent.empty:
        df_recent = df_calls.copy()

    future_ts = pd.date_range(
        last_ts + pd.Timedelta(hours=1),
        periods=horizon_days * 24,
        freq="h",
        tz=TIMEZONE,
    )

    # Planner iterativo (con feriados futuro si los tienes)
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

    # Ajustes de llamadas por feriado/post-feriado (solo llamadas)
    if holidays_set and len(holidays_set) > 0:
        f_calls_by_hour, post_calls_by_hour = compute_holiday_factors(df_calls, holidays_set)
    else:
        f_calls_by_hour, post_calls_by_hour = None, None

    # ========= TMO (HISTORICO_TMO) =========
    # Perfiles y proporciones se calculan EXCLUSIVAMENTE de df_tmo_hist
    df_tmo = df_tmo_hist.copy()
    df_tmo = df_tmo[
        [c for c in [
            "q_llamadas_comercial",
            "q_llamadas_tecnico",
            "proporcion_comercial",
            "proporcion_tecnica",
            "tmo_comercial",
            "tmo_tecnico",
            "tmo_general",
        ] if c in df_tmo.columns]
    ].dropna(how="all")

    # ventana (84d) para estacionalidad de TMO
    if not df_tmo.empty:
        df_tmo = df_tmo.loc[df_tmo.index >= (df_tmo.index.max() - pd.Timedelta(days=84))]
        df_tmo = add_time_parts(df_tmo)
    else:
        # fallback vacío con valores neutros
        df_tmo = pd.DataFrame(index=future_ts)
        df_tmo["proporcion_comercial"] = 0.5
        df_tmo["proporcion_tecnica"] = 0.5
        df_tmo["tmo_comercial"] = 180.0
        df_tmo["tmo_tecnico"] = 180.0
        df_tmo = add_time_parts(df_tmo)

    def _profile(df_source, col, default_val):
        if col in df_source.columns and df_source[col].notna().any():
            return df_source.groupby(["dow","hour"])[col].median()
        # serie constante por si acaso
        return pd.Series(default_val, index=pd.MultiIndex.from_product([range(7), range(24)]))

    prof_prop_com = _profile(df_tmo, "proporcion_comercial", 0.5).clip(0,1)
    prof_prop_tec = (1.0 - prof_prop_com).clip(0,1)
    prof_tmo_com  = _profile(df_tmo, "tmo_comercial", 180.0)
    prof_tmo_tec  = _profile(df_tmo, "tmo_tecnico",   180.0)

    # Construimos base futura de TMO a partir de perfiles (dow,hour)
    base_tmo = pd.DataFrame(index=future_ts)
    parts_future = add_time_parts(pd.DataFrame(index=future_ts))

    def _map_profile(profile, default_value=0.0):
        vals = []
        for dw, hr in zip(parts_future["dow"].values, parts_future["hour"].values):
            try:
                v = profile.loc[(int(dw), int(hr))]
            except KeyError:
                v = default_value
            vals.append(v)
        return pd.Series(vals, index=future_ts).astype(float)

    prop_com = _map_profile(prof_prop_com, 0.5).clip(0,1)
    prop_tec = _map_profile(prof_prop_tec, 0.5).clip(0,1)
    tmo_com  = _map_profile(prof_tmo_com, 180.0)
    tmo_tec  = _map_profile(prof_tmo_tec, 180.0)

    base_tmo["proporcion_comercial"] = prop_com.values
    base_tmo["proporcion_tecnica"]   = prop_tec.values
    base_tmo["tmo_comercial"]        = tmo_com.values
    base_tmo["tmo_tecnico"]          = tmo_tec.values

    # Features para el modelo de TMO (con medias del scaler en faltantes)
    base_tmo["calls"] = pred_calls.values  # si tu modelo TMO usa calls, queda disponible
    base_tmo = add_time_parts(base_tmo)
    Xt = dummies_and_reindex_with_scaler_means(base_tmo, cols_tmo, sc_tmo)
    y_tmo = m_tmo.predict(sc_tmo.transform(Xt), verbose=0).flatten()
    y_tmo = np.maximum(y_tmo, 30.0)

    # ========= Resultado horario =========
    df_hourly = pd.DataFrame(index=future_ts)
    df_hourly["calls"] = np.round(pred_calls).astype(int)
    df_hourly["tmo_s"] = np.round(y_tmo).astype(int)
    df_hourly["proporcion_comercial"] = prop_com.values
    df_hourly["proporcion_tecnica"]   = prop_tec.values
    df_hourly["tmo_comercial"]        = np.round(tmo_com.values).astype(int)
    df_hourly["tmo_tecnico"]          = np.round(tmo_tec.values).astype(int)

    # Ajustes por feriados SOLO a llamadas (si están activados)
    if holidays_set and len(holidays_set) > 0 and f_calls_by_hour is not None:
        df_hourly = apply_holiday_adjustment_calls(
            df_hourly, holidays_set, f_calls_by_hour, col_calls_future="calls"
        )
        df_hourly = apply_post_holiday_adjustment_calls(
            df_hourly, holidays_set, post_calls_by_hour, col_calls_future="calls"
        )

    # Erlang
    df_hourly["agents_prod"] = 0
    for ts in df_hourly.index:
        a, _ = required_agents(
            float(df_hourly.at[ts, "calls"]), float(df_hourly.at[ts, "tmo_s"])
        )
        df_hourly.at[ts, "agents_prod"] = int(a)
    df_hourly["agents_sched"] = df_hourly["agents_prod"].apply(schedule_agents)

    # Salidas
    write_hourly_json(
        f"{PUBLIC_DIR}/prediccion_horaria.json",
        df_hourly,
        "calls",
        "tmo_s",
        "agents_sched",
    )
    write_daily_json(
        f"{PUBLIC_DIR}/prediccion_diaria.json",
        df_hourly,
        "calls",
        "tmo_s",
    )

    return df_hourly

