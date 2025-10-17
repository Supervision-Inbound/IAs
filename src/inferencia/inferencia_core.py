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

TARGET_CALLS = "recibidos_nacional"
TARGET_TMO = "tmo_general"

HIST_WINDOW_DAYS = 90

def _load_cols(path: str):
    with open(path, "r") as f:
        return json.load(f)

def _is_holiday(ts, holidays_set: set) -> int:
    """ts es Timestamp con tz; devuelve 1 si su fecha está en el set de feriados."""
    if holidays_set is None:
        return 0
    try:
        d = ts.tz_convert(TIMEZONE).date()
    except Exception:
        d = ts.date()
    return 1 if d in holidays_set else 0

def forecast_120d(df_hist_calls: pd.DataFrame, horizon_days: int = 120, holidays_set: set | None = None):
    """
    - Parser robusto.
    - Filtro dropna(subset=[TARGET_CALLS]) (sin cap a hoy).
    - Horizonte = 1h después de last_ts.
    - Planner iterativo usando 'feriados' también en FUTURO.
    - TMO horario usando 'feriados' futuro.
    - Erlang C y salidas JSON.
    """
    # === Artefactos ===
    m_pl = tf.keras.models.load_model(PLANNER_MODEL, compile=False)
    sc_pl = joblib.load(PLANNER_SCALER)
    cols_pl = _load_cols(PLANNER_COLS)

    m_tmo = tf.keras.models.load_model(TMO_MODEL, compile=False)
    sc_tmo = joblib.load(TMO_SCALER)
    cols_tmo = _load_cols(TMO_COLS)

    # === Base histórica ===
    df = ensure_ts(df_hist_calls)

    if TARGET_CALLS not in df.columns:
        raise ValueError(f"Falta columna {TARGET_CALLS} en historical_data.csv")

    df = df[[TARGET_CALLS, TARGET_TMO] if TARGET_TMO in df.columns else [TARGET_CALLS]].copy()
    df = df.dropna(subset=[TARGET_CALLS])

    # forward-fill de auxiliares comunes (por si existen)
    for aux in ["feriados", "es_dia_de_pago", "tmo_comercial", "tmo_tecnico",
                "proporcion_comercial", "proporcion_tecnica"]:
        if aux in df.columns:
            df[aux] = df[aux].ffill()

    last_ts = df.index.max()

    start_hist = last_ts - pd.Timedelta(days=HIST_WINDOW_DAYS)
    df_recent = df.loc[df.index >= start_hist].copy()
    if df_recent.empty:
        df_recent = df.copy()

    # ===== Horizonte futuro =====
    future_ts = pd.date_range(
        last_ts + pd.Timedelta(hours=1),
        periods=horizon_days * 24,
        freq="h",
        tz=TIMEZONE
    )

    # ===== Planner iterativo (con 'feriados' futuro) =====
    # base dfp con histórico (si tenía 'feriados', lo mantenemos)
    if "feriados" in df_recent.columns:
        dfp = df_recent[[TARGET_CALLS, "feriados"]].copy()
    else:
        dfp = df_recent[[TARGET_CALLS]].copy()

    dfp[TARGET_CALLS] = pd.to_numeric(dfp[TARGET_CALLS], errors="coerce").ffill().fillna(0.0)

    for ts in future_ts:
        tmp = pd.concat([dfp, pd.DataFrame(index=[ts])])
        # forward fill calls para construir lags/MA
        tmp[TARGET_CALLS] = tmp[TARGET_CALLS].ffill()

        # marcar feriado futuro
        if "feriados" in tmp.columns:
            tmp.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)

        tmp = add_lags_mas(tmp, TARGET_CALLS)
        tmp = add_time_parts(tmp)

        X = dummies_and_reindex(tmp.tail(1), cols_pl)
        yhat = float(m_pl.predict(sc_pl.transform(X), verbose=0).flatten()[0])
        dfp.loc[ts, TARGET_CALLS] = max(0.0, yhat)

        # persistir bandera (para el siguiente paso si existía en histórico)
        if "feriados" in dfp.columns:
            dfp.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)

    pred_calls = dfp.loc[future_ts, TARGET_CALLS]

    # ===== TMO por hora (incluye 'feriados' futuro si aplica) =====
    base_tmo = pd.DataFrame(index=future_ts)
    base_tmo[TARGET_CALLS] = pred_calls.values

    # Rellenos seguros desde el último histórico
    if {"proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico"}.issubset(df.columns):
        last_vals = df.ffill().iloc[[-1]][["proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico"]]
    else:
        last_vals = pd.DataFrame([[0,0,0,0]], columns=["proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico"])

    for c in ["proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico"]:
        base_tmo[c] = float(last_vals[c].iloc[0]) if c in last_vals.columns else 0.0

    # bandera de feriado hacia el futuro (si el training la usó)
    if "feriados" in df.columns:
        base_tmo["feriados"] = [ _is_holiday(ts, holidays_set) for ts in base_tmo.index ]

    base_tmo = add_time_parts(base_tmo)
    Xt = dummies_and_reindex(base_tmo, cols_tmo)
    y_tmo = m_tmo.predict(sc_tmo.transform(Xt), verbose=0).flatten()
    y_tmo = np.maximum(0, y_tmo)

    # ===== Erlang por hora =====
    df_hourly = pd.DataFrame(index=future_ts)
    df_hourly["calls"] = np.round(pred_calls).astype(int)
    df_hourly["tmo_s"] = np.round(y_tmo).astype(int)
    df_hourly["agents_prod"] = 0

    for ts in df_hourly.index:
        a, _ = required_agents(float(df_hourly.at[ts, "calls"]), float(df_hourly.at[ts, "tmo_s"]))
        df_hourly.at[ts, "agents_prod"] = int(a)

    df_hourly["agents_sched"] = df_hourly["agents_prod"].apply(schedule_agents)

    # ===== Salidas =====
    write_hourly_json(f"{PUBLIC_DIR}/prediccion_horaria.json",
                      df_hourly, "calls", "tmo_s", "agents_sched")
    write_daily_json(f"{PUBLIC_DIR}/prediccion_diaria.json",
                     df_hourly, "calls", "tmo_s")

    return df_hourly
