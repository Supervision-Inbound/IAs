# src/inferencia/inferencia_core.py
import json, joblib, numpy as np, pandas as pd, tensorflow as tf
from pathlib import Path
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

def _load_cols(path):
    with open(path,"r") as f: return json.load(f)

def forecast_120d(df_hist_calls: pd.DataFrame, horizon_days: int = 120):
    # Cargamos artefactos
    m_pl = tf.keras.models.load_model(PLANNER_MODEL, compile=False)
    sc_pl = joblib.load(PLANNER_SCALER)
    cols_pl = _load_cols(PLANNER_COLS)

    m_tmo = tf.keras.models.load_model(TMO_MODEL, compile=False)
    sc_tmo = joblib.load(TMO_SCALER)
    cols_tmo = _load_cols(TMO_COLS)

    # Base histórica (mínimo: ts, recibidos_nacional, feriados, es_dia_de_pago si existe)
    df = ensure_ts(df_hist_calls)
    if TARGET_CALLS not in df.columns:
        raise ValueError(f"Falta columna {TARGET_CALLS} en historical_data.csv")

    # forward-fill de auxiliares
    for aux in ["feriados","es_dia_de_pago","tmo_comercial","tmo_tecnico","proporcion_comercial","proporcion_tecnica"]:
        if aux in df.columns:
            df[aux] = df[aux].ffill()

    # Horizonte futuro
    start = df.index.max() + pd.Timedelta(hours=1)
    future_ts = pd.date_range(start, periods=horizon_days*24, freq="H", tz=TIMEZONE)

    # ===== Planner iterativo =====
    dfp = df[[TARGET_CALLS,"feriados"]].copy() if "feriados" in df.columns else df[[TARGET_CALLS]].copy()
    dfp[TARGET_CALLS] = dfp[TARGET_CALLS].astype(float)
    for ts in future_ts:
        tmp = pd.concat([dfp, pd.DataFrame(index=[ts])])
        tmp[TARGET_CALLS] = tmp[TARGET_CALLS].ffill()
        tmp = add_lags_mas(tmp, TARGET_CALLS)
        tmp = add_time_parts(tmp)
        X = dummies_and_reindex(tmp.tail(1), cols_pl)
        yhat = float(m_pl.predict(sc_pl.transform(X), verbose=0).flatten()[0])
        dfp.loc[ts, TARGET_CALLS] = max(0.0, yhat)

    pred_calls = dfp.loc[future_ts, TARGET_CALLS]

    # ===== TMO por hora =====
    base_tmo = pd.DataFrame(index=future_ts)
    base_tmo[TARGET_CALLS] = pred_calls.values
    # usar últimos valores conocidos para columnas necesarias si no están
    last_vals = df.ffill().iloc[[-1]].reindex(columns=["proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico"], fill_value=0)
    for c in last_vals.columns:
        base_tmo[c] = float(last_vals[c].iloc[0]) if c in last_vals else 0.0
    if "feriados" in df.columns: base_tmo["feriados"] = 0

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
        a,_ = required_agents(float(df_hourly.at[ts,"calls"]), float(df_hourly.at[ts,"tmo_s"]))
        df_hourly.at[ts,"agents_prod"] = a
    df_hourly["agents_sched"] = df_hourly["agents_prod"].apply(schedule_agents)

    # Output JSONs
    write_hourly_json(f"{PUBLIC_DIR}/prediccion_horaria.json", df_hourly, "calls", "tmo_s", "agents_sched")
    write_daily_json(f"{PUBLIC_DIR}/prediccion_diaria.json", df_hourly, "calls", "tmo_s")
    return df_hourly

