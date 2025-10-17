# src/inferencia/inferencia_core.py
import json, joblib, numpy as np, pandas as pd, tensorflow as tf
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

# Usar ventana reciente para construir lags/MA (coincide con “últimos ~90 días”)
HIST_WINDOW_DAYS = 90

def _load_cols(path):
    with open(path,"r") as f:
        return json.load(f)

def forecast_120d(df_hist_calls: pd.DataFrame, horizon_days: int = 120):
    """
    1) Usa SOLO histórico válido (llamadas > 0 preferente) y corta cualquier fila futura.
    2) Construye features a partir de una ventana reciente (90d) para lags/MA.
    3) Planner iterativo → calls por hora.
    4) TMO hora (con rellenos seguros).
    5) Erlang C → agentes programados.
    6) Salidas JSON horaria y diaria.
    """
    # ===== Artefactos =====
    m_pl = tf.keras.models.load_model(PLANNER_MODEL, compile=False)
    sc_pl = joblib.load(PLANNER_SCALER)
    cols_pl = _load_cols(PLANNER_COLS)

    m_tmo = tf.keras.models.load_model(TMO_MODEL, compile=False)
    sc_tmo = joblib.load(TMO_SCALER)
    cols_tmo = _load_cols(TMO_COLS)

    # ===== Base histórica =====
    df = ensure_ts(df_hist_calls)
    if TARGET_CALLS not in df.columns:
        raise ValueError(f"Falta columna {TARGET_CALLS} en historical_data.csv")

    # Filtrado robusto de llamadas válidas (coherente con main.py)
    df[TARGET_CALLS] = pd.to_numeric(df[TARGET_CALLS], errors="coerce")
    mask_valid = df[TARGET_CALLS].notna() & (df[TARGET_CALLS] > 0)
    if not mask_valid.any():
        mask_valid = df[TARGET_CALLS].notna()
    df = df.loc[mask_valid]

    # forward-fill de auxiliares comunes
    for aux in ["feriados","es_dia_de_pago","tmo_comercial","tmo_tecnico","proporcion_comercial","proporcion_tecnica"]:
        if aux in df.columns:
            df[aux] = df[aux].ffill()

    # ===== Horizonte futuro (justo después de la última fila válida) =====
    last_ts = df.index.max()
    # Asegura que no usamos filas que el CSV tenga más allá de last_ts
    df = df.loc[:last_ts]

    # Usar solo una ventana reciente para construir lags/MA
    start_hist = last_ts - pd.Timedelta(days=HIST_WINDOW_DAYS)
    df_recent = df.loc[df.index >= start_hist].copy()
    if df_recent.empty:
        # fallback: usa todo df
        df_recent = df.copy()

    # Fechas futuras a predecir
    future_ts = pd.date_range(last_ts + pd.Timedelta(hours=1),
                              periods=horizon_days*24, freq="h", tz=TIMEZONE)

    # ===== Planner iterativo (sobre df_recent) =====
    if "feriados" in df_recent.columns:
        dfp = df_recent[[TARGET_CALLS,"feriados"]].copy()
    else:
        dfp = df_recent[[TARGET_CALLS]].copy()

    dfp[TARGET_CALLS] = pd.to_numeric(dfp[TARGET_CALLS], errors="coerce").ffill().fillna(0.0)

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

    # Rellenos seguros desde el último histórico
    last_vals = df.ffill().iloc[[-1]].reindex(
        columns=["proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico"], 
        fill_value=0
    )
    for c in ["proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico"]:
        base_tmo[c] = float(last_vals[c].iloc[0]) if c in last_vals.columns else 0.0

    if "feriados" in df.columns:
        base_tmo["feriados"] = 0  # hacia futuro

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
        df_hourly.at[ts,"agents_prod"] = int(a)

    df_hourly["agents_sched"] = df_hourly["agents_prod"].apply(schedule_agents)

    # ===== Salidas =====
    write_hourly_json(f"{PUBLIC_DIR}/prediccion_horaria.json", df_hourly, "calls", "tmo_s", "agents_sched")
    write_daily_json(f"{PUBLIC_DIR}/prediccion_diaria.json", df_hourly, "calls", "tmo_s")

    return df_hourly

