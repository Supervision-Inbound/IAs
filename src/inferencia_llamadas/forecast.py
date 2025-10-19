# src/inferencia_llamadas/forecast.py
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from typing import List, Optional

from src.inferencia.features import (
    ensure_ts,
    add_time_parts,
    add_lags_mas,
    dummies_and_reindex_with_scaler_means,
)
from src.inferencia.utils_io import write_json, write_daily_json

TZ = "America/Santiago"
PUBLIC_DIR = "public"
MODELS_DIR = "models"

TARGET_CALLS = "recibidos_nacional"

PLANNER_MODEL_PATH  = os.path.join(MODELS_DIR, "modelo_planner.keras")
PLANNER_SCALER_PATH = os.path.join(MODELS_DIR, "scaler_planner.pkl")
PLANNER_COLS_PATH   = os.path.join(MODELS_DIR, "training_columns_planner.json")

def _load_planner():
    model = tf.keras.models.load_model(PLANNER_MODEL_PATH, compile=False)
    scaler = joblib.load(PLANNER_SCALER_PATH)
    with open(PLANNER_COLS_PATH, "r", encoding="utf-8") as f:
        cols = json.load(f)
    return model, scaler, cols

def _future_index(last_ts: pd.Timestamp, horizon_days: int) -> pd.DatetimeIndex:
    hours = horizon_days * 24
    tz = last_ts.tz if getattr(last_ts, "tz", None) is not None else TZ
    start = last_ts + pd.Timedelta(hours=1)
    return pd.date_range(start=start, periods=hours, freq="h", tz=tz)

def forecast_llamadas(df_hist_calls: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """
    Devuelve DF horario con columna 'calls' (enteros).
    NO toca TMO. NO mete feriados ni es_dia_de_pago si no estaban en train.
    """
    os.makedirs(PUBLIC_DIR, exist_ok=True)

    # 0) Normalizar histórico y target
    d = ensure_ts(df_hist_calls).sort_index()
    if TARGET_CALLS not in d.columns:
        # renombrados compatibles
        for c in ["recibidos_nacional", "recibidos", "total_llamadas", "llamadas"]:
            if c in d.columns:
                d = d.rename(columns={c: TARGET_CALLS})
                break
    d = d[[TARGET_CALLS]].dropna()

    # 1) Artefactos
    m, sc, cols = _load_planner()

    # 2) Iterativo "seguro": sólo lags/MA + time-parts -> EXACTO a entrenamiento
    last_ts = d.index.max()
    fut_idx = _future_index(last_ts, horizon_days)
    current = d.copy()
    out = []

    for ts in fut_idx:
        current.loc[ts, TARGET_CALLS] = np.nan
        tmp = add_time_parts(current)
        tmp = add_lags_mas(tmp, TARGET_CALLS)

        X = dummies_and_reindex_with_scaler_means(tmp.tail(1), cols, sc)
        Xs = sc.transform(X)

        y = float(m.predict(Xs, verbose=0).flatten()[0])
        y = max(0.0, y)
        out.append(y)
        current.at[ts, TARGET_CALLS] = y

    pred = pd.Series(out, index=fut_idx, name="calls").round().astype(int)
    df_hourly = pd.DataFrame({"calls": pred}, index=fut_idx)

    # 3) Salidas JSON (sólo llamadas)
    hourly_out = (df_hourly.reset_index()
                  .rename(columns={"index": "ts"})
                  .assign(ts=lambda x: x["ts"].dt.strftime("%Y-%m-%d %H:%M:%S"))
                  .to_dict(orient="records"))
    write_json(os.path.join(PUBLIC_DIR, "llamadas_horaria.json"), hourly_out)

    write_daily_json(
        os.path.join(PUBLIC_DIR, "llamadas_diaria.json"),
        df_hourly,
        col_calls="calls",
        col_tmo="calls"  # dummy; write_daily_json ignora col_tmo si no existe mezcla
    )

    # Debug rápido de escala
    try:
        hist_7d = d[TARGET_CALLS].tail(24*7)
        stats = {
            "hist_hourly_mean_last_7d": float(hist_7d.mean()) if len(hist_7d) else None,
            "pred_hourly_mean": float(df_hourly["calls"].mean()),
            "pred_daily_mean": float(df_hourly["calls"].values.reshape(-1,24).sum(axis=1).mean()) if (len(df_hourly) % 24 == 0) else None
        }
        write_json(os.path.join(PUBLIC_DIR, "debug_calls_only_stats.json"), stats)
    except Exception:
        pass

    return df_hourly
