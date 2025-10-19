# src/inferencia_tmo/forecast.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from typing import Optional

from src.inferencia.features import (
    ensure_ts,
    add_time_parts,
    dummies_and_reindex_with_scaler_means,
)
from src.inferencia.utils_io import write_json, write_daily_json

TZ = "America/Santiago"
PUBLIC_DIR = "public"
MODELS_DIR = "models"

# Artefactos TMO
TMO_MODEL_PATH  = os.path.join(MODELS_DIR, "modelo_tmo.keras")
TMO_SCALER_PATH = os.path.join(MODELS_DIR, "scaler_tmo.pkl")
TMO_COLS_PATH   = os.path.join(MODELS_DIR, "training_columns_tmo.json")

COL_TMO_GEN = "tmo_general"

def _load_tmo():
    model = tf.keras.models.load_model(TMO_MODEL_PATH, compile=False)
    scaler = joblib.load(TMO_SCALER_PATH)
    with open(TMO_COLS_PATH, "r", encoding="utf-8") as f:
        cols = __import__("json").load(f)
    return model, scaler, cols

def _build_tmo_profiles(df_tmo_hist: pd.DataFrame) -> pd.DataFrame:
    if df_tmo_hist is None or df_tmo_hist.empty:
        idx = pd.MultiIndex.from_product([range(7), range(24)], names=["dow","hour"])
        return pd.DataFrame({
            "proporcion_comercial": 0.5,
            "proporcion_tecnica": 0.5,
            "tmo_comercial": 300.0,
            "tmo_tecnico": 300.0,
            "tmo_general": 300.0,
        }, index=idx)

    d = df_tmo_hist.copy()
    d = add_time_parts(d)

    cols_need = [
        "proporcion_comercial","proporcion_tecnica",
        "tmo_comercial","tmo_tecnico","tmo_general"
    ]
    for c in cols_need:
        if c not in d.columns:
            d[c] = 0.5 if c.startswith("proporcion_") else 300.0

    grp = (d.groupby(["dow","hour"])[cols_need]
             .median()
             .ffill()
             .bfill())
    return grp

def _project_profiles_to_future(future_idx: pd.DatetimeIndex, profiles: pd.DataFrame) -> pd.DataFrame:
    fut = pd.DataFrame(index=future_idx)
    fut["dow"] = fut.index.dayofweek
    fut["hour"] = fut.index.hour
    fut = fut.join(profiles, on=["dow","hour"])
    fut["proporcion_comercial"] = fut["proporcion_comercial"].clip(0,1).fillna(0.5)
    fut["proporcion_tecnica"]   = fut["proporcion_tecnica"].clip(0,1).fillna(0.5)
    for c in ["tmo_comercial","tmo_tecnico","tmo_general"]:
        fut[c] = fut[c].ffill().bfill().fillna(300.0)
    return fut.drop(columns=["dow","hour"])

def _clip_tmo_from_hist(y_raw: np.ndarray, df_tmo_hist: pd.DataFrame) -> np.ndarray:
    try:
        base = df_tmo_hist.get(COL_TMO_GEN, pd.Series([180]))
        p10 = float(np.nanpercentile(base, 10))
        p98 = float(np.nanpercentile(base, 98))
        floor = max(30.0, p10 * 0.8); cap = max(floor + 1, p98 * 1.1)
    except Exception:
        floor, cap = 60.0, 900.0
    return np.clip(y_raw, floor, cap)

def forecast_tmo(df_future_calls: pd.DataFrame,
                 df_tmo_hist: pd.DataFrame,
                 holidays_set: Optional[set] = None) -> pd.DataFrame:
    """
    Recibe DF de llamadas futuras (index horario, col 'calls')
    y DF histórico TMO (para perfiles por dow/hour).
    Devuelve DF horario con tmo_s (+ columnas de mezcla si hay).
    """
    os.makedirs(PUBLIC_DIR, exist_ok=True)

    # 0) Normalizar
    calls = df_future_calls.copy().sort_index()
    calls = calls[["calls"]].astype(float)
    future_idx = calls.index

    df_tmo_hist_ts = ensure_ts(df_tmo_hist) if df_tmo_hist is not None and not df_tmo_hist.empty else df_tmo_hist
    profiles = _build_tmo_profiles(df_tmo_hist_ts)
    fut_prof = _project_profiles_to_future(future_idx, profiles)

    # 1) Artefactos
    m, sc, cols = _load_tmo()

    # 2) Feature set para TMO (usa llamadas futuras + perfiles + time-parts)
    Xt = calls.join(fut_prof)
    Xt = add_time_parts(Xt)
    Xt_ready = dummies_and_reindex_with_scaler_means(Xt, cols, sc)

    # 3) Predicción + clip
    y_raw = m.predict(sc.transform(Xt_ready), verbose=0).flatten()
    y = _clip_tmo_from_hist(y_raw, df_tmo_hist_ts if df_tmo_hist_ts is not None else pd.DataFrame())

    df_tmo = pd.DataFrame(index=future_idx)
    df_tmo["tmo_s"] = np.round(y).astype(int)

    if "tmo_comercial" in fut_prof.columns and "tmo_tecnico" in fut_prof.columns:
        df_tmo["tmo_comercial"] = np.round(fut_prof["tmo_comercial"].values).astype(int)
        df_tmo["tmo_tecnico"]   = np.round(fut_prof["tmo_tecnico"].values).astype(int)
    if "proporcion_comercial" in fut_prof.columns and "proporcion_tecnica" in fut_prof.columns:
        df_tmo["proporcion_comercial"] = np.round(fut_prof["proporcion_comercial"].values, 4)
        df_tmo["proporcion_tecnica"]   = np.round(fut_prof["proporcion_tecnica"].values, 4)

    # 4) Salidas propias de TMO
    hourly_out = (df_tmo.reset_index()
                  .rename(columns={"index":"ts"})
                  .assign(ts=lambda d: d["ts"].dt.strftime("%Y-%m-%d %H:%M:%S"))
                  .to_dict(orient="records"))
    write_json(os.path.join(PUBLIC_DIR, "tmo_horaria.json"), hourly_out)

    # diario ponderado por llamadas del DF futuro
    merged = df_tmo.join(calls)
    write_daily_json(
        os.path.join(PUBLIC_DIR, "tmo_diaria.json"),
        merged.rename(columns={"calls":"calls"}),
        col_calls="calls",
        col_tmo="tmo_s"
    )
    return df_tmo
