# src/inferencia/inferencia_core.py
from __future__ import annotations
import os
import json
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
from src.inferencia.utils_io import write_daily_json, write_json

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
MODELS_DIR   = "models"
PUBLIC_DIR   = "public"
TZ           = "America/Santiago"

TARGET_CALLS = "recibidos_nacional"   # estandarizado desde main
COL_TMO_GEN  = "tmo_general"

# Artefactos
PLANNER_MODEL_PATH   = os.path.join(MODELS_DIR, "modelo_planner.keras")
PLANNER_SCALER_PATH  = os.path.join(MODELS_DIR, "scaler_planner.pkl")
PLANNER_COLS_PATH    = os.path.join(MODELS_DIR, "training_columns_planner.json")

TMO_MODEL_PATH       = os.path.join(MODELS_DIR, "modelo_tmo.keras")
TMO_SCALER_PATH      = os.path.join(MODELS_DIR, "scaler_tmo.pkl")
TMO_COLS_PATH        = os.path.join(MODELS_DIR, "training_columns_tmo.json")


# ---------------------------------------------------------------------
# Helpers de artefactos
# ---------------------------------------------------------------------
def _load_artifacts(model_path: str, scaler_path: str, cols_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró modelo: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"No se encontró scaler: {scaler_path}")
    if not os.path.exists(cols_path):
        raise FileNotFoundError(f"No se encontró json columnas: {cols_path}")

    model = tf.keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    with open(cols_path, "r", encoding="utf-8") as f:
        cols = json.load(f)
    return model, scaler, cols


# ---------------------------------------------------------------------
# Índice futuro
# ---------------------------------------------------------------------
def _make_future_index(last_ts: pd.Timestamp, horizon_days: int) -> pd.DatetimeIndex:
    hours = horizon_days * 24
    tz = last_ts.tz if getattr(last_ts, "tz", None) is not None else TZ
    start = last_ts + pd.Timedelta(hours=1)
    return pd.date_range(start=start, periods=hours, freq="h", tz=tz)


# ---------------------------------------------------------------------
# Predicción iterativa de llamadas (versión "SAFE": sin inyecciones extras)
# ---------------------------------------------------------------------
def _predict_calls_iterative_safe(
    df_calls_ts: pd.DataFrame,
    model,
    scaler,
    train_cols: List[str],
    horizon_hours: int,
) -> pd.Series:
    """
    Predicción iterativa 1-step-ahead sin tocar calendario ni imputar nada adicional.
    Esto reproduce el entorno que te daba buen nivel previamente:
      - Se construyen time-parts + lags/MA como en train
      - Se reindexa a training_columns con medias del scaler (para columnas faltantes)
      - NO se rellena ni modifica 'feriados' / 'es_dia_de_pago'
      - NO se hace MA(24) ni otras imputaciones que puedan aplanar la escala
    """
    hist = df_calls_ts.copy().sort_index()
    last_ts = hist.index.max()
    fut_idx = pd.date_range(start=last_ts + pd.Timedelta(hours=1),
                            periods=horizon_hours, freq="h",
                            tz=(last_ts.tz if getattr(last_ts, "tz", None) is not None else TZ))

    out = []
    current = hist[[TARGET_CALLS]].copy()  # SOLO el target: así era cuando funcionaba bien

    for ts in fut_idx:
        # placeholder
        current.loc[ts, TARGET_CALLS] = np.nan

        # features exactamente como en train
        tmp = add_time_parts(current)
        tmp = add_lags_mas(tmp, TARGET_CALLS)

        X = dummies_and_reindex_with_scaler_means(tmp.tail(1), train_cols, scaler)
        Xs = scaler.transform(X)

        pred = float(model.predict(Xs, verbose=0).flatten()[0])
        pred = max(0.0, pred)
        out.append(pred)
        current.at[ts, TARGET_CALLS] = pred

    return pd.Series(out, index=fut_idx, name=TARGET_CALLS)


# ---------------------------------------------------------------------
# Perfiles TMO desde histórico por tipo (para variación realista)
# ---------------------------------------------------------------------
def _build_tmo_profiles(df_tmo_hist: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve perfiles medianos por (dow,hour):
      proporcion_comercial, proporcion_tecnica, tmo_comercial, tmo_tecnico, tmo_general
    """
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
            if c.startswith("proporcion_"):
                d[c] = 0.5
            else:
                d[c] = 300.0

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


# ---------------------------------------------------------------------
# Clip conservador del TMO horario (evita suelos/techos absurdos)
# ---------------------------------------------------------------------
def _clip_tmo_from_hist(y_tmo_raw: np.ndarray, df_tmo_hist: pd.DataFrame) -> np.ndarray:
    try:
        base = df_tmo_hist.get(COL_TMO_GEN, pd.Series([180]))
        p10 = float(np.nanpercentile(base, 10))
        p98 = float(np.nanpercentile(base, 98))
        tmo_floor = max(30.0, p10 * 0.8)
        tmo_cap   = max(tmo_floor + 1, p98 * 1.1)
    except Exception:
        tmo_floor, tmo_cap = 60.0, 900.0
    return np.clip(y_tmo_raw, tmo_floor, tmo_cap)


# ---------------------------------------------------------------------
# Forecast principal
# ---------------------------------------------------------------------
def forecast_120d(
    df_hist_calls: pd.DataFrame,
    df_tmo_hist: pd.DataFrame,
    horizon_days: int = 120,
    holidays_set: Optional[set] = None,  # no lo usamos en llamadas SAFE
) -> pd.DataFrame:
    """
    Entradas:
      - df_hist_calls: histórico horario (TARGET_CALLS ya alineado por ensure_ts en main)
      - df_tmo_hist:   histórico TMO desagregado (para perfiles TMO)
    Salidas:
      - public/prediccion_horaria.json
      - public/prediccion_diaria.json
      - Retorna DataFrame horario con: calls, tmo_s, y si hay: tmo_*, proporciones
    """
    os.makedirs(PUBLIC_DIR, exist_ok=True)

    # 0) Normalizar índices; ensure_ts no forza tz_convert si ya hay tz
    df_calls = ensure_ts(df_hist_calls).sort_index()

    # 1) Cargar artefactos
    m_planner, sc_planner, cols_planner = _load_artifacts(
        PLANNER_MODEL_PATH, PLANNER_SCALER_PATH, PLANNER_COLS_PATH
    )
    m_tmo, sc_tmo, cols_tmo = _load_artifacts(
        TMO_MODEL_PATH, TMO_SCALER_PATH, TMO_COLS_PATH
    )

    # 2) Predicción iterativa de llamadas (SAFE: sin tocar calendario ni imputaciones)
    horizon_hours = horizon_days * 24
    base_for_iter = df_calls[[TARGET_CALLS]]
    pred_calls = _predict_calls_iterative_safe(
        base_for_iter,
        m_planner,
        sc_planner,
        cols_planner,
        horizon_hours=horizon_hours,
    )
    future_idx = pred_calls.index

    # 3) Perfiles TMO -> futuro
    df_tmo_hist_ts = ensure_ts(df_tmo_hist) if df_tmo_hist is not None and not df_tmo_hist.empty else df_tmo_hist
    profiles = _build_tmo_profiles(df_tmo_hist_ts)
    fut_prof = _project_profiles_to_future(future_idx, profiles)

    # 4) Features TMO (usa llamadas predichas + perfiles; SIN alterar planner)
    Xt = pd.DataFrame(index=future_idx)
    Xt["calls"] = pred_calls.values
    Xt = Xt.join(fut_prof)
    Xt = add_time_parts(Xt)
    Xt_ready = dummies_and_reindex_with_scaler_means(Xt, cols_tmo, sc_tmo)

    # 5) Predicción TMO horario + clip
    y_tmo_raw = m_tmo.predict(sc_tmo.transform(Xt_ready), verbose=0).flatten()
    y_tmo = _clip_tmo_from_hist(y_tmo_raw, df_tmo_hist_ts if df_tmo_hist_ts is not None else pd.DataFrame())

    # 6) Dataframe final horario
    df_hourly = pd.DataFrame(index=future_idx)
    df_hourly["calls"] = np.round(pred_calls.values).astype(int)
    df_hourly["tmo_s"] = np.round(y_tmo).astype(int)

    if "tmo_comercial" in fut_prof.columns and "tmo_tecnico" in fut_prof.columns:
        df_hourly["tmo_comercial"] = np.round(fut_prof["tmo_comercial"].values).astype(int)
        df_hourly["tmo_tecnico"]   = np.round(fut_prof["tmo_tecnico"].values).astype(int)
    if "proporcion_comercial" in fut_prof.columns and "proporcion_tecnica" in fut_prof.columns:
        df_hourly["proporcion_comercial"] = np.round(fut_prof["proporcion_comercial"].values, 4)
        df_hourly["proporcion_tecnica"]   = np.round(fut_prof["proporcion_tecnica"].values, 4)

    # 7) Salidas
    hourly_out = (df_hourly.reset_index()
                  .rename(columns={"index":"ts"})
                  .assign(ts=lambda d: d["ts"].dt.strftime("%Y-%m-%d %H:%M:%S"))
                  .to_dict(orient="records"))
    write_json(os.path.join(PUBLIC_DIR, "prediccion_horaria.json"), hourly_out)

    write_daily_json(
        os.path.join(PUBLIC_DIR, "prediccion_diaria.json"),
        df_hourly,
        col_calls="calls",
        col_tmo="tmo_s"
    )

    return df_hourly
