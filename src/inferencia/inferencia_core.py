# src/inferencia/inferencia_core.py
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from typing import Tuple, List, Optional

from src.inferencia.features import (
    ensure_ts,
    add_time_parts,
    add_lags_mas,
    dummies_and_reindex,
    dummies_and_reindex_with_scaler_means,
)
from src.inferencia.utils_io import write_daily_json, write_json

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
MODELS_DIR   = "models"
PUBLIC_DIR   = "public"
TZ           = "America/Santiago"

TARGET_CALLS = "recibidos_nacional"  # estandarizado desde main
COL_TMO_GEN  = "tmo_general"

# Nombres de artefactos
PLANNER_MODEL_PATH   = os.path.join(MODELS_DIR, "modelo_planner.keras")
PLANNER_SCALER_PATH  = os.path.join(MODELS_DIR, "scaler_planner.pkl")
PLANNER_COLS_PATH    = os.path.join(MODELS_DIR, "training_columns_planner.json")

TMO_MODEL_PATH       = os.path.join(MODELS_DIR, "modelo_tmo.keras")
TMO_SCALER_PATH      = os.path.join(MODELS_DIR, "scaler_tmo.pkl")
TMO_COLS_PATH        = os.path.join(MODELS_DIR, "training_columns_tmo.json")

# ---------------------------------------------------------------------
# Helpers de carga de artefactos
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
    # usar 'h' para evitar FutureWarning
    return pd.date_range(start=start, periods=hours, freq="h", tz=tz)

# ---------------------------------------------------------------------
# Predicción iterativa de llamadas (inyectando calendario en cada paso)
# ---------------------------------------------------------------------
def _predict_calls_iterative(
    df_calls_ts: pd.DataFrame,
    model,
    scaler,
    train_cols: List[str],
    horizon_hours: int,
    holidays_set: Optional[set] = None,
) -> pd.Series:
    """
    Predicción iterativa 1-step-ahead sobre el horizonte (horas).
    Inyecta 'feriados' y 'es_dia_de_pago' en cada paso, además de time-parts y lags/MA,
    para replicar el entorno de entrenamiento del planner.
    """
    df_hist = df_calls_ts.copy().sort_index()

    # índice futuro con misma TZ que el histórico
    last_ts = df_hist.index.max()
    tz = last_ts.tz if getattr(last_ts, "tz", None) is not None else TZ
    fut_idx = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=horizon_hours, freq="h", tz=tz)

    # columnas disponibles en el histórico (por si no están ambas)
    have_fer = "feriados" in df_hist.columns
    have_pay = "es_dia_de_pago" in df_hist.columns

    # base mínima
    use_cols = [TARGET_CALLS] + ([ "feriados"] if have_fer else []) + ([ "es_dia_de_pago"] if have_pay else [])
    current = df_hist[use_cols].copy()

    dias_pago = {1, 2, 15, 16, 29, 30, 31}
    out_vals = []

    for ts in fut_idx:
        # 1) placeholder de fila futura con calendario
        row = {}
        if have_fer:
            d = ts.tz_convert(TZ).date if ts.tz is not None else ts.date()
            row["feriados"] = int(d in holidays_set) if holidays_set is not None else 0
        if have_pay:
            row["es_dia_de_pago"] = int(ts.day in dias_pago)

        # anexar fila
        for k, v in row.items():
            current.loc[ts, k] = v
        # target placeholder
        current.loc[ts, TARGET_CALLS] = np.nan

        # 2) features de tiempo + lags/MA
        tmp = add_time_parts(current)
        tmp = add_lags_mas(tmp, TARGET_CALLS)

        # 3) X conforme a entrenamiento (relleno robusto con medias del scaler si aplica)
        X_step = dummies_and_reindex_with_scaler_means(tmp.tail(1), train_cols, scaler)
        Xs = scaler.transform(X_step)

        # 4) predecir y “cerrar” el paso
        pred = float(model.predict(Xs, verbose=0).flatten()[0])
        pred = max(0.0, pred)
        out_vals.append(pred)
        current.at[ts, TARGET_CALLS] = pred

    return pd.Series(out_vals, index=fut_idx, name=TARGET_CALLS)

# ---------------------------------------------------------------------
# Perfiles TMO desde histórico por tipo
# ---------------------------------------------------------------------
def _build_tmo_profiles(df_tmo_hist: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un DataFrame por (dow,hour) con perfiles medianos:
      proporcion_comercial, proporcion_tecnica, tmo_comercial, tmo_tecnico, tmo_general
    Si faltan columnas, usa valores neutros/ffill/bfill.
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
            elif c.startswith("tmo_"):
                d[c] = 300.0

    grp = (d.groupby(["dow","hour"])[cols_need]
             .median()
             .ffill()
             .bfill())
    return grp

def _project_profiles_to_future(future_idx: pd.DatetimeIndex, profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Expande perfiles (dow,hour) al índice futuro horario.
    """
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
# Clip conservador del TMO horario (evita “suelo”/“techo” absurdos)
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
    holidays_set: Optional[set] = None,
) -> pd.DataFrame:
    """
    Entradas:
      - df_hist_calls: histórico horario de llamadas (con TARGET_CALLS y opcionalmente feriados/es_dia_de_pago)
      - df_tmo_hist:   histórico horario de TMO desagregado por tipo (para perfiles)
      - horizon_days:  días a predecir
      - holidays_set:  set de fechas feriado (para inyectar feriados 0/1 si no vienen)

    Salidas:
      - Escribe:
          public/prediccion_horaria.json
          public/prediccion_diaria.json
      - Retorna DataFrame horario con columnas: calls, tmo_s, y (si hay) tmo_* y proporciones
    """
    os.makedirs(PUBLIC_DIR, exist_ok=True)

    # 0) Normalizar índices (ensure_ts no convierte TZ si ya existe)
    df_calls = ensure_ts(df_hist_calls).sort_index()

    # Si faltan columnas calendario, crearlas a partir de holidays_set
    if "feriados" not in df_calls.columns:
        if holidays_set:
            idx_dates = (df_calls.index.date if df_calls.index.tz is None
                         else df_calls.index.tz_convert(TZ).date)
            df_calls["feriados"] = pd.Series([int(d in holidays_set) for d in idx_dates],
                                             index=df_calls.index, dtype=int)
        else:
            df_calls["feriados"] = 0
    if "es_dia_de_pago" not in df_calls.columns:
        dias = [1,2,15,16,29,30,31]
        df_calls["es_dia_de_pago"] = (df_calls.index.day.isin(dias)).astype(int)

    # 1) Cargar artefactos
    m_planner, sc_planner, cols_planner = _load_artifacts(
        PLANNER_MODEL_PATH, PLANNER_SCALER_PATH, PLANNER_COLS_PATH
    )
    m_tmo, sc_tmo, cols_tmo = _load_artifacts(
        TMO_MODEL_PATH, TMO_SCALER_PATH, TMO_COLS_PATH
    )

    # 2) Predicción iterativa de llamadas (inyectando calendario)
    horizon_hours = horizon_days * 24
    base_for_iter = (df_calls[[TARGET_CALLS, "feriados", "es_dia_de_pago"]]
                     if all(c in df_calls.columns for c in ["feriados","es_dia_de_pago"])
                     else df_calls[[TARGET_CALLS]])
    pred_calls = _predict_calls_iterative(
        base_for_iter,
        m_planner,
        sc_planner,
        cols_planner,
        horizon_hours=horizon_hours,
        holidays_set=holidays_set,
    )
    future_idx = pred_calls.index

    # 3) Perfiles TMO desde histórico por tipo (dow,hour) -> proyectados al futuro
    df_tmo_hist_ts = ensure_ts(df_tmo_hist) if df_tmo_hist is not None and not df_tmo_hist.empty else df_tmo_hist
    profiles = _build_tmo_profiles(df_tmo_hist_ts)
    fut_prof = _project_profiles_to_future(future_idx, profiles)

    # 4) Construcción de features para el modelo TMO
    X_tmo_base = pd.DataFrame(index=future_idx)
    X_tmo_base["calls"] = pred_calls.values
    X_tmo_base["feriados"] = 0
    if holidays_set:
        dates = future_idx.tz_convert(TZ).date if future_idx.tz is not None else future_idx.date
        X_tmo_base["feriados"] = [int(d in holidays_set) for d in dates]
    dias = [1,2,15,16,29,30,31]
    X_tmo_base["es_dia_de_pago"] = (future_idx.day.isin(dias)).astype(int)

    Xt = X_tmo_base.join(fut_prof)
    Xt = add_time_parts(Xt)
    Xt_ready = dummies_and_reindex_with_scaler_means(Xt, cols_tmo, sc_tmo)

    # 5) Predicción TMO horario + clip conservador
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

    # 7) Escribir salidas
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


