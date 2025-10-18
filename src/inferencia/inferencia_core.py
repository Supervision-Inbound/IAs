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

TARGET_CALLS = "recibidos_nacional"  # estandarizamos el nombre en main
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
    # Respetar tz de last_ts si la trae; si no, usar TZ
    tz = last_ts.tz if getattr(last_ts, "tz", None) is not None else TZ
    start = last_ts + pd.Timedelta(hours=1)
    return pd.date_range(start=start, periods=hours, freq="H", tz=tz)

# ---------------------------------------------------------------------
# Predicción iterativa de llamadas
# ---------------------------------------------------------------------
def _predict_calls_iterative(
    df_calls_ts: pd.DataFrame,
    model,
    scaler,
    train_cols: List[str],
    horizon_hours: int
) -> pd.Series:
    """
    Predicción iterativa 1-step-ahead sobre horizonte (horas).
    Mantiene escala/estacionalidad del entrenamiento (lags/MA).
    """
    df_hist = df_calls_ts.copy()
    df_hist = df_hist.sort_index()

    fut_idx = _make_future_index(df_hist.index.max(), horizon_hours // 24)

    out_vals = []
    current = df_hist[[TARGET_CALLS]].copy()

    for ts in fut_idx:
        # Construir features sobre histórico + "placeholder" de 1 fila
        tmp = pd.concat([current, pd.DataFrame(index=[ts])], axis=0)
        tmp = add_time_parts(tmp)
        tmp = add_lags_mas(tmp, TARGET_CALLS)

        x_step = dummies_and_reindex_with_scaler_means(
            tmp.tail(1), train_cols, scaler
        )
        x_step_s = scaler.transform(x_step)
        pred = float(model.predict(x_step_s, verbose=0).flatten()[0])

        # nunca negativo
        pred = max(0.0, pred)
        out_vals.append(pred)

        # actualizar histórico con la predicción para el siguiente paso
        current.loc[ts, TARGET_CALLS] = pred

    return pd.Series(out_vals, index=fut_idx, name=TARGET_CALLS)

# ---------------------------------------------------------------------
# Perfiles TMO desde histórico por tipo
# ---------------------------------------------------------------------
def _build_tmo_profiles(df_tmo_hist: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un DataFrame por (dow,hour) con perfiles medianos:
      proporcion_comercial, proporcion_tecnica, tmo_comercial, tmo_tecnico, tmo_general
    Si faltan columnas, usa valores neutros/ffill.
    """
    if df_tmo_hist is None or df_tmo_hist.empty:
        # Perfiles neutros
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

    # columnas esperadas (pueden no estar todas)
    cols_need = [
        "proporcion_comercial","proporcion_tecnica",
        "tmo_comercial","tmo_tecnico","tmo_general"
    ]
    for c in cols_need:
        if c not in d.columns:
            # valores fallback razonables
            if c.startswith("proporcion_"):
                d[c] = 0.5
            elif c.startswith("tmo_"):
                d[c] = 300.0

    grp = d.groupby(["dow","hour"])[cols_need].median().fillna(method="ffill").fillna(method="bfill")
    return grp

def _project_profiles_to_future(future_idx: pd.DatetimeIndex, profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Expande perfiles (dow,hour) al índice futuro horario.
    """
    fut = pd.DataFrame(index=future_idx)
    fut["dow"] = fut.index.dayofweek
    fut["hour"] = fut.index.hour
    fut = fut.join(profiles, on=["dow","hour"])
    # asegurar límites coherentes
    fut["proporcion_comercial"] = fut["proporcion_comercial"].clip(0,1).fillna(0.5)
    fut["proporcion_tecnica"]   = fut["proporcion_tecnica"].clip(0,1).fillna(0.5)
    for c in ["tmo_comercial","tmo_tecnico","tmo_general"]:
        fut[c] = fut[c].fillna(method="ffill").fillna(method="bfill").fillna(300.0)
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
# Forecast principal (120 días por defecto)
# ---------------------------------------------------------------------
def forecast_120d(
    df_hist_calls: pd.DataFrame,
    df_tmo_hist: pd.DataFrame,
    horizon_days: int = 120,
    holidays_set: Optional[set] = None,
) -> pd.DataFrame:
    """
    Entradas:
      - df_hist_calls: histórico horario de llamadas (con TARGET_CALLS y, opcionalmente, feriados/es_dia_de_pago)
      - df_tmo_hist:   histórico horario de TMO desagregado por tipo (comercial/técnico) para perfiles
      - horizon_days:  días a predecir
      - holidays_set:  set de fechas feriado (para inyectar feriados 0/1 si no vienen)

    Salidas:
      - Escribe:
          public/prediccion_horaria.json
          public/prediccion_diaria.json
      - Retorna DataFrame horario con columnas: calls, tmo_s, (y si hay) tmo_comercial/tecnico y proporciones
    """
    os.makedirs(PUBLIC_DIR, exist_ok=True)

    # 0) Normalizar índices (no convierte TZ si ya existe; ver features.ensure_ts)
    df_calls = ensure_ts(df_hist_calls)
    df_calls = df_calls.sort_index()

    # Si faltan columnas calendario, crearlas a partir de holidays_set
    if "feriados" not in df_calls.columns:
        if holidays_set:
            idx_dates = (df_calls.index.date if df_calls.index.tz is None
                         else df_calls.index.tz_convert(TZ).date)
            df_calls["feriados"] = pd.Series([int(d in holidays_set) for d in idx_dates], index=df_calls.index, dtype=int)
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

    # 2) Predicción iterativa de llamadas
    horizon_hours = horizon_days * 24
    pred_calls = _predict_calls_iterative(
        df_calls[[TARGET_CALLS]],
        m_planner,
        sc_planner,
        cols_planner,
        horizon_hours=horizon_hours
    )
    future_idx = pred_calls.index

    # 3) Perfiles TMO desde histórico por tipo (dow,hour) -> proyectados al futuro
    df_tmo_hist_ts = ensure_ts(df_tmo_hist) if df_tmo_hist is not None and not df_tmo_hist.empty else df_tmo_hist
    profiles = _build_tmo_profiles(df_tmo_hist_ts)
    fut_prof = _project_profiles_to_future(future_idx, profiles)

    # 4) Construcción de features para el modelo TMO
    # Base con calendario + llamadas (las llamadas influyen en AHT operativo)
    X_tmo_base = pd.DataFrame(index=future_idx)
    X_tmo_base["calls"] = pred_calls.values  # TARGET_CALLS futuro
    X_tmo_base["feriados"] = 0
    if holidays_set:
        dates = future_idx.tz_convert(TZ).date if future_idx.tz is not None else future_idx.date
        X_tmo_base["feriados"] = [int(d in holidays_set) for d in dates]
    dias = [1,2,15,16,29,30,31]
    X_tmo_base["es_dia_de_pago"] = (future_idx.day.isin(dias)).astype(int)

    # Unir perfiles (proporciones + tmo tipo)
    Xt = X_tmo_base.join(fut_prof)

    # Asegurar parts of time
    Xt = add_time_parts(Xt)

    # Algunas pipelines de entrenamiento incluyen dummies y columnas extras;
    # usamos dummies_and_reindex_with_scaler_means para rellenar faltantes con medias del scaler.
    Xt_ready = dummies_and_reindex_with_scaler_means(Xt, cols_tmo, sc_tmo)

    # 5) Predicción TMO horario + clip conservador
    y_tmo_raw = m_tmo.predict(sc_tmo.transform(Xt_ready), verbose=0).flatten()
    y_tmo = _clip_tmo_from_hist(y_tmo_raw, df_tmo_hist_ts if df_tmo_hist_ts is not None else pd.DataFrame())

    # 6) Dataframe final horario
    df_hourly = pd.DataFrame(index=future_idx)
    df_hourly["calls"] = np.round(pred_calls.values).astype(int)
    df_hourly["tmo_s"] = np.round(y_tmo).astype(int)

    # Si tenemos tmo por tipo, exportarlo también para trazabilidad
    if "tmo_comercial" in fut_prof.columns and "tmo_tecnico" in fut_prof.columns:
        df_hourly["tmo_comercial"] = np.round(fut_prof["tmo_comercial"].values).astype(int)
        df_hourly["tmo_tecnico"]   = np.round(fut_prof["tmo_tecnico"].values).astype(int)
    if "proporcion_comercial" in fut_prof.columns and "proporcion_tecnica" in fut_prof.columns:
        df_hourly["proporcion_comercial"] = np.round(fut_prof["proporcion_comercial"].values, 4)
        df_hourly["proporcion_tecnica"]   = np.round(fut_prof["proporcion_tecnica"].values, 4)

    # 7) Escribir salidas
    # Horaria (llamadas + tmo horario)
    hourly_out = (df_hourly.reset_index()
                  .rename(columns={"index":"ts"})
                  .assign(ts=lambda d: d["ts"].dt.strftime("%Y-%m-%d %H:%M:%S"))
                  .to_dict(orient="records"))
    write_json(os.path.join(PUBLIC_DIR, "prediccion_horaria.json"), hourly_out)

    # Diaria (tmo ponderado por llamadas) — usa util para mantener contrato de salida
    write_daily_json(
        os.path.join(PUBLIC_DIR, "prediccion_diaria.json"),
        df_hourly,
        col_calls="calls",
        col_tmo="tmo_s"
    )

    return df_hourly

