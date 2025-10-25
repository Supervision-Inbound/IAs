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

# Artefactos (entrenados previamente)
PLANNER_MODEL = "models/modelo_planner.keras"
PLANNER_SCALER = "models/scaler_planner.pkl"
PLANNER_COLS = "models/training_columns_planner.json"

TMO_MODEL = "models/modelo_tmo.keras"
TMO_SCALER = "models/scaler_tmo.pkl"
TMO_COLS = "models/training_columns_tmo.json"

TARGET_CALLS = "recibidos_nacional"
TARGET_TMO = "tmo_general"


# ========= Helpers de carga =========
def _load_json_cols(path):
    with open(path, "r") as f:
        return json.load(f)


# ========= Helpers de FERIADOS =========
def _safe_ratio(num, den, fallback=1.0):
    num = float(num) if num is not None and not np.isnan(num) else np.nan
    den = float(den) if den is not None and not np.isnan(den) and den != 0 else np.nan
    if np.isnan(num) or np.isnan(den) or den == 0:
        return fallback
    return num / den


def _series_is_holiday(idx, holidays_set):
    # idx es un DatetimeIndex (con o sin tz)
    as_local = idx.tz_localize("UTC").tz_convert(TIMEZONE) if idx.tz is None else idx.tz_convert(TIMEZONE)
    return as_local.date.astype("object").isin(holidays_set).astype(int)


# ========= INFERENCIA PRINCIPAL =========
def forecast_120d(df_hist_joined: pd.DataFrame,
                  df_hist_tmo_only: pd.DataFrame | None,
                  horizon_days: int = 120,
                  holidays_set: set | None = None) -> pd.DataFrame:
    """
    Genera pronóstico horario para 'horizon_days' días.
    - Primero predice CALLS con el planner (sin tocar su lógica).
    - Luego predice TMO en forma AUTORREGRESIVA (lags/MA de TMO + exógenas).
    - Finalmente calcula agentes (Erlang) y escribe JSONs.

    df_hist_joined: histórico con al menos ts, recibidos_nacional, feriados
    df_hist_tmo_only: histórico puro de TMO (para AR), puede incluir proporciones/otros TMO
    """

    # ===== 0) Cargar modelos/escálers/columnas =====
    m_pl = tf.keras.models.load_model(PLANNER_MODEL)
    sc_pl = joblib.load(PLANNER_SCALER)
    cols_pl = _load_json_cols(PLANNER_COLS)

    m_tmo = tf.keras.models.load_model(TMO_MODEL)
    sc_tmo = joblib.load(TMO_SCALER)
    cols_tmo = _load_json_cols(TMO_COLS)

    # ===== 1) Base histórica =====
    df = ensure_ts(df_hist_joined.copy())
    df = df.sort_values("ts").set_index("ts")
    # columnas mínimas
    if TARGET_CALLS not in df.columns:
        raise ValueError(f"'{TARGET_CALLS}' no está en el histórico.")
    if "feriados" not in df.columns:
        df["feriados"] = 0

    # Si traemos TMO puro, lo aplicamos como sobrescritura (sin romper otras columnas)
    if df_hist_tmo_only is not None and not df_hist_tmo_only.empty:
        try:
            tmo_pure = ensure_ts(df_hist_tmo_only.copy()).sort_values("ts").set_index("ts")
            # Normaliza nombre por si viene 'tmo_general'
            if TARGET_TMO not in tmo_pure.columns and "tmo_general" in tmo_pure.columns:
                tmo_pure = tmo_pure.rename(columns={"tmo_general": TARGET_TMO})
            df.update(tmo_pure, overwrite=True)
        except Exception as e:
            print(f"[WARN] No se pudo aplicar TMO puro: {e}")

    # ===== 2) Ventana base para iterar (últimos 180d por seguridad) =====
    last_ts = df.index.max()
    freq = pd.infer_freq(df.index) or "H"
    future_idx = pd.date_range(start=last_ts + pd.Timedelta(hours=1),
                               periods=horizon_days * 24, freq="H", tz=None)

    # Preparamos DataFrame pivote (hist + futuro)
    dfp = df.copy()
    if TARGET_TMO not in dfp.columns:
        dfp[TARGET_TMO] = np.nan

    # ratios internos (si existen)
    if "q_llamadas_general" in dfp.columns:
        dfp["proporcion_comercial"] = dfp.get("q_llamadas_comercial", 0) / (dfp["q_llamadas_general"] + 1e-6)
        dfp["proporcion_tecnica"] = dfp.get("q_llamadas_tecnico", 0) / (dfp["q_llamadas_general"] + 1e-6)

    # ===== 3) Iteración en el tiempo: CALLS -> TMO (AR) =====
    # Flags calendario (en inferencia forzamos pago=0 si no corresponde; feriados vienen del set)
    def _ensure_calendar_block(tmp_df):
        tmp_df = add_time_parts(tmp_df.reset_index().rename(columns={"index": "ts"}))
        tmp_df = tmp_df.set_index("ts")
        tmp_df["es_dia_de_pago"] = tmp_df["day"].isin([1, 2, 15, 16, 29, 30, 31]).astype(int)
        if holidays_set:
            tmp_df["feriados"] = _series_is_holiday(tmp_df.index, holidays_set)
        else:
            tmp_df["feriados"] = tmp_df.get("feriados", 0).fillna(0).astype(int)
        return tmp_df

    # Proceso iterativo
    for ts in future_idx:
        # expandimos una fila en blanco
        if ts not in dfp.index:
            dfp.loc[ts, :] = np.nan

        # completar calendario y feriados
        tmp = dfp.loc[:ts].tail(24 * 200)  # mantener la ventana razonable
        tmp = _ensure_calendar_block(tmp)

        # Lags/MA de CALLS (sin tocar lógica del planner)
        tmp_calls = add_lags_mas(tmp, TARGET_CALLS)

        # ---- 1) Predecir CALLS con planner ----
        row_pl = tmp_calls.tail(1)  # la fila en ts
        X_pl = dummies_and_reindex(row_pl, cols_pl, dummies_cols=['dow', 'month', 'hour'])
        X_pl_s = sc_pl.transform(X_pl)
        yhat_calls = float(m_pl.predict(X_pl_s, verbose=0).flatten()[0])
        yhat_calls = max(0.0, yhat_calls)
        dfp.at[ts, TARGET_CALLS] = yhat_calls

        # ---- 2) Preparar features TMO (AR sobre TMO + exógenas) ----
        # Insertamos la predicción de llamadas en la fila actual
        tmp.loc[ts, TARGET_CALLS] = yhat_calls

        # Construimos lags/MA de TMO
        for lag in [24, 48, 72, 168]:
            tmp[f'lag_tmo_{lag}'] = tmp[TARGET_TMO].shift(lag)
        for window in [24, 72, 168]:
            tmp[f'ma_tmo_{window}'] = tmp[TARGET_TMO].rolling(window, min_periods=1).mean()

        # Partes de tiempo ya están (add_time_parts en _ensure_calendar_block)
        # Interacción opcional
        if "precipitacion" in tmp.columns:
            # Definir es_dia_habil para la interacción
            tmp["es_dia_habil"] = ((tmp["dow"].isin([0, 1, 2, 3, 4])) & (tmp["feriados"] == 0)).astype(int)
            tmp["precipitacion_x_dia_habil"] = tmp["precipitacion"] * tmp["es_dia_habil"]

        # Fila actual para TMO
        row_tmo = tmp.tail(1)

        # Reindex a columnas esperadas por el modelo TMO
        X_tmo = dummies_and_reindex(row_tmo, cols_tmo, dummies_cols=['dow', 'month', 'hour'])
        X_tmo_s = sc_tmo.transform(X_tmo)

        # ---- 3) Predecir TMO ----
        yhat_tmo = float(m_tmo.predict(X_tmo_s, verbose=0).flatten()[0])
        yhat_tmo = max(0.0, yhat_tmo)
        dfp.at[ts, TARGET_TMO] = yhat_tmo

    # ===== 4) Ensamblar salida horaria =====
    out = dfp.loc[future_idx].copy()
    out = _ensure_calendar_block(out)
    out = out.rename(columns={TARGET_CALLS: "calls", TARGET_TMO: "tmo_s"})
    # Completar faltantes por si algún merge previo dejó NaN
    out["calls"] = out["calls"].fillna(0.0)
    out["tmo_s"] = out["tmo_s"].fillna(method="ffill").fillna(out["tmo_s"].median() if not out["tmo_s"].isna().all() else 180.0)

    # ===== 5) Agentes (Erlang) =====
    df_hourly = out[["calls", "tmo_s"]].copy()
    df_hourly["agents_prod"] = 0
    for ts in df_hourly.index:
        a, _ = required_agents(float(df_hourly.at[ts, "calls"]), float(df_hourly.at[ts, "tmo_s"]))
        df_hourly.at[ts, "agents_prod"] = int(a)
    df_hourly["agents_sched"] = df_hourly["agents_prod"].apply(schedule_agents)

    # ===== 6) Salidas JSON =====
    write_hourly_json(f"{PUBLIC_DIR}/prediccion_horaria.json",
                      df_hourly, "calls", "tmo_s", "agents_sched")
    write_daily_json(f"{PUBLIC_DIR}/prediccion_diaria.json",
                     df_hourly, "calls", "tmo_s")

    return df_hourly
