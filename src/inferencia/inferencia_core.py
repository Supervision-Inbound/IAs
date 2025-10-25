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

# Artefactos entrenados (planner + tmo)
PLANNER_MODEL = "models/modelo_planner.keras"
PLANNER_SCALER = "models/scaler_planner.pkl"
PLANNER_COLS = "models/training_columns_planner.json"

TMO_MODEL = "models/modelo_tmo.keras"
TMO_SCALER = "models/scaler_tmo.pkl"
TMO_COLS = "models/training_columns_tmo.json"

TARGET_CALLS = "recibidos_nacional"
TARGET_TMO = "tmo_general"


def _load_json_cols(path):
    with open(path, "r") as f:
        return json.load(f)


def _series_is_holiday(idx: pd.DatetimeIndex, holidays_set: set) -> pd.Series:
    """
    Devuelve una Series (0/1) indexada por idx indicando si cada fecha es feriado.
    Soporta idx con o sin tz; normaliza todo a TIMEZONE.
    """
    # Asegurar tz en TIMEZONE
    if idx.tz is None:
        as_local = idx.tz_localize(TIMEZONE)
    else:
        as_local = idx.tz_convert(TIMEZONE)
    # DatetimeIndex.date -> ndarray de datetime.date; envolver en Index para usar .isin
    dates_idx = pd.Index(as_local.date, name="date")
    mask = dates_idx.isin(holidays_set)  # boolean np.ndarray-like
    return pd.Series(mask.astype(int), index=idx)


def forecast_120d(df_hist_joined: pd.DataFrame,
                  df_hist_tmo_only: pd.DataFrame,
                  horizon_days: int = 120,
                  holidays_set: set | None = None) -> pd.DataFrame:
    """
    Genera pronóstico horario para 'horizon_days' días.
    - Planner: predice CALLS (misma lógica, mismos artefactos).
    - TMO: autoregresivo usando SOLO lags/MA de la serie TMO proveniente del archivo TMO.
      (las llamadas solo se usan como exógena; NUNCA alimentan el histórico TMO).
    - Luego calcula agentes (Erlang) y escribe JSONs.
    """

    # === 0) Artefactos ===
    m_pl = tf.keras.models.load_model(PLANNER_MODEL)
    sc_pl = joblib.load(PLANNER_SCALER)
    cols_pl = _load_json_cols(PLANNER_COLS)

    m_tmo = tf.keras.models.load_model(TMO_MODEL)
    sc_tmo = joblib.load(TMO_SCALER)
    cols_tmo = _load_json_cols(TMO_COLS)

    # === 1) Histórico de exógenas/llamadas ===
    hist = ensure_ts(df_hist_joined.copy()).sort_index()
    if TARGET_CALLS not in hist.columns:
        raise ValueError(f"'{TARGET_CALLS}' no está en el histórico.")
    if "feriados" not in hist.columns:
        hist["feriados"] = 0

    # === 2) Histórico TMO puro ===
    if df_hist_tmo_only is None or df_hist_tmo_only.empty:
        raise ValueError("Se requiere df_hist_tmo_only con el histórico TMO puro (archivo TMO).")
    tmo_base = ensure_ts(df_hist_tmo_only.copy()).sort_index()

    # Armamos DF de trabajo con:
    # - exógenas + llamadas desde 'hist'
    # - TMO únicamente desde 'tmo_base'
    dfp = hist.copy()
    # quitamos cualquier TMO accidental del histórico principal (para evitar contaminación)
    if TARGET_TMO in dfp.columns:
        dfp = dfp.drop(columns=[TARGET_TMO])
    # agregamos el TMO puro (clave)
    if TARGET_TMO in tmo_base.columns:
        dfp = dfp.join(tmo_base[[TARGET_TMO]], how="left")

    # sumamos proporciones y tmos desagregados si existen (no obligatorio)
    for col in ["proporcion_comercial", "proporcion_tecnica", "tmo_comercial", "tmo_tecnico",
                "q_llamadas_general", "q_llamadas_comercial", "q_llamadas_tecnico"]:
        if col in tmo_base.columns and col not in dfp.columns:
            dfp = dfp.join(tmo_base[[col]], how="left")

    last_ts = dfp.index.max()
    # 'h' en vez de 'H' para evitar FutureWarning
    future_idx = pd.date_range(start=last_ts + pd.Timedelta(hours=1),
                               periods=horizon_days * 24, freq="h")

    # === 3) Calendario ===
    def _ensure_calendar(tmp_df):
        tmp_df = add_time_parts(tmp_df.reset_index().rename(columns={"index": "ts"}))
        tmp_df = tmp_df.set_index("ts")
        tmp_df["es_dia_de_pago"] = tmp_df["day"].isin([1, 2, 15, 16, 29, 30, 31]).astype(int)
        if holidays_set:
            fer = _series_is_holiday(tmp_df.index, holidays_set)
            # Reindex por seguridad (aunque ya coincide)
            tmp_df["feriados"] = fer.reindex(tmp_df.index, fill_value=0).astype(int)
        else:
            tmp_df["feriados"] = tmp_df.get("feriados", 0).fillna(0).astype(int)
        return tmp_df

    # === 4) Iteración CALLS -> TMO (AR solo con histórico TMO) ===
    for ts in future_idx:
        if ts not in dfp.index:
            dfp.loc[ts, :] = np.nan

        tmp = dfp.loc[:ts].tail(24 * 200)
        tmp = _ensure_calendar(tmp)

        # Lags/MA de llamadas (planner)
        tmp_calls = add_lags_mas(tmp, TARGET_CALLS)

        # 1) Predicción CALLS (planner)
        row_pl = tmp_calls.tail(1)
        X_pl = dummies_and_reindex(row_pl, cols_pl, dummies_cols=['dow', 'month', 'hour'])
        X_pl_s = sc_pl.transform(X_pl)
        yhat_calls = float(m_pl.predict(X_pl_s, verbose=0).flatten()[0])
        yhat_calls = max(0.0, yhat_calls)
        dfp.at[ts, TARGET_CALLS] = yhat_calls

        # 2) TMO AR: construir lags/MA SOLO desde la serie TMO ya unida desde tmo_base
        tmp.loc[ts, TARGET_CALLS] = yhat_calls  # llamadas como exógena
        for lag in [24, 48, 72, 168]:
            tmp[f'lag_tmo_{lag}'] = tmp[TARGET_TMO].shift(lag)
        for window in [24, 72, 168]:
            tmp[f'ma_tmo_{window}'] = tmp[TARGET_TMO].rolling(window, min_periods=1).mean()

        if "precipitacion" in tmp.columns:
            tmp["es_dia_habil"] = ((tmp["dow"].isin([0, 1, 2, 3, 4])) & (tmp["feriados"] == 0)).astype(int)
            tmp["precipitacion_x_dia_habil"] = tmp["precipitacion"] * tmp["es_dia_habil"]

        row_tmo = tmp.tail(1)
        X_tmo = dummies_and_reindex(row_tmo, cols_tmo, dummies_cols=['dow', 'month', 'hour'])
        X_tmo_s = sc_tmo.transform(X_tmo)

        # 3) Predicción TMO (autoregresivo)
        yhat_tmo = float(m_tmo.predict(X_tmo_s, verbose=0).flatten()[0])
        yhat_tmo = max(0.0, yhat_tmo)
        dfp.at[ts, TARGET_TMO] = yhat_tmo

    # === 5) Salida horaria y Erlang ===
    out = dfp.loc[future_idx].copy()
    out = _ensure_calendar(out)
    out = out.rename(columns={TARGET_CALLS: "calls", TARGET_TMO: "tmo_s"})
    out["calls"] = out["calls"].fillna(0.0)
    out["tmo_s"] = out["tmo_s"].ffill()
    if out["tmo_s"].isna().all():
        out["tmo_s"] = 180.0

    df_hourly = out[["calls", "tmo_s"]].copy()
    df_hourly["agents_prod"] = 0
    for ts in df_hourly.index:
        a, _ = required_agents(float(df_hourly.at[ts, "calls"]), float(df_hourly.at[ts, "tmo_s"]))
        df_hourly.at[ts, "agents_prod"] = int(a)
    df_hourly["agents_sched"] = df_hourly["agents_prod"].apply(schedule_agents)

    # === 6) JSONs ===
    write_hourly_json(f"{PUBLIC_DIR}/prediccion_horaria.json",
                      df_hourly, "calls", "tmo_s", "agents_sched")
    write_daily_json(f"{PUBLIC_DIR}/prediccion_diaria.json",
                     df_hourly, "calls", "tmo_s")

    return df_hourly
