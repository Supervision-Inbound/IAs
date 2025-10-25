# -*- coding: utf-8 -*-
"""
inferencia_core.py
Módulo de utilidades de inferencia v7:
- Construcción de esqueleto futuro (horario) con calendario
- Simulación/Procesamiento de clima (solo para alertas/riesgo)
- Predicción de llamadas (Planificador MLP)
- Predicción de TMO autorregresivo (desacoplado del clima)
- Cálculo de Erlang simple

Este módulo NO cambia la lógica de llamadas; el TMO NO usa clima.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# =========================
# Configuración
# =========================
TZ = "America/Santiago"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# =========================
# Utilidades generales
# =========================
def read_data(path, hoja=None):
    pl = path.lower()
    if not os.path.exists(path):
        raise FileNotFoundError(f"No encontrado: {path}")
    if pl.endswith(".csv"):
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            df = None
        if df is None or (df.shape[1] == 1 and df.iloc[0, 0] is not None and ";" in str(df.iloc[0, 0])):
            df = pd.read_csv(path, delimiter=";", low_memory=False)
        return df
    elif pl.endswith((".xlsx", ".xls")):
        return pd.read_excel(path, sheet_name=hoja if hoja is not None else 0)
    else:
        raise ValueError(f"Formato no soportado: {path}")

def ensure_ts_and_tz(df):
    df = df.copy()
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    else:
        date_col = next((c for c in df.columns if "fecha" in c), None)
        hour_col = next((c for c in df.columns if "hora" in c), None)
        if date_col and hour_col:
            try:
                df["ts"] = pd.to_datetime(df[date_col] + " " + df[hour_col],
                                          format="%d-%m-%Y %H:%M:%S", errors="raise")
            except Exception:
                df["ts"] = pd.to_datetime(df[date_col].astype(str) + " " + df[hour_col].astype(str), errors="coerce")
        elif "datatime" in df.columns:
            df["ts"] = pd.to_datetime(df["datatime"], errors="coerce")
        else:
            raise ValueError("Se requiere 'ts' o ('fecha' + 'hora') o 'datatime'.")
    df = df.dropna(subset=["ts"])
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    else:
        df["ts"] = df["ts"].dt.tz_convert(TZ)
    df = df.dropna(subset=["ts"])
    return df.sort_values("ts")

def add_time_parts(df):
    df = df.copy()
    df["dow"] = df["ts"].dt.dayofweek
    df["month"] = df["ts"].dt.month
    df["hour"] = df["ts"].dt.hour
    df["day"] = df["ts"].dt.day
    df["es_dia_de_pago"] = df["day"].isin([1, 2, 15, 16, 29, 30, 31]).astype(int)
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_dow"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["dow"] / 7)
    return df

def calculate_erlang_agents(calls_per_hour, tmo_seconds, occupancy_target=0.85):
    calls = pd.to_numeric(calls_per_hour, errors="coerce").fillna(0)
    tmo = pd.to_numeric(tmo_seconds, errors="coerce").fillna(0)
    if (calls.sum() == 0) or (tmo <= 0).all():
        return pd.Series(0, index=calls_per_hour.index)
    tmo_safe = tmo.replace(0, 1e-6)
    traffic_intensity = (calls * tmo_safe) / 3600.0
    agents = np.ceil(traffic_intensity / occupancy_target)
    agents = pd.Series(agents, index=calls.index)
    agents[calls > 0] = agents[calls > 0].apply(lambda x: max(int(x), 1))
    return agents.replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)

# =========================
# Calendario / Feriados
# =========================
def _series_is_holiday(ts_index: pd.DatetimeIndex, holidays_set) -> pd.Series:
    """
    Corrige el bug de 'numpy.ndarray' sin .isin:
    Trabaja siempre con una Serie de fechas (dtype=object) y usa .isin de pandas.
    """
    if not isinstance(ts_index, pd.DatetimeIndex):
        ts_index = pd.DatetimeIndex(ts_index)
    if ts_index.tz is None:
        ts_index = ts_index.tz_localize(TZ)
    as_local = ts_index.tz_convert(TZ)
    # Serie de objetos date
    dates_ser = pd.Series(as_local.date, index=as_local)
    return dates_ser.astype("object").isin(holidays_set).astype(int).reindex(as_local)

def _ensure_calendar(tmp_df: pd.DataFrame, holidays_set) -> pd.DataFrame:
    """
    Añade columna 'feriados' a partir del índice de fechas (se espera 'ts' como índice o columna).
    """
    if "ts" in tmp_df.columns:
        tmp_df = tmp_df.set_index(pd.DatetimeIndex(tmp_df["ts"]))
    if not isinstance(tmp_df.index, pd.DatetimeIndex):
        raise ValueError("Se requiere índice DatetimeIndex para calendario.")
    tmp_df = tmp_df.copy()
    tmp_df["feriados"] = _series_is_holiday(tmp_df.index, holidays_set).values
    return tmp_df.reset_index().rename(columns={"index": "ts"})

# =========================
# Clima (solo para riesgo/alertas)
# =========================
def normalize_climate_columns(df: pd.DataFrame) -> pd.DataFrame:
    column_map = {
        "temperatura": ["temperature_2m", "temperatura", "temp", "temp_2m"],
        "precipitacion": ["precipitation", "precipitacion", "precipitación", "rain_mm", "rain"],
        "lluvia": ["rain", "lluvia", "rainfall"],
    }
    df_ren = df.copy()
    df_ren.columns = [c.lower().strip().replace(" ", "_") for c in df_ren.columns]
    for std, poss in column_map.items():
        for name in poss:
            if name in df_ren.columns:
                df_ren.rename(columns={name: std}, inplace=True)
                break
    return df_ren

def simulate_future_weather(df_hist_clima: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    """
    Simula una serie futura por matching con año anterior o última semana.
    Si no hay datos, crea dummy.
    """
    if df_hist_clima is None or df_hist_clima.empty:
        comunas = ["Santiago"]
        dates = pd.date_range(start=start_date, end=end_date, freq="h", tz=TZ)
        d = pd.DataFrame(index=pd.MultiIndex.from_product([comunas, dates], names=["comuna", "ts"]))
        d["temperatura"] = 15
        d["precipitacion"] = 0
        d["lluvia"] = 0
        return d.reset_index()

    df_hist = ensure_ts_and_tz(df_hist_clima)
    df_hist = normalize_climate_columns(df_hist)
    if "comuna" not in df_hist.columns:
        df_hist["comuna"] = "Santiago"

    future_dates = pd.date_range(start=start_date, end=end_date, freq="h", tz=TZ)
    out = []
    for date in future_dates:
        try:
            sim_date = date.replace(year=date.year - 1)
        except ValueError:
            sim_date = date - pd.Timedelta(days=365)
        got = df_hist[df_hist["ts"] == sim_date]
        if not got.empty:
            g = got.copy()
            g["ts"] = date
            out.append(g)

    if not out:
        # fallback: última semana
        last_week = df_hist[df_hist["ts"] >= df_hist["ts"].max() - pd.Timedelta(days=7)]
        if last_week.empty:
            comunas = ["Santiago"]
            dates = pd.date_range(start=start_date, end=end_date, freq="h", tz=TZ)
            d = pd.DataFrame(index=pd.MultiIndex.from_product([comunas, dates], names=["comuna", "ts"]))
            d["temperatura"] = 15
            d["precipitacion"] = 0
            d["lluvia"] = 0
            return d.reset_index()
        lw_map = last_week.set_index("ts")
        for date in future_dates:
            sim = (date - pd.Timedelta(days=7)).floor("h")
            if sim in lw_map.index:
                g = lw_map.loc[[sim]].reset_index(drop=True)
                g["ts"] = date
                out.append(g)

    df_sim = pd.concat(out) if out else pd.DataFrame(columns=["comuna", "ts"])
    all_comunas = df_hist["comuna"].unique() if not df_hist.empty else np.array(["Santiago"])
    full_index = pd.MultiIndex.from_product([all_comunas, future_dates], names=["comuna", "ts"])
    df_final = df_sim.set_index(["comuna", "ts"]).reindex(full_index)
    df_final = df_final.groupby(level="comuna").ffill().bfill().fillna(0)
    return df_final.reset_index()

def process_future_climate(df_future_weather: pd.DataFrame, df_baselines: pd.DataFrame):
    df = normalize_climate_columns(df_future_weather.copy())
    if "ts" not in df.columns or df["ts"].isnull().all():
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    else:
        df["ts"] = df["ts"].dt.tz_convert(TZ)
    df = df.dropna(subset=["ts"]).sort_values(["comuna", "ts"])
    df["dow"] = df["ts"].dt.dayofweek
    df["hour"] = df["ts"].dt.hour

    if isinstance(df_baselines, pd.DataFrame) and not df_baselines.empty:
        df_m = pd.merge(df, df_baselines, on=["comuna", "dow", "hour"], how="left")
    else:
        df_m = df.copy()
        for m in ["temperatura", "precipitacion", "lluvia"]:
            df_m[f"{m}_median"] = 0
            df_m[f"{m}_std"] = 1

    for metric in ["temperatura", "precipitacion", "lluvia"]:
        if metric in df_m.columns and f"{metric}_median" in df_m.columns and f"{metric}_std" in df_m.columns:
            df_m[f"anomalia_{metric}"] = (df_m[metric] - df_m[f"{metric}_median"]) / (df_m[f"{metric}_std"] + 1e-6)
        else:
            df_m[f"anomalia_{metric}"] = 0

    n_comunas = max(1, df_m["comuna"].nunique())
    anomaly_cols = [c for c in df_m.columns if c.startswith("anomalia_")]
    if anomaly_cols:
        agg = {}
        for c in anomaly_cols:
            agg[c] = ["max", "sum", lambda x, nc=n_comunas: (x > 2.5).sum() / nc if nc > 0 else 0]
        df_ag = df_m.groupby("ts").agg(agg).reset_index()
        new_cols = ["ts"]
        for col in df_ag.columns[1:]:
            aggname = col[1] if col[1] != "<lambda_0>" else "pct_comunas_afectadas"
            new_cols.append(f"{col[0]}_{aggname}")
        df_ag.columns = new_cols
    else:
        df_ag = pd.DataFrame({"ts": df_m["ts"].unique()})

    return df_ag, df_m[["ts", "comuna"] + anomaly_cols] if anomaly_cols else df_m[["ts", "comuna"]]

def predict_risk(df_future: pd.DataFrame, model, scaler, cols_risk: list) -> pd.Series:
    if not cols_risk or not all(c in df_future.columns for c in cols_risk):
        return pd.Series(0.0, index=df_future.index)
    X = df_future.reindex(columns=cols_risk, fill_value=0)
    Xn = X.select_dtypes(include=np.number).columns
    X[Xn] = X[Xn].fillna(X[Xn].mean()).fillna(0)
    Xs = scaler.transform(X)
    proba = model.predict(Xs)
    if isinstance(proba, (list, tuple, np.ndarray)):
        proba = np.array(proba).reshape(-1)
    return pd.Series(proba, index=df_future.index)

# =========================
# Planificador de llamadas
# =========================
def predict_calls(df_hosting_proc: pd.DataFrame,
                  df_future_base: pd.DataFrame,
                  model, scaler, cols_planner: list,
                  target_calls: str = "recibidos_nacional") -> pd.DataFrame:
    """
    Devuelve df_future con columna 'llamadas_hora' sin modificar la lógica original.
    """
    df_full = pd.concat([df_hosting_proc, df_future_base], ignore_index=True).sort_values("ts")
    for lag in [24, 48, 72, 168]:
        df_full[f"lag_{lag}"] = df_full[target_calls].shift(lag)
    for window in [24, 72, 168]:
        df_full[f"ma_{window}"] = df_full[target_calls].shift(1).rolling(window, min_periods=1).mean()

    df_future_feats = df_full[df_full["ts"] >= df_future_base["ts"].min()].copy()
    X = pd.get_dummies(df_future_feats, columns=["dow", "month", "hour"])
    X = X.reindex(columns=cols_planner, fill_value=0)
    num_cols = X.select_dtypes(include=np.number).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].mean()).fillna(0)
    Xs = scaler.transform(X)

    y = model.predict(Xs).clip(0)
    df_future_out = df_future_base.copy()
    df_future_out["llamadas_hora"] = y.astype(int)
    return df_future_out

# =========================
# TMO Autorregresivo
# =========================
def _ensure_hourly_continuous(series: pd.Series, tz: str) -> pd.Series:
    if series.empty:
        return series
    idx = pd.date_range(series.index.min().floor("h"), series.index.max().floor("h"), freq="h", tz=tz)
    s = series.reindex(idx).sort_index().ffill()
    return s

def _rolling_mean_safe(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def predict_tmo_autoregressive(df_future: pd.DataFrame,
                               df_tmo_hist: pd.DataFrame,
                               model, scaler, cols_tmo: list,
                               target_calls_col: str = "llamadas_hora",
                               target_tmo: str = "tmo_general",
                               force_zero_anomalias: bool = True) -> pd.DataFrame:
    """
    Predice TMO por hora de forma autoregresiva:
      - Usa lags/medias móviles de TMO anteriores (hist + pred)
      - Puede usar llamadas_hora como exógena
      - Si cols_tmo tiene 'anomalia_*' y force_zero_anomalias=True, pone 0.
    Devuelve df con 'tmo_hora'.
    """
    if df_tmo_hist.empty:
        raise ValueError("TMO_HISTORICO vacío.")

    hist = df_tmo_hist.copy().sort_values("ts")
    if hist["ts"].dt.tz is None:
        hist["ts"] = hist["ts"].dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    hist_idx = hist["ts"].dt.floor("h")
    if target_tmo not in hist.columns:
        raise ValueError(f"No se encontró '{target_tmo}' en TMO_HISTORICO.csv.")

    # serie continua mínima 168h
    hist_s = pd.Series(hist[target_tmo].values, index=hist_idx)
    hist_s = _ensure_hourly_continuous(hist_s, TZ)
    min_boot = 168
    if len(hist_s) < min_boot:
        pad_hours = min_boot - len(hist_s)
        pad_idx = pd.date_range(end=hist_s.index.min() - pd.Timedelta(hours=1),
                                periods=pad_hours, freq="h", tz=TZ)
        pad_series = pd.Series(hist_s.iloc[0], index=pad_idx).sort_index()
        hist_s = pd.concat([pad_series, hist_s]).sort_index()

    # semillas
    seeds = {
        "proporcion_comercial": hist["proporcion_comercial"].iloc[-1] if "proporcion_comercial" in hist.columns else 0.0,
        "proporcion_tecnica":   hist["proporcion_tecnica"].iloc[-1]   if "proporcion_tecnica"   in hist.columns else 0.0,
        "tmo_comercial":        hist["tmo_comercial"].iloc[-1]        if "tmo_comercial"        in hist.columns else 0.0,
        "tmo_tecnico":          hist["tmo_tecnico"].iloc[-1]          if "tmo_tecnico"          in hist.columns else 0.0,
    }

    df_fut = df_future.copy().sort_values("ts").reset_index(drop=True)

    # desconectar clima del TMO si aplica
    if force_zero_anomalias:
        anom_cols = [c for c in cols_tmo if c.startswith("anomalia_")]
        for c in anom_cols:
            if c not in df_fut.columns:
                df_fut[c] = 0.0
            else:
                df_fut[c] = 0.0

    future_idx = df_fut["ts"].dt.floor("h")
    backfill_start = hist_s.index.max() - pd.Timedelta(hours=min_boot - 1)
    hist_boot = hist_s[hist_s.index >= backfill_start]
    hist_boot = hist_boot.reindex(pd.date_range(start=backfill_start,
                                                end=hist_s.index.max(),
                                                freq="h", tz=TZ)).ffill()

    full_idx = pd.date_range(start=hist_boot.index.min(),
                             end=future_idx.max(), freq="h", tz=TZ)
    tmo_full = pd.Series(index=full_idx, dtype="float64")
    tmo_full.loc[hist_boot.index] = hist_boot.values

    preds = []
    for ts in future_idx:
        def _safe(ts_lag):
            return tmo_full.loc[ts_lag] if ts_lag in tmo_full.index else tmo_full.ffill().iloc[-1]

        lag_24  = _safe(ts - pd.Timedelta(hours=24))
        lag_48  = _safe(ts - pd.Timedelta(hours=48))
        lag_72  = _safe(ts - pd.Timedelta(hours=72))
        lag_168 = _safe(ts - pd.Timedelta(hours=168))

        prev_end = ts - pd.Timedelta(hours=1)
        if prev_end in tmo_full.index:
            ma_24  = _rolling_mean_safe(tmo_full.loc[ts - pd.Timedelta(hours=23): prev_end], 24).iloc[-1]
            ma_72  = _rolling_mean_safe(tmo_full.loc[ts - pd.Timedelta(hours=71): prev_end], 72).iloc[-1]
            ma_168 = _rolling_mean_safe(tmo_full.loc[ts - pd.Timedelta(hours=167): prev_end], 168).iloc[-1]
        else:
            ma_24 = ma_72 = ma_168 = lag_24

        row = df_fut.loc[df_fut["ts"].dt.floor("h") == ts].iloc[0].copy()

        # exógenas (llamadas y proporciones)
        if target_calls_col in row:
            row["recibidos_nacional"] = row[target_calls_col]  # el modelo pudo entrenar con este nombre
        else:
            row["recibidos_nacional"] = 0
        row["proporcion_comercial"] = seeds["proporcion_comercial"]
        row["proporcion_tecnica"]   = seeds["proporcion_tecnica"]
        if "tmo_comercial" in cols_tmo:
            row["tmo_comercial"] = seeds["tmo_comercial"]
        if "tmo_tecnico" in cols_tmo:
            row["tmo_tecnico"] = seeds["tmo_tecnico"]

        # lags/MA en fila
        row["lag_tmo_24"]  = float(lag_24)
        row["lag_tmo_48"]  = float(lag_48)
        row["lag_tmo_72"]  = float(lag_72)
        row["lag_tmo_168"] = float(lag_168)
        row["ma_tmo_24"]   = float(ma_24)
        row["ma_tmo_72"]   = float(ma_72)
        row["ma_tmo_168"]  = float(ma_168)

        # dummies + alineación a columnas de entrenamiento
        row_df = pd.DataFrame([row])
        row_df = pd.get_dummies(row_df, columns=["dow", "month", "hour"])
        row_df = row_df.reindex(columns=cols_tmo, fill_value=0)

        Xs = scaler.transform(row_df.values)
        y_hat = float(model.predict(Xs, verbose=0).reshape(-1)[0])
        y_hat = max(0.0, y_hat)

        tmo_full.loc[ts] = y_hat
        preds.append((ts, y_hat))

    out = pd.DataFrame(preds, columns=["ts", "tmo_hora"])
    return out

# =========================
# Construcción de futuro
# =========================
def build_future_frame(last_ts: pd.Timestamp, horizonte_dias: int, feriados_set) -> pd.DataFrame:
    start_future = last_ts + pd.Timedelta(hours=1)
    end_future = start_future + pd.Timedelta(days=horizonte_dias, hours=23)
    df_future = pd.DataFrame(pd.date_range(start=start_future, end=end_future, freq="h", tz=TZ), columns=["ts"])
    df_future = add_time_parts(df_future)
    df_future["feriados"] = df_future["ts"].dt.date.isin(feriados_set).astype(int)
    return df_future

# =========================
# Compacto tipo "forecast_120d"
# =========================
def forecast_120d(df_hosting_proc: pd.DataFrame,
                  df_future_base: pd.DataFrame,
                  holidays_set,
                  planner_artifacts: dict,
                  risk_artifacts: dict,
                  clima_hist_df: pd.DataFrame,
                  baselines_clima: pd.DataFrame) -> pd.DataFrame:
    """
    Implementación compacta para mantener compatibilidad con proyectos que llamen a forecast_120d.
    Devuelve df_future con columnas: ts, feriados, risk_proba, llamadas_hora.
    """
    # Asegura feriados
    tmp = df_future_base.copy()
    tmp = _ensure_calendar(tmp, holidays_set)

    # Clima -> anomalías -> riesgo
    df_weather_future = simulate_future_weather(clima_hist_df, tmp["ts"].min(), tmp["ts"].max())
    df_agg_anoms, _ = process_future_climate(df_weather_future, baselines_clima if isinstance(baselines_clima, pd.DataFrame) else pd.DataFrame())
    tmp = pd.merge(tmp, df_agg_anoms, on="ts", how="left")
    num_cols = tmp.select_dtypes(include=np.number).columns
    tmp[num_cols] = tmp[num_cols].fillna(tmp[num_cols].mean())
    tmp = tmp.fillna(0)

    # Riesgo
    if all(k in risk_artifacts for k in ("model", "scaler", "cols")):
        tmp["risk_proba"] = predict_risk(tmp, risk_artifacts["model"], risk_artifacts["scaler"], risk_artifacts["cols"])
    else:
        tmp["risk_proba"] = 0.0

    # Llamadas
    if all(k in planner_artifacts for k in ("model", "scaler", "cols", "target_calls")):
        tmp = predict_calls(
            df_hosting_proc=df_hosting_proc,
            df_future_base=tmp,
            model=planner_artifacts["model"],
            scaler=planner_artifacts["scaler"],
            cols_planner=planner_artifacts["cols"],
            target_calls=planner_artifacts["target_calls"],
        )
    else:
        tmp["llamadas_hora"] = 0

    return tmp

# =========================
# Carga de artefactos helper
# =========================
def load_artifacts(model_dir: str, kind: str):
    """
    kind in {"planner", "risk", "tmo"}
    Retorna dict con {"model","scaler","cols"} (según corresponda).
    """
    paths = {
        "planner": {
            "model": os.path.join(model_dir, "modelo_planner.keras"),
            "scaler": os.path.join(model_dir, "scaler_planner.pkl"),
            "cols":  os.path.join(model_dir, "training_columns_planner.json"),
        },
        "risk": {
            "model": os.path.join(model_dir, "modelo_riesgos.keras"),
            "scaler": os.path.join(model_dir, "scaler_riesgos.pkl"),
            "cols":  os.path.join(model_dir, "training_columns_riesgos.json"),
        },
        "tmo": {
            "model": os.path.join(model_dir, "modelo_tmo.keras"),
            "scaler": os.path.join(model_dir, "scaler_tmo.pkl"),
            "cols":  os.path.join(model_dir, "training_columns_tmo.json"),
        },
    }
    p = paths[kind]
    model = tf.keras.models.load_model(p["model"])
    scaler = joblib.load(p["scaler"])
    with open(p["cols"], "r") as f:
        cols = json.load(f)
    return {"model": model, "scaler": scaler, "cols": cols}

