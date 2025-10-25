# -*- coding: utf-8 -*-
"""
main.py
Inferencia v7 con:
- Llamadas (Planificador MLP) => SIN cambios
- Riesgos/Clima => solo para alertas; NO afecta TMO
- TMO Autorregresivo verdadero, consistente con training_columns_tmo.json
Requiere:
  models/
    modelo_planner.keras, scaler_planner.pkl, training_columns_planner.json
    modelo_riesgos.keras,  scaler_riesgos.pkl,  training_columns_riesgos.json, baselines_clima.pkl (opcional)
    modelo_tmo.keras,      scaler_tmo.pkl,      training_columns_tmo.json
  data/
    historical_data.csv (llamadas históricas)
    TMO_HISTORICO.csv    (tmo histórico puro)
    Feriados_Chilev2.csv (feriados; opcional)
"""

import os
import json
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# =========================
# Configuración Global
# =========================
TZ = "America/Santiago"
os.environ["TZ"] = TZ
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
PUBLIC_DIR = os.path.join(ROOT_DIR, "public")

# Archivos estándar (coinciden con tu flujo)
PLANNER_MODEL_FILE = os.path.join(MODEL_DIR, "modelo_planner.keras")
PLANNER_SCALER_FILE = os.path.join(MODEL_DIR, "scaler_planner.pkl")
PLANNER_COLS_FILE = os.path.join(MODEL_DIR, "training_columns_planner.json")

RISK_MODEL_FILE = os.path.join(MODEL_DIR, "modelo_riesgos.keras")
RISK_SCALER_FILE = os.path.join(MODEL_DIR, "scaler_riesgos.pkl")
RISK_COLS_FILE = os.path.join(MODEL_DIR, "training_columns_riesgos.json")
RISK_BASELINES_FILE = os.path.join(MODEL_DIR, "baselines_clima.pkl")

TMO_MODEL_FILE = os.path.join(MODEL_DIR, "modelo_tmo.keras")
TMO_SCALER_FILE = os.path.join(MODEL_DIR, "scaler_tmo.pkl")
TMO_COLS_FILE = os.path.join(MODEL_DIR, "training_columns_tmo.json")

HOSTING_FILE = os.path.join(DATA_DIR, "historical_data.csv")
TMO_FILE = os.path.join(DATA_DIR, "TMO_HISTORICO.csv")
FERIADOS_FILE = os.path.join(DATA_DIR, "Feriados_Chilev2.csv")
CLIMA_HIST_FILE = os.path.join(DATA_DIR, "historical_data.csv")  # solo para simular clima futuro

TARGET_CALLS = "recibidos_nacional"
TARGET_TMO = "tmo_general"

# =========================
# Utilidades
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
        # Si viene con ; en una sola columna, reintenta con delimitador
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
    # Admite (ts) o (fecha + hora) o (datatime)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    else:
        date_col = next((c for c in df.columns if "fecha" in c), None)
        hour_col = next((c for c in df.columns if "hora" in c), None)
        if date_col and hour_col:
            # intenta DD-MM-YYYY HH:MM:SS primero; si falla, infiere
            try:
                df["ts"] = pd.to_datetime(df[date_col] + " " + df[hour_col],
                                          format="%d-%m-%Y %H:%M:%S", errors="raise")
            except Exception:
                df["ts"] = pd.to_datetime(df[date_col].astype(str) + " " + df[hour_col].astype(str), errors="coerce")
        elif "datatime" in df.columns:  # por compatibilidad con cargas antiguas
            df["ts"] = pd.to_datetime(df["datatime"], errors="coerce")
        else:
            raise ValueError("Se requiere 'ts' o ('fecha' + 'hora') o 'datatime' en el CSV.")
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

def normalize_climate_columns(df: pd.DataFrame) -> pd.DataFrame:
    # mapea nombres a estándar si existieran
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

def calculate_erlang_agents(calls_per_hour, tmo_seconds, occupancy_target=0.85):
    calls = pd.to_numeric(calls_per_hour, errors="coerce").fillna(0)
    tmo = pd.to_numeric(tmo_seconds, errors="coerce").fillna(0)
    if (calls.sum() == 0) or (tmo <= 0).all():
        return pd.Series(0, index=calls_per_hour.index)
    tmo_safe = tmo.replace(0, 1e-6)
    traffic_intensity = (calls * tmo_safe) / 3600.0
    agents = np.ceil(traffic_intensity / occupancy_target)
    if hasattr(calls, "index"):
        agents = pd.Series(agents, index=calls.index)
        agents[calls > 0] = agents[calls > 0].apply(lambda x: max(int(x), 1))
        return agents.replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)
    return pd.Series(agents).replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)

# =========================
# Clima (solo para alertas)
# =========================
def fetch_future_weather(start_date, end_date):
    print("    [Clima] Simulando clima futuro con histórico...")
    try:
        df_hist = read_data(CLIMA_HIST_FILE)
    except FileNotFoundError:
        # Dummy simple
        comunas = ["Santiago"]
        dates = pd.date_range(start=start_date, end=end_date, freq="h", tz=TZ)
        df_sim = pd.DataFrame(index=pd.MultiIndex.from_product([comunas, dates], names=["comuna", "ts"]))
        df_sim["temperatura"] = 15
        df_sim["precipitacion"] = 0
        df_sim["lluvia"] = 0
        return df_sim.reset_index()

    df_hist = ensure_ts_and_tz(df_hist)
    df_hist = normalize_climate_columns(df_hist)
    if "comuna" not in df_hist.columns:
        df_hist["comuna"] = "Santiago"

    future_dates = pd.date_range(start=start_date, end=end_date, freq="h", tz=TZ)
    df_future_list = []
    for date in future_dates:
        # toma el mismo timestamp del año anterior (si existe)
        try:
            sim_date = date.replace(year=date.year - 1)
        except ValueError:
            sim_date = date - pd.Timedelta(days=365)
        data_sim = df_hist[df_hist["ts"] == sim_date]
        if not data_sim.empty:
            d = data_sim.copy()
            d["ts"] = date
            df_future_list.append(d)

    if not df_future_list:
        # fallback: última semana desplazada
        last_week = df_hist[df_hist["ts"] >= df_hist["ts"].max() - pd.Timedelta(days=7)]
        if last_week.empty:
            comunas = ["Santiago"]
            dates = pd.date_range(start=start_date, end=end_date, freq="h", tz=TZ)
            df_sim = pd.DataFrame(index=pd.MultiIndex.from_product([comunas, dates], names=["comuna", "ts"]))
            df_sim["temperatura"] = 15
            df_sim["precipitacion"] = 0
            df_sim["lluvia"] = 0
            return df_sim.reset_index()
        lw_map = last_week.set_index("ts")
        for date in future_dates:
            sim = date - pd.Timedelta(days=7)
            sim_floor = sim.floor("h")
            if sim_floor in lw_map.index:
                d = lw_map.loc[[sim_floor]].reset_index(drop=True)
                d["ts"] = date
                df_future_list.append(d)

    df_simulado = pd.concat(df_future_list) if df_future_list else pd.DataFrame(columns=["comuna", "ts"])
    all_comunas = df_hist["comuna"].unique() if not df_hist.empty else np.array(["Santiago"])
    full_index = pd.MultiIndex.from_product([all_comunas, future_dates], names=["comuna", "ts"])
    df_final = df_simulado.set_index(["comuna", "ts"]).reindex(full_index)
    df_final = df_final.groupby(level="comuna").ffill().bfill().fillna(0)
    print(f"    [Clima] Simulación completada: {len(df_final)} filas.")
    return df_final.reset_index()

def process_future_climate(df_future_weather, df_baselines):
    print("    [Clima] Procesando anomalías...")
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
        # Renombrar columnas multi-índice
        new_cols = ["ts"]
        for col in df_ag.columns[1:]:
            aggname = col[1] if col[1] != "<lambda_0>" else "pct_comunas_afectadas"
            new_cols.append(f"{col[0]}_{aggname}")
        df_ag.columns = new_cols
    else:
        df_ag = pd.DataFrame({"ts": df_m["ts"].unique()})

    return df_ag, df_m[["ts", "comuna"] + anomaly_cols] if anomaly_cols else df_m[["ts", "comuna"]]

def generate_alerts_json(df_per_comuna, df_risk_proba, proba_threshold=0.5, impact_factor=100):
    print("    [Alertas] Generando JSON...")
    df_alertas = pd.merge(df_per_comuna, df_risk_proba, on="ts", how="left")
    if "risk_proba" not in df_alertas.columns:
        return []
    df_alertas = df_alertas[df_alertas["risk_proba"] > proba_threshold].copy()
    if df_alertas.empty:
        return []
    # construir bloques por comuna
    out = []
    for comuna, g in df_alertas.groupby("comuna"):
        g = g.sort_values("ts").copy()
        g["time_diff"] = g["ts"].diff().dt.total_seconds().div(3600)
        g["bloque"] = (g["time_diff"] > 1).cumsum()
        for _, b in g.groupby("bloque"):
            ts_i = b["ts"].min()
            ts_f = b["ts"].max() + pd.Timedelta(minutes=59)
            anoms = {c.replace("anomalia_", "") + "_z_max": round(b[c].max(), 2)
                     for c in b.columns if c.startswith("anomalia_")}
            impacto = ((b["risk_proba"] - proba_threshold) * impact_factor).sum()
            out.append({
                "comuna": comuna,
                "ts_inicio": ts_i.strftime("%Y-%m-%d %H:%M:%S"),
                "ts_fin": ts_f.strftime("%Y-%m-%d %H:%M:%S"),
                "anomalias": anoms,
                "impacto_llamadas_adicionales": int(impacto)
            })
    return out

# =========================
# Fase TMO AR - helpers
# =========================
def _ensure_hourly_continuous(series, tz):
    if series.empty:
        return series
    idx = pd.date_range(series.index.min().floor("h"), series.index.max().floor("h"), freq="h", tz=tz)
    s = series.reindex(idx).sort_index()
    return s.ffill()

def _rolling_mean_safe(series, window):
    return series.rolling(window=window, min_periods=1).mean()

# =========================
# Pipeline Principal
# =========================
def main(horizonte_dias: int):
    print("=" * 60)
    print(f"INFERENCIA v7 | TZ={TZ} | HORIZONTE={horizonte_dias} días")
    print("=" * 60)

    # 1) Cargar modelos/artefactos
    print("\n--- Fase 1: Cargando modelos y artefactos ---")
    model_planner = tf.keras.models.load_model(PLANNER_MODEL_FILE)
    scaler_planner = joblib.load(PLANNER_SCALER_FILE)
    with open(PLANNER_COLS_FILE, "r") as f:
        cols_planner = json.load(f)

    model_risk = tf.keras.models.load_model(RISK_MODEL_FILE)
    scaler_risk = joblib.load(RISK_SCALER_FILE)
    try:
        with open(RISK_COLS_FILE, "r") as f:
            cols_risk = json.load(f)
    except FileNotFoundError:
        cols_risk = []
    try:
        baselines_clima = joblib.load(RISK_BASELINES_FILE)
    except Exception:
        baselines_clima = pd.DataFrame()

    model_tmo = tf.keras.models.load_model(TMO_MODEL_FILE)
    scaler_tmo = joblib.load(TMO_SCALER_FILE)
    with open(TMO_COLS_FILE, "r") as f:
        cols_tmo = json.load(f)

    print("  [OK] Modelos cargados.")

    # 2) Cargar datos históricos
    print("\n--- Fase 2: Cargando históricos ---")
    df_hosting_full = read_data(HOSTING_FILE)
    df_hosting = ensure_ts_and_tz(df_hosting_full)

    # feriados
    try:
        df_fer = read_data(FERIADOS_FILE)
        df_fer["Fecha_dt"] = pd.to_datetime(df_fer["Fecha"], format="%d-%m-%Y", errors="coerce").dt.date
        feriados_set = set(df_fer["Fecha_dt"].dropna())
    except Exception:
        feriados_set = set()

    # columna de llamadas
    if TARGET_CALLS not in df_hosting.columns:
        if "recibidos" in df_hosting.columns:
            df_hosting = df_hosting.rename(columns={"recibidos": TARGET_CALLS})
        else:
            # no tocamos nombres; si no está, es error
            raise ValueError(f"No se encontró columna '{TARGET_CALLS}' ni 'recibidos' en historical_data.csv.")

    # feriados numérico
    if "feriados" not in df_hosting.columns:
        df_hosting["feriados"] = df_hosting["ts"].dt.date.isin(feriados_set).astype(int)
    else:
        df_hosting["feriados"] = pd.to_numeric(df_hosting["feriados"], errors="coerce").fillna(0).astype(int)

    # agrupar por ts (sum para llamadas, max feriados)
    df_hosting_agg = df_hosting.groupby("ts").agg({TARGET_CALLS: "sum", "feriados": "max"}).reset_index()
    df_hosting_proc = add_time_parts(df_hosting_agg)

    # TMO histórico
    df_tmo_hist = read_data(TMO_FILE)
    df_tmo_hist = ensure_ts_and_tz(df_tmo_hist)
    df_tmo_hist.columns = [c.lower().strip().replace(" ", "_") for c in df_tmo_hist.columns]
    if TARGET_TMO not in df_tmo_hist.columns:
        # si tu archivo lo trae con otro nombre, mapea aquí (pero por estándar es tmo_general)
        raise ValueError(f"No se encontró '{TARGET_TMO}' en TMO_HISTORICO.csv.")
    # proporciones si existen
    if "q_llamadas_comercial" in df_tmo_hist.columns and "q_llamadas_general" in df_tmo_hist.columns:
        df_tmo_hist["proporcion_comercial"] = df_tmo_hist["q_llamadas_comercial"] / (df_tmo_hist["q_llamadas_general"] + 1e-6)
        if "q_llamadas_tecnico" in df_tmo_hist.columns:
            df_tmo_hist["proporcion_tecnica"] = df_tmo_hist["q_llamadas_tecnico"] / (df_tmo_hist["q_llamadas_general"] + 1e-6)
        else:
            df_tmo_hist["proporcion_tecnica"] = 0.0
    else:
        df_tmo_hist["proporcion_comercial"] = 0.0
        df_tmo_hist["proporcion_tecnica"] = 0.0

    last_hist_ts = df_hosting_proc["ts"].max()
    print(f"  [OK] Último TS histórico: {last_hist_ts}")

    # 3) Esqueleto futuro
    print("\n--- Fase 3: Esqueleto futuro ---")
    start_future = last_hist_ts + pd.Timedelta(hours=1)
    end_future = start_future + pd.Timedelta(days=horizonte_dias, hours=23)
    df_future = pd.DataFrame(pd.date_range(start=start_future, end=end_future, freq="h", tz=TZ), columns=["ts"])
    # calendario y feriados
    df_future = add_time_parts(df_future)
    df_future["feriados"] = df_future["ts"].dt.date.isin(feriados_set).astype(int)
    print(f"  [OK] Esqueleto: {df_future['ts'].min()} -> {df_future['ts'].max()}")

    # 4) Clima (solo para alertas y riesgos; NO se usa en TMO)
    print("\n--- Fase 4: Clima/Riesgo (solo alertas) ---")
    df_weather_future = fetch_future_weather(start_future, end_future)
    df_agg_anoms, df_per_comuna_anoms = process_future_climate(df_weather_future, baselines_clima if isinstance(baselines_clima, pd.DataFrame) else pd.DataFrame())
    df_future = pd.merge(df_future, df_agg_anoms, on="ts", how="left")
    # rellenar NaN numéricos
    num_cols = df_future.select_dtypes(include=np.number).columns
    df_future[num_cols] = df_future[num_cols].fillna(df_future[num_cols].mean())
    df_future = df_future.fillna(0)

    # riesgo (probabilidad de picos) si hay columnas y scaler
    if len(cols_risk) > 0 and all(c in df_future.columns for c in cols_risk):
        X_risk = df_future.reindex(columns=cols_risk, fill_value=0)
        X_risk_s = scaler_risk.transform(X_risk)
        df_future["risk_proba"] = model_risk.predict(X_risk_s)
    else:
        df_future["risk_proba"] = 0.0

    # 5) Planificador de llamadas (SIN cambios)
    print("\n--- Fase 5: Llamadas (Planificador MLP v7) ---")
    df_full = pd.concat([df_hosting_proc, df_future], ignore_index=True).sort_values("ts")
    # lags/MA de llamadas (misma receta que entrenamiento)
    for lag in [24, 48, 72, 168]:
        df_full[f"lag_{lag}"] = df_full[TARGET_CALLS].shift(lag)
    for window in [24, 72, 168]:
        df_full[f"ma_{window}"] = df_full[TARGET_CALLS].shift(1).rolling(window, min_periods=1).mean()

    df_future_feats = df_full[df_full["ts"] >= start_future].copy()
    X_pl = pd.get_dummies(df_future_feats, columns=["dow", "month", "hour"])
    X_pl = X_pl.reindex(columns=cols_planner, fill_value=0)
    ncp = X_pl.select_dtypes(include=np.number).columns
    X_pl[ncp] = X_pl[ncp].fillna(X_pl[ncp].mean()).fillna(0)
    X_pl_s = scaler_planner.transform(X_pl)
    df_future["llamadas_hora"] = model_planner.predict(X_pl_s).clip(0).astype(int)
    print("  [OK] Llamadas predichas.")

    # 6) TMO Autorregresivo verdadero
    print("\n--- Fase 6: TMO (Autorregresivo v7) ---")
    if df_tmo_hist.empty:
        raise ValueError("El TMO autoregresivo requiere TMO_HISTORICO.csv no vacío.")

    df_tmo_hist_proc = df_tmo_hist.copy().sort_values("ts")
    if df_tmo_hist_proc["ts"].dt.tz is None:
        df_tmo_hist_proc["ts"] = df_tmo_hist_proc["ts"].dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")

    if TARGET_TMO not in df_tmo_hist_proc.columns:
        raise ValueError(f"No se encontró '{TARGET_TMO}' en TMO_HISTORICO.csv.")

    # Serie continua para lags
    hist_s = pd.Series(df_tmo_hist_proc[TARGET_TMO].values,
                       index=df_tmo_hist_proc["ts"].dt.floor("h"))
    hist_s = _ensure_hourly_continuous(hist_s, tz=TZ)

    # bootstrap mínimo 168h
    min_boot = 168
    if len(hist_s) < min_boot:
        if len(hist_s) == 0:
            raise ValueError("Histórico TMO vacío tras normalizar. No se puede inicializar lags.")
        pad_hours = min_boot - len(hist_s)
        pad_idx = pd.date_range(end=hist_s.index.min() - pd.Timedelta(hours=1),
                                periods=pad_hours, freq="h", tz=TZ)
        pad_series = pd.Series(hist_s.iloc[0], index=pad_idx).sort_index()
        hist_s = pd.concat([pad_series, hist_s]).sort_index()

    # semillas desde última fila histórica
    last_row = df_tmo_hist_proc.iloc[-1]
    seed_vals = {
        "proporcion_comercial": last_row["proporcion_comercial"] if "proporcion_comercial" in df_tmo_hist_proc.columns else 0.0,
        "proporcion_tecnica":   last_row["proporcion_tecnica"]   if "proporcion_tecnica"   in df_tmo_hist_proc.columns else 0.0,
        "tmo_comercial":        last_row["tmo_comercial"]        if "tmo_comercial"        in df_tmo_hist_proc.columns else 0.0,
        "tmo_tecnico":          last_row["tmo_tecnico"]          if "tmo_tecnico"          in df_tmo_hist_proc.columns else 0.0,
    }

    # Si el modelo espera anomalias_* (clima) y NO quieres conectarlas al TMO, fuerza a 0
    anom_cols = [c for c in cols_tmo if c.startswith("anomalia_")]
    for c in anom_cols:
        if c not in df_future.columns:
            df_future[c] = 0.0
        else:
            # si existen por Fase 4, igualmente las forzamos a 0 para desconectar clima del TMO
            df_future[c] = 0.0

    # Índices
    future_idx = df_future["ts"].dt.floor("h")
    full_idx = pd.date_range(start=hist_s.index.max() - pd.Timedelta(hours=min_boot - 1),
                             end=future_idx.max(), freq="h", tz=TZ)
    backfill_start = hist_s.index.max() - pd.Timedelta(hours=min_boot - 1)
    hist_boot = hist_s[hist_s.index >= backfill_start]
    hist_boot = hist_boot.reindex(pd.date_range(start=backfill_start, end=hist_s.index.max(), freq="h", tz=TZ)).ffill()

    tmo_full = pd.Series(index=full_idx, dtype="float64")
    tmo_full.loc[hist_boot.index] = hist_boot.values

    preds = []
    for ts in future_idx:
        # lags
        def _safe(ts_lag):
            return tmo_full.loc[ts_lag] if ts_lag in tmo_full.index else tmo_full.ffill().iloc[-1]

        lag_24  = _safe(ts - pd.Timedelta(hours=24))
        lag_48  = _safe(ts - pd.Timedelta(hours=48))
        lag_72  = _safe(ts - pd.Timedelta(hours=72))
        lag_168 = _safe(ts - pd.Timedelta(hours=168))

        # MAs
        prev_end = ts - pd.Timedelta(hours=1)
        if prev_end in tmo_full.index:
            ma_24  = _rolling_mean_safe(tmo_full.loc[ts - pd.Timedelta(hours=23): prev_end], 24).iloc[-1]
            ma_72  = _rolling_mean_safe(tmo_full.loc[ts - pd.Timedelta(hours=71): prev_end], 72).iloc[-1]
            ma_168 = _rolling_mean_safe(tmo_full.loc[ts - pd.Timedelta(hours=167): prev_end], 168).iloc[-1]
        else:
            ma_24 = ma_72 = ma_168 = lag_24

        # fila base
        row = df_future.loc[df_future["ts"].dt.floor("h") == ts].iloc[0].copy()
        # trig si estuvieran en cols_tmo y no en row (normalmente add_time_parts ya las tiene)
        if "sin_hour" in cols_tmo and "sin_hour" not in row:
            row["sin_hour"] = np.sin(2 * np.pi * row["hour"] / 24)
        if "cos_hour" in cols_tmo and "cos_hour" not in row:
            row["cos_hour"] = np.cos(2 * np.pi * row["hour"] / 24)
        if "sin_dow" in cols_tmo and "sin_dow" not in row:
            row["sin_dow"] = np.sin(2 * np.pi * row["dow"] / 7)
        if "cos_dow" in cols_tmo and "cos_dow" not in row:
            row["cos_dow"] = np.cos(2 * np.pi * row["dow"] / 7)

        # exógenas
        row[TARGET_CALLS] = row["llamadas_hora"]
        row["proporcion_comercial"] = seed_vals["proporcion_comercial"]
        row["proporcion_tecnica"]   = seed_vals["proporcion_tecnica"]
        if "tmo_comercial" in cols_tmo:
            row["tmo_comercial"] = seed_vals["tmo_comercial"]
        if "tmo_tecnico" in cols_tmo:
            row["tmo_tecnico"] = seed_vals["tmo_tecnico"]

        # lags/MA en fila
        row["lag_tmo_24"]  = float(lag_24)
        row["lag_tmo_48"]  = float(lag_48)
        row["lag_tmo_72"]  = float(lag_72)
        row["lag_tmo_168"] = float(lag_168)
        row["ma_tmo_24"]   = float(ma_24)
        row["ma_tmo_72"]   = float(ma_72)
        row["ma_tmo_168"]  = float(ma_168)

        # dummies + alineación a training_columns_tmo.json
        row_df = pd.DataFrame([row])
        row_df = pd.get_dummies(row_df, columns=["dow", "month", "hour"])
        row_df = row_df.reindex(columns=cols_tmo, fill_value=0)

        X_s = scaler_tmo.transform(row_df.values)
        y_hat = float(model_tmo.predict(X_s, verbose=0).reshape(-1)[0])
        y_hat = max(0.0, y_hat)

        tmo_full.loc[ts] = y_hat
        preds.append((ts, y_hat))

    tmo_df = pd.DataFrame(preds, columns=["ts", "tmo_hora"]).set_index("ts")
    df_future = df_future.set_index(df_future["ts"].dt.floor("h"))
    df_future["tmo_hora"] = tmo_df["tmo_hora"]
    df_future = df_future.reset_index(drop=True)
    print("  [OK] TMO AR generado.")

    # 7) Salidas JSON
    print("\n--- Fase 7: Salidas ---")
    os.makedirs(PUBLIC_DIR, exist_ok=True)

    # Agentes (Erlang simple)
    df_future["agentes_requeridos"] = calculate_erlang_agents(df_future["llamadas_hora"], df_future["tmo_hora"])
    df_horaria = df_future[["ts", "llamadas_hora", "tmo_hora", "agentes_requeridos"]].copy()
    df_horaria["ts"] = df_horaria["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out_h = os.path.join(PUBLIC_DIR, "prediccion_horaria.json")
    df_horaria.to_json(out_h, orient="records", indent=2, force_ascii=False)
    print(f"  [OK] {out_h}")

    # Diario ponderado por llamadas
    aux = df_future.copy()
    aux["fecha"] = aux["ts"].dt.date
    aux["tmo_pond_num"] = aux["tmo_hora"] * aux["llamadas_hora"]
    df_d = aux.groupby("fecha").agg(
        llamadas_totales_dia=("llamadas_hora", "sum"),
        tmo_pond_num=("tmo_pond_num", "sum"),
    )
    df_d["tmo_promedio_diario"] = df_d["tmo_pond_num"] / (df_d["llamadas_totales_dia"] + 1e-6)
    if (df_d["llamadas_totales_dia"] == 0).any():
        tmo_simple = aux.groupby("fecha")["tmo_hora"].mean().fillna(0)
        df_d["tmo_promedio_diario"] = df_d["tmo_promedio_diario"].where(df_d["llamadas_totales_dia"] > 0, tmo_simple).fillna(0)
    df_d = df_d.reset_index()[["fecha", "llamadas_totales_dia", "tmo_promedio_diario"]]
    df_d["fecha"] = df_d["fecha"].astype(str)
    df_d["llamadas_totales_dia"] = df_d["llamadas_totales_dia"].astype(int)
    out_d = os.path.join(PUBLIC_DIR, "Predicion_daria.json")
    df_d.to_json(out_d, orient="records", indent=2, force_ascii=False)
    print(f"  [OK] {out_d}")

    # Alertas clima (a partir de riesgo)
    df_risk_out = df_future[["ts", "risk_proba"]].copy()
    alerts = generate_alerts_json(df_per_comuna_anoms, df_risk_out)
    out_alert = os.path.join(PUBLIC_DIR, "alertas_climaticas.json")
    with open(out_alert, "w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {out_alert}")

    print("\n" + "=" * 60)
    print("INFERENCIA COMPLETADA")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia v7")
    parser.add_argument("--horizonte", type=int, default=120, help="Horizonte de predicción en días")
    args = parser.parse_args()
    main(horizonte_dias=args.horizonte)
