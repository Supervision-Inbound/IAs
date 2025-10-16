# src/inferencia/alertas_clima.py
import os, json, time
import numpy as np
import pandas as pd
import joblib, tensorflow as tf
import requests, requests_cache
from retry_requests import retry
from .features import add_time_parts
from .utils_io import write_json

TIMEZONE = "America/Santiago"
PUBLIC_DIR = "public"
COORDS_CSV = "data/Comunas_Coordenadas.csv"

# Modelos / artefactos de riesgos
RIESGOS_MODEL = "models/modelo_riesgos.keras"
RIESGOS_SCALER = "models/scaler_riesgos.pkl"
RIESGOS_COLS = "models/training_columns_riesgos.json"
CLIMA_BASELINES = "models/baselines_clima.pkl"

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = ["temperature_2m","precipitation","rain"]
UNITS = {"temperature_unit":"celsius","precipitation_unit":"mm"}

FORECAST_DAYS = 8
ALERT_Z = 2.5            # umbral z por comuna-hora
UPLIFT_ALPHA = 0.25      # factor de conversión prob_evento → % extra

def _load_cols(path): 
    with open(path,"r") as f: return json.load(f)

def _client():
    sess = requests_cache.CachedSession(".openmeteo_cache", expire_after=3600)
    return retry(sess, retries=3, backoff_factor=1.5)

def _read_coords(path):
    df = pd.read_csv(path)
    # columnas laxas
    def pick(cols, cand):
        m = {c.lower().strip(): c for c in cols}
        for k in cand:
            if k in m: return m[k]
        return None
    c = pick(df.columns, ["comuna","municipio","localidad","ciudad","name","nombre"])
    la = pick(df.columns, ["lat","latitude","latitud","y"])
    lo = pick(df.columns, ["lon","lng","long","longitude","longitud","x"])
    if not c or not la or not lo:
        raise ValueError(f"CSV coords debe contener comuna/lat/lon. Tiene: {list(df.columns)}")
    df = df.rename(columns={c:"comuna", la:"lat", lo:"lon"})
    for k in ["lat","lon"]:
        if df[k].dtype == object:
            df[k] = df[k].astype(str).str.replace(",",".",regex=False)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat","lon"]).drop_duplicates("comuna")
    return df.reset_index(drop=True)

def _fetch_forecast(sess, lat, lon):
    params = dict(latitude=float(lat), longitude=float(lon),
                  hourly=",".join(HOURLY_VARS),
                  forecast_days=int(FORECAST_DAYS),
                  timezone=TIMEZONE, **UNITS)
    r = sess.get(OPEN_METEO_URL, params=params); r.raise_for_status()
    js = r.json()
    times = pd.to_datetime(js["hourly"]["time"])
    df = pd.DataFrame({"ts": times})
    for v in HOURLY_VARS: df[v] = js["hourly"].get(v, np.nan)
    return df

def _anomalias_vs_baseline(df_clima, baselines):
    # baselines: groupby(['comuna','dow','hour']) ['temperatura','precipitacion','lluvia'] median/std
    d = add_time_parts(df_clima.set_index("ts"))
    d = d.reset_index()
    mcols = ["temperatura","precipitacion","lluvia"]
    # renombrar entradas a estándar
    rename = {"temperature_2m":"temperatura", "rain":"lluvia"}
    d = d.rename(columns=rename)
    for metric in mcols:
        if metric not in d.columns: d[metric] = np.nan
    # join a baseline
    b = baselines.copy()
    # columnas del baseline vienen como metric_median, metric_std
    d = d.merge(b, left_on=["comuna","dow","hour"], right_on=["comuna_","dow_","hour_"], how="left")
    for metric in mcols:
        med = f"{metric}_median"; std = f"{metric}_std"
        if med in d.columns and std in d.columns:
            d[f"z_{metric}"] = (d[metric] - d[med]) / (d[std] + 1e-6)
        else:
            d[f"z_{metric}"] = 0.0
    return d

def generar_alertas(pred_calls_hourly: pd.DataFrame) -> None:
    """
    pred_calls_hourly: DataFrame index ts con columna 'calls' (predicción planner) para asignar uplift.
    """
    # Cargar artefactos de riesgos
    m = tf.keras.models.load_model(RIESGOS_MODEL, compile=False)
    sc = joblib.load(RIESGOS_SCALER)
    cols = _load_cols(RIESGOS_COLS)
    baselines = joblib.load(CLIMA_BASELINES)

    coords = _read_coords(COORDS_CSV)
    sess = _client()

    # 1) Descargar clima por comuna y calcular z-scores
    registros = []
    for _, r in coords.iterrows():
        comuna, lat, lon = r["comuna"], r["lat"], r["lon"]
        dfc = _fetch_forecast(sess, lat, lon)
        dfc["comuna"] = comuna
        registros.append(dfc)
        time.sleep(0.2)
    clima = pd.concat(registros, ignore_index=True)
    clima["ts"] = pd.to_datetime(clima["ts"]).dt.tz_localize(TIMEZONE)

    zed = _anomalias_vs_baseline(clima, baselines)

    # 2) Features agregadas (matching training_columns_riesgos)
    n_comunas = coords["comuna"].nunique()
    agg = zed.groupby("ts").agg({
        "z_temperatura":["max","sum", lambda x: (x>ALERT_Z).sum()/max(1,n_comunas)],
        "z_precipitacion":["max","sum", lambda x: (x>ALERT_Z).sum()/max(1,n_comunas)],
        "z_lluvia":["max","sum", lambda x: (x>ALERT_Z).sum()/max(1,n_comunas)],
    })
    agg.columns = [
        "anomalia_temperatura_max","anomalia_temperatura_sum","anomalia_temperatura_pct_comunas_afectadas",
        "anomalia_precipitacion_max","anomalia_precipitacion_sum","anomalia_precipitacion_pct_comunas_afectadas",
        "anomalia_lluvia_max","anomalia_lluvia_sum","anomalia_lluvia_pct_comunas_afectadas"
    ]
    agg = agg.reindex(columns=cols, fill_value=0.0).fillna(0.0)
    proba_evento = m.predict(sc.transform(agg.values), verbose=0).flatten()
    proba = pd.Series(proba_evento, index=agg.index)

    # 3) Para cada comuna, horas con alerta por z-score y uplift vs planner
    salida = []
    zed["score_comuna"] = zed[["z_temperatura","z_precipitacion","z_lluvia"]].clip(lower=0).mean(axis=1)
    for comuna, dfc in zed.groupby("comuna"):
        dfc = dfc.sort_values("ts")
        dfc["alerta"] = dfc["score_comuna"] > ALERT_Z
        dfc["proba_global"] = proba.reindex(dfc["ts"]).fillna(0.0).values
        # uplift = proba_global * alpha * score_norm * planner_calls
        max_s = max(dfc["score_comuna"].max(), ALERT_Z)
        score_norm = (dfc["score_comuna"] / max_s).clip(0,1)
        planner = pred_calls_hourly.reindex(dfc["ts"]).fillna(method="ffill").fillna(0)["calls"].values
        extra = np.round(dfc["proba_global"] * UPLIFT_ALPHA * score_norm * planner).astype(int)
        dfc["extra_calls"] = extra

        # agrupar en rangos por día
        d = dfc[["ts","alerta","extra_calls"]].copy()
        d["fecha"] = d["ts"].dt.date
        d["hora"] = d["ts"].dt.hour
        d = d[d["alerta"]]
        rangos = []
        if not d.empty:
            cur_date, h0, h1, vals = None, None, None, []
            for _, r0 in d.iterrows():
                f, h, v = r0["fecha"], int(r0["hora"]), int(r0["extra_calls"])
                if cur_date is None:
                    cur_date, h0, h1, vals = f, h, h, [v]; continue
                if f == cur_date and h == h1 + 1:
                    h1, vals = h, vals+[v]
                else:
                    rangos.append({
                        "fecha": str(cur_date), "hora_inicio": h0, "hora_fin": h1,
                        "impacto_llamadas_adicionales": int(max(sum(vals), 0))
                    })
                    cur_date, h0, h1, vals = f, h, h, [v]
            rangos.append({
                "fecha": str(cur_date), "hora_inicio": h0, "hora_fin": h1,
                "impacto_llamadas_adicionales": int(max(sum(vals), 0))
            })
        salida.append({"comuna": comuna, "rango_alertas": rangos})

    # Ordenar: con alertas primero
    salida.sort(key=lambda x: (len(x["rango_alertas"]) == 0, x["comuna"]))
    write_json(f"{PUBLIC_DIR}/alertas_clima.json", salida)

