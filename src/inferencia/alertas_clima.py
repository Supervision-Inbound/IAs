# src/inferencia/alertas_clima.py
import os, json, time
import numpy as np
import pandas as pd
import joblib, tensorflow as tf
import requests, requests_cache

# Fallback de retry: usa retry-requests si está disponible; si no, urllib3 Retry
try:
    from retry_requests import retry  # pip install retry-requests
    def _wrap_retry(session, retries=3, backoff_factor=1.5):
        return retry(session, retries=retries, backoff_factor=backoff_factor)
except Exception:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    def _wrap_retry(session, retries=3, backoff_factor=1.5):
        r = Retry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=False
        )
        session.mount("https://", HTTPAdapter(max_retries=r))
        session.mount("http://", HTTPAdapter(max_retries=r))
        return session

from .features import add_time_parts
from .utils_io import write_json

TIMEZONE = "America/Santiago"
PUBLIC_DIR = "public"

# Aceptar ambos nombres (Coordenadas vs Cordenadas)
CANDIDATE_COORDS = [
    "data/Comunas_Coordenadas.csv",   # nombre correcto
    "data/Comunas_Cordenadas.csv",    # variante en tu repo
]

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
UPLIFT_ALPHA = 0.25      # factor de conversión proba_evento → % extra

def _load_cols(path):
    with open(path,"r") as f:
        return json.load(f)

def _client():
    sess = requests_cache.CachedSession(".openmeteo_cache", expire_after=3600)
    return _wrap_retry(sess, retries=3, backoff_factor=1.5)

def _resolve_coords_path() -> str | None:
    for p in CANDIDATE_COORDS:
        if os.path.exists(p):
            return p
    return None

def _read_csv_smart(path: str) -> pd.DataFrame:
    # Intenta varios encodings y separadores, como en tu inferencia antigua
    encodings = ("utf-8", "utf-8-sig", "latin1", "cp1252")
    seps = [";", ",", "\t", "|"]
    last_err = None
    for enc in encodings:
        try:
            # intento auto
            df = pd.read_csv(path, encoding=enc, engine="python")
            if df.shape[1] > 1:
                return df
        except Exception as e:
            last_err = e
        # intentos forzando separador
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                if df.shape[1] > 1:
                    return df
            except Exception as e2:
                last_err = e2
                continue
    if last_err:
        raise last_err
    raise ValueError(f"No pude leer {path} con encodings/separadores estándar.")

def _pick_col(cols, candidates):
    m = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        key = cand.lower().strip()
        for k, orig in m.items():
            if key == k or key in k:
                return orig
    return None

def _read_coords() -> pd.DataFrame:
    path = _resolve_coords_path()
    if not path:
        return pd.DataFrame()  # handled upstream (no romper)
    df = _read_csv_smart(path)
    df.columns = [c.strip() for c in df.columns]

    comuna_col = _pick_col(df.columns, ["comuna","municipio","localidad","ciudad","name","nombre"])
    lat_col    = _pick_col(df.columns, ["lat","latitude","latitud","y"])
    lon_col    = _pick_col(df.columns, ["lon","lng","long","longitude","longitud","x"])

    if not comuna_col or not lat_col or not lon_col:
        raise ValueError(f"CSV de coordenadas debe tener comuna/lat/lon. Columnas: {list(df.columns)}")

    df = df.rename(columns={comuna_col:"comuna", lat_col:"lat", lon_col:"lon"}).copy()
    for c in ["lat","lon"]:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace(",", ".", regex=False).str.strip()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat","lon"])
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]
    df = df.drop_duplicates(subset=["comuna"]).reset_index(drop=True)
    return df

def _fetch_forecast(sess, lat, lon):
    params = dict(
        latitude=float(lat), longitude=float(lon),
        hourly=",".join(HOURLY_VARS),
        forecast_days=int(FORECAST_DAYS),
        timezone=TIMEZONE, **UNITS
    )
    r = sess.get(OPEN_METEO_URL, params=params)
    r.raise_for_status()
    js = r.json()
    times = pd.to_datetime(js["hourly"]["time"])
    df = pd.DataFrame({"ts": times})
    for v in HOURLY_VARS:
        df[v] = js["hourly"].get(v, np.nan)
    return df

def _anomalias_vs_baseline(df_clima, baselines):
    # baselines con columnas: metric_median, metric_std por (comuna,dow,hour)
    d = add_time_parts(df_clima.set_index("ts"))
    d = d.reset_index()
    # normalizar nombres de clima
    d = d.rename(columns={"temperature_2m":"temperatura", "rain":"lluvia"})
    for metric in ["temperatura","precipitacion","lluvia"]:
        if metric not in d.columns:
            d[metric] = np.nan
    b = baselines.copy()
    d = d.merge(b, left_on=["comuna","dow","hour"], right_on=["comuna_","dow_","hour_"], how="left")
    for metric in ["temperatura","precipitacion","lluvia"]:
        med = f"{metric}_median"; std = f"{metric}_std"
        if med in d.columns and std in d.columns:
            d[f"z_{metric}"] = (d[metric] - d[med]) / (d[std] + 1e-6)
        else:
            d[f"z_{metric}"] = 0.0
    return d

def generar_alertas(pred_calls_hourly: pd.DataFrame) -> None:
    """
    pred_calls_hourly: DataFrame index=ts con columna 'calls' (predicción planner) para calcular uplift adicional.
    Si no hay CSV de coordenadas, genera alertas_clima.json vacío y sale sin error.
    """
    # Si no hay coords, escribe salida vacía y termina
    coords = _read_coords()
    if coords.empty:
        write_json(f"{PUBLIC_DIR}/alertas_clima.json", [])
        print("⚠️ No se encontró data/Comunas_Coordenadas.csv ni data/Comunas_Cordenadas.csv. Se generó alertas_clima.json vacío.")
        return

    # Cargar artefactos
    m = tf.keras.models.load_model(RIESGOS_MODEL, compile=False)
    sc = joblib.load(RIESGOS_SCALER)
    cols = _load_cols(RIESGOS_COLS)
    baselines = joblib.load(CLIMA_BASELINES)

    sess = _client()

    # 1) Descargar clima por comuna
    registros = []
    for _, r in coords.iterrows():
        comuna, lat, lon = r["comuna"], r["lat"], r["lon"]
        dfc = _fetch_forecast(sess, lat, lon)
        dfc["comuna"] = comuna
        registros.append(dfc)
        time.sleep(0.2)
    clima = pd.concat(registros, ignore_index=True)
    clima["ts"] = pd.to_datetime(clima["ts"]).dt.tz_localize(TIMEZONE)

    # 2) Z-scores vs baseline y agregados a nivel ts (matching training columns)
    zed = _anomalias_vs_baseline(clima, baselines)
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

    # Asegurar DataFrame con mismas columnas y orden que el scaler/modelo
    agg = agg.reindex(columns=cols, fill_value=0.0).astype(float)

    # 3) Probabilidad de evento alto volumen (usar DataFrame, no .values)
    Xs = sc.transform(agg)  # conserva nombres de columnas
    proba_evento = m.predict(Xs, verbose=0).flatten()
    proba = pd.Series(proba_evento, index=agg.index)

    # 4) Construir salida por comuna con uplift vs planner
    salida = []
    zed["score_comuna"] = zed[["z_temperatura","z_precipitacion","z_lluvia"]].clip(lower=0).mean(axis=1)

    # Normalizar el índice del planner por si viene naive/otra zona
    pred_calls_hourly = pred_calls_hourly.copy()
    if getattr(pred_calls_hourly.index, "tz", None) is None:
        pred_calls_hourly.index = pred_calls_hourly.index.tz_localize(TIMEZONE)

    for comuna, dfc in zed.groupby("comuna"):
        dfc = dfc.sort_values("ts")
        dfc["alerta"] = dfc["score_comuna"] > ALERT_Z

        # Alinear proba_global y planner
        dfc["proba_global"] = proba.reindex(dfc["ts"]).ffill().fillna(0.0).values
        planner_series = pred_calls_hourly.reindex(dfc["ts"]).ffill().fillna(0.0)["calls"].astype(float).values

        # score normalizado y robustez numérica
        max_s = float(max(dfc["score_comuna"].max(), ALERT_Z))
        score_norm = (dfc["score_comuna"] / max_s).clip(0, 1).astype(float)

        # Uplift robusto: reemplaza NaN/inf por 0, garantiza no-negatividad
        extra_raw = dfc["proba_global"].astype(float).values * float(UPLIFT_ALPHA) * score_norm.values * planner_series
        extra_clean = np.nan_to_num(extra_raw, nan=0.0, posinf=0.0, neginf=0.0)
        extra_clean = np.clip(extra_clean, 0, np.iinfo(np.int32).max)
        dfc["extra_calls"] = np.round(extra_clean).astype(int)

        # agrupar horas consecutivas en rangos por día
        d = dfc.loc[dfc["alerta"], ["ts","extra_calls"]].copy()
        d["fecha"] = d["ts"].dt.date
        d["hora"] = d["ts"].dt.hour

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

    # Ordenar: comunas con alertas primero
    salida.sort(key=lambda x: (len(x["rango_alertas"]) == 0, x["comuna"]))
    write_json(f"{PUBLIC_DIR}/alertas_clima.json", salida)
