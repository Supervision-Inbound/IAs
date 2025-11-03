import os
import glob
import pathlib
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# Nota: Asumo que las funciones importadas (dummies_and_reindex, add_time_parts, etc.) están disponibles.
from .features import ensure_ts, add_time_parts, dummies_and_reindex 
from .erlang import required_agents, schedule_agents
from .utils_io import write_daily_json, write_hourly_json

TIMEZONE = "America/Santiago"
PUBLIC_DIR = "public"

# --- Constantes LSTM ---
LOOKBACK_STEPS = 168 
HORIZON_STEPS = 24  

# --- (El resto de las funciones auxiliares como _candidate_dirs, _find_one, _resolve_tmo_artifacts, 
#      _load_cols, _safe_ratio, compute_holiday_factors, apply_holiday_adjustment, 
#      _baseline_median_mad, apply_outlier_cap, _is_holiday, y generate_features van aquí sin cambios) ---

# ... [INICIO DEL CÓDIGO AUXILIAR QUE NO CAMBIÓ] ...

def _candidate_dirs():
    here = pathlib.Path(".").resolve()
    bases = [
        here,
        here / "models",
        here / "modelos",
        here / "release",
        here / "releases",
        here / "artifacts",
        here / "outputs",
        here / "output",
        here / "dist",
        here / "build",
        pathlib.Path(os.environ.get("GITHUB_WORKSPACE", ".")),
        pathlib.Path("/kaggle/working/models"),
        pathlib.Path("/kaggle/working"),
    ]
    uniq = []
    for p in bases:
        try:
            p = p.resolve()
            if p.exists() and p not in uniq:
                uniq.append(p)
        except Exception:
            pass
    return uniq

def _find_one(patterns, search_dirs=None):
    if isinstance(patterns, str):
        patterns = [patterns]
    search_dirs = search_dirs or _candidate_dirs()
    for d in search_dirs:
        for pat in patterns:
            for match in glob.glob(str(d / "**" / pat), recursive=True):
                p = pathlib.Path(match)
                if p.is_file():
                    return str(p)
    return None

def _resolve_tmo_artifacts():
    return {
        "keras": _find_one(["modelo_tmo.keras", "tmo*.keras", "*_tmo*.keras"]),
        "scaler": _find_one(["scaler_tmo.pkl", "*tmo*scaler*.pkl", "*scaler*_tmo*.pkl"]),
        "cols": _find_one(["training_columns_tmo.json", "*tmo*columns*.json", "*training*columns*to*.json"]),
    }

_paths = _resolve_tmo_artifacts()

# --- Rutas de Artefactos (Sin cambios) ---
PLANNER_MODEL = "models/modelo_planner.keras"
PLANNER_SCALER = "models/scaler_planner.pkl"
PLANNER_COLS   = "models/training_columns_planner.json"

TMO_MODEL    = _paths.get("keras")    or "models/modelo_tmo.keras"
TMO_SCALER   = _paths.get("scaler")   or "models/scaler_tmo.pkl"
TMO_COLS     = _paths.get("cols")     or "models/training_columns_tmo.json"

# --- Negocio (Sin cambios) ---
TARGET_CALLS = "recibidos_nacional"
TARGET_TMO   = "tmo_general"
ENABLE_OUTLIER_CAP = True
K_WEEKDAY = 6.0
K_WEEKEND = 7.0

# (Funciones auxiliares, solo muestro un par, asumir que el resto va aquí)

def _load_cols(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ... [TODAS LAS FUNCIONES compute_holiday_factors, apply_holiday_adjustment, 
#      _baseline_median_mad, apply_outlier_cap, _is_holiday, etc., VAN AQUÍ SIN CAMBIOS] ...

def add_time_parts(df):
    # Función dummy para simular la importación (Asegúrate que la tuya está bien)
    d = df.copy()
    d["dow"] = d.index.dayofweek
    d["month"] = d.index.month
    d["hour"] = d.index.hour
    return d

def generate_features(df, target_calls, target_tmo, feriados_col):
    d = add_time_parts(df.copy())
    
    # Lógica de Feriados
    d['es_post_feriado'] = ((d[feriados_col].shift(1).fillna(0) == 1) & (d[feriados_col] == 0)).astype(int)
    d['es_pre_feriado'] = ((d[feriados_col].shift(-1).fillna(0) == 1) & (d[feriados_col] == 0)).astype(int)
    
    # Lógica de Planner
    s_calls = d[target_calls]
    for lag in [24, 48, 72, 168]:
        d[f'lag_{lag}'] = s_calls.shift(lag)
    for window in [24, 72, 168]:
        d[f'ma_{window}'] = s_calls.rolling(window, min_periods=1).mean()
        
    # Lógica de TMO
    s_tmo_total = d[target_tmo]
    for lag in [1, 2, 3, 6, 12, 24, 168]: d[f"lag_tmo_total_{lag}"] = s_tmo_total.shift(lag)
    s_tmo_total_s1 = s_tmo_total.shift(1)
    for w in [6, 12, 24, 72]: d[f"ma_tmo_total_{w}"] = s_tmo_total_s1.rolling(w, min_periods=1).mean()
    
    s_contest = d[target_calls] # Usamos 'recibidos' como proxy de 'contestadas'
    for lag in [1, 24, 48, 168]: d[f"lag_contest_{lag}"] = s_contest.shift(lag)
    s_contest_s1 = s_contest.shift(1)
    for w in [6, 24, 72]: d[f"ma_contest_{w}"] = s_contest_s1.rolling(w, min_periods=1).mean()
        
    return d

# ... [FIN DEL CÓDIGO AUXILIAR QUE NO CAMBIÓ] ...


# ---------- Núcleo (MODIFICADO para corregir KeyError) ----------
def forecast_120d(df_hist_joined: pd.DataFrame, 
                  horizon_days: int = 120, holidays_set: set | None = None):
    
    # 1. Cargar Artefactos
    print("Cargando artefactos LSTM...")
    m_pl = tf.keras.models.load_model(PLANNER_MODEL)
    sc_pl = joblib.load(PLANNER_SCALER)
    cols_pl = _load_cols(PLANNER_COLS)

    m_tmo = tf.keras.models.load_model(TMO_MODEL)
    sc_tmo = joblib.load(TMO_SCALER)
    cols_tmo = _load_cols(TMO_COLS)

    # 2. Preparar Dataframe Histórico (dfp)
    df = ensure_ts(df_hist_joined)
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
    df.index = df.index.tz_convert(TIMEZONE)

    # Columnas base necesarias
    if TARGET_CALLS not in df.columns: df[TARGET_CALLS] = 0
    if TARGET_TMO not in df.columns: df[TARGET_TMO] = 0
    if "feriados" not in df.columns: df["feriados"] = 0
    
    # Generar *todos* los features en el histórico (Lags, MAs, etc.)
    dfp = generate_features(df, TARGET_CALLS, TARGET_TMO, "feriados")
    
    # 🚨 CORRECCIÓN DEL KEY ERROR APLICADA AQUÍ: 
    # 1. Asegurar que las columnas de tiempo existen para el get_dummies
    dfp = add_time_parts(dfp)
    
    # 2. Generar las DUMMIES (one-hot encoding) para que existan las columnas 'dow_0', 'month_1', etc.
    dfp = pd.get_dummies(dfp, columns=['dow', 'month', 'hour'], drop_first=False, dtype=float)

    # 3. Reindexar el histórico para asegurar que tenga *exactamente* el mismo orden y columnas que training
    all_cols_expected = list(set(cols_pl) | set(cols_tmo)) # Combina todas las columnas esperadas
    dfp = dfp.reindex(columns=all_cols_expected, fill_value=np.nan) # Rellena las faltantes con NaN
    
    # Rellenar los NaN (provenientes de lags/MAs y de las dummies creadas por reindex/get_dummies)
    # dfp.fillna(0.0) es más robusto que sub-seleccionar, pero mantendremos la tuya:
    dfp[cols_pl] = dfp[cols_pl].fillna(0.0)
    dfp[cols_tmo] = dfp[cols_tmo].fillna(0.0)
    
    last_ts = dfp.index.max()
    
    # ===== Bucle de Inferencia (1 paso por DÍA, no por hora) =====
    print(f"Iniciando predicción iterativa (LSTM) para {horizon_days} días...")
    
    future_predictions = []
    
    for i in range(horizon_days):
        print(f"Prediciendo Día {i+1}/{horizon_days}...")
        
        # 1. Preparar Ventana de Entrada (Input)
        # Tomamos los últimos 168 pasos (7 días)
        input_df = dfp.iloc[-LOOKBACK_STEPS:]
        
        # 2. Preparar Input del Planner
        # El input_df YA TIENE LAS DUMMIES, solo necesita ser escalado
        input_features_pl = input_df[cols_pl] # Usamos el subconjunto de columnas
        input_scaled_pl = sc_pl.transform(input_features_pl)
        input_lstm_pl = input_scaled_pl.reshape((1, LOOKBACK_STEPS, len(cols_pl)))
        
        # 3. Preparar Input del TMO
        input_features_tmo = input_df[cols_tmo] # Usamos el subconjunto de columnas
        input_scaled_tmo = sc_tmo.transform(input_features_tmo)
        input_lstm_tmo = input_scaled_tmo.reshape((1, LOOKBACK_STEPS, len(cols_tmo)))

        # 4. Predecir 24 horas de golpe
        yhat_calls_24h = m_pl.predict(input_lstm_pl, verbose=0).flatten()
        yhat_tmo_24h = m_tmo.predict(input_lstm_tmo, verbose=0).flatten()
        
        # 5. Crear Dataframe para las próximas 24 horas
        future_index = pd.date_range(
            start=input_df.index.max() + pd.Timedelta(hours=1),
            periods=HORIZON_STEPS,
            freq="h",
            tz=TIMEZONE
        )
        
        df_future_day = pd.DataFrame(index=future_index)
        df_future_day[TARGET_CALLS] = np.maximum(0, yhat_calls_24h) 
        df_future_day[TARGET_TMO] = np.maximum(0, yhat_tmo_24h)  
        
        # 6. Generar "Known Future Features" (Calendario y Feriados) para este nuevo día
        df_future_day["ts"] = df_future_day.index
        df_future_day = add_time_parts(df_future_day)
        df_future_day["feriados"] = df_future_day.index.to_series().apply(lambda ts: _is_holiday(ts, holidays_set))
        
        # 7. Apendizar al histórico (dfp) para la *siguiente* iteración
        # Necesitamos generar los lags/MAs y las dummies para este nuevo día
        dfp_with_future = pd.concat([dfp, df_future_day])
        
        # Volver a generar features (lags/MAs)
        dfp_with_future = generate_features(dfp_with_future, TARGET_CALLS, TARGET_TMO, "feriados")

        # Volver a generar dummies y reindexar (para el bucle)
        dfp_with_future = add_time_parts(dfp_with_future)
        dfp_with_future = pd.get_dummies(dfp_with_future, columns=['dow', 'month', 'hour'], drop_first=False, dtype=float)
        dfp_with_future = dfp_with_future.reindex(columns=all_cols_expected, fill_value=0.0)
        
        # Actualizar dfp para el siguiente bucle
        dfp = dfp_with_future
        
        # Guardar las predicciones de este día
        future_predictions.append(dfp_with_future.iloc[-HORIZON_STEPS:])

    print("Predicción iterativa completada.")

    # ===== Salida horaria =====
    df_hourly = pd.concat(future_predictions)
    
    # ... [El resto de la lógica de salida, ajustes y Erlang va aquí] ...
    
    # Renombrar columnas a formato de salida
    df_hourly = df_hourly.rename(columns={TARGET_CALLS: "calls", TARGET_TMO: "tmo_s"})
    
    # Saneamiento final (por si acaso el modelo predijo NaN)
    df_hourly["calls"] = df_hourly["calls"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df_hourly["tmo_s"] = df_hourly["tmo_s"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    df_hourly["calls"] = np.round(df_hourly["calls"]).astype(int)
    df_hourly["tmo_s"] = np.round(df_hourly["tmo_s"]).astype(int)

    # ===== Ajustes feriados (Solo el día feriado en sí) =====
    if holidays_set and len(holidays_set) > 0:
        (f_calls_by_hour, f_tmo_by_hour, _gc, _gt) = compute_holiday_factors(
            df, holidays_set, col_calls=TARGET_CALLS, col_tmo=TARGET_TMO
        )
        # El modelo ya aprendió pre/post feriado, solo aplicamos el factor del día
        df_hourly = apply_holiday_adjustment(df_hourly, holidays_set, f_calls_by_hour, f_tmo_by_hour,
                                              col_calls_future="calls", col_tmo_future="tmo_s")

    # ===== CAP outliers (llamadas) =====
    if ENABLE_OUTLIER_CAP:
        base_mad = _baseline_median_mad(df, col=TARGET_CALLS)
        df_hourly = apply_outlier_cap(df_hourly, base_mad, holidays_set,
                                      col_calls_future="calls",
                                      k_weekday=K_WEEKDAY, k_weekend=K_WEEKEND)

    # ===== Erlang (Sin cambios) =====
    df_hourly["agents_prod"] = 0
    for ts in df_hourly.index:
        a, _ = required_agents(float(df_hourly.at[ts, "calls"]), float(df_hourly.at[ts, "tmo_s"]))
        df_hourly.at[ts, "agents_prod"] = int(a)
    df_hourly["agents_sched"] = df_hourly["agents_prod"].apply(schedule_agents)

    # ===== Salidas JSON (Sin cambios) =====
    write_hourly_json(f"{PUBLIC_DIR}/prediccion_horaria.json", df_hourly, "calls", "tmo_s", "agents_sched")
    write_daily_json(f"{PUBLIC_DIR}/prediccion_diaria.json", df_hourly, "calls", "tmo_s",
                      weights_col="calls")

    return df_hourly
