# src/inferencia/inferencia_core.py
import json
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm # Importar tqdm

# --- IMPORTACIONES CORREGIDAS ---
from .features import ensure_ts, add_time_parts, add_lags_mas, dummies_and_reindex, mark_holidays_index, add_es_dia_de_pago
from .erlang import required_agents, schedule_agents
from .utils_io import write_daily_json, write_hourly_json, write_json

TIMEZONE = "America/Santiago"
PUBLIC_DIR = "public"

# --- Planner (Llamadas) ---
PLANNER_MODEL = "models/modelo_planner.keras"
PLANNER_SCALER = "models/scaler_planner.pkl"
PLANNER_COLS = "models/training_columns_planner.json"

# --- TMO (Nuevos artefactos v7-residual) ---
TMO_MODEL = "models/modelo_tmo.keras"
TMO_SCALER = "models/scaler_tmo.pkl"
TMO_COLS = "models/training_columns_tmo.json"
TMO_BASELINE = "models/tmo_baseline_dow_hour.csv" # <-- NUEVO
TMO_META = "models/tmo_residual_meta.json"      # <-- NUEVO

# --- ¡¡AQUÍ ESTÁ LA CORRECCIÓN!! ---
TARGET_CALLS = "recibidos" # <- Ajustado a tu CSV (era "recibidos_nacional")
TARGET_TMO = "tmo_general" # Este es el nombre estándar (ej: 'tmo_general')
# ------------------------------------

# Ventana reciente para lags/MA
HIST_WINDOW_DAYS = 90

# (Configuración de Outliers)
ENABLE_OUTLIER_CAP = True
K_WEEKDAY = 6.0
K_WEEKEND = 7.0


def _load_cols(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# (Helpers de Outliers - Asumimos que están aquí)
def _baseline_median_mad(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Calcula mediana y MAD por (dow, hour) del histórico."""
    # Usar solo datos recientes (ej. últimos 90 días) para el baseline
    df_b = df.loc[df.index > df.index.max() - pd.DateOffset(days=90)].copy()
    if 'dow' not in df_b.columns or 'hour' not in df_b.columns:
        df_b = add_time_parts(df_b) # Asegurar que existan dow y hour
    
    # Calcular mediana y MAD (Median Absolute Deviation)
    base = df_b.groupby(['dow','hour'])[col].agg(['median', 'mad']).reset_index()
    # Rellenar NaNs en MAD (si hay pocos datos) con la media del MAD
    base['mad'] = base['mad'].fillna(base['mad'].mean())
    return base.set_index(['dow','hour'])

def apply_outlier_cap(df_hourly: pd.DataFrame, baseline_mad: pd.DataFrame, 
                      holidays_set: set, col_calls_future: str,
                      k_weekday: float, k_weekend: float) -> pd.DataFrame:
    """Aplica un techo (cap) a las predicciones de llamadas basado en baseline + K*MAD."""
    df = df_hourly.copy()
    if baseline_mad.empty:
        print("WARN: No se puede aplicar Cap de Outliers, baseline_mad está vacío.")
        return df
        
    df = df.merge(baseline_mad, on=['dow','hour'], how='left')
    df['mad'] = df['mad'].fillna(method='ffill').fillna(method='bfill')
    df['median'] = df['median'].fillna(method='ffill').fillna(method='bfill')
    
    # Definir K (factor de desviación)
    df['k'] = np.where(df['dow'].isin([5, 6]), k_weekend, k_weekday) # Sáb/Dom = 5, 6
    # Calcular techo
    df['cap'] = df['median'] + df['k'] * df['mad']
    
    # No aplicar cap en feriados
    is_holiday = mark_holidays_index(df.index, holidays_set).values
    df.loc[is_holiday > 0, col_calls_future] = df.loc[is_holiday > 0, col_calls_future]
    
    # Aplicar cap solo donde no es feriado
    df.loc[is_holiday == 0, col_calls_future] = np.minimum(
        df.loc[is_holiday == 0, col_calls_future],
        df.loc[is_holiday == 0, 'cap']
    )
    
    # Asegurar no negativos
    df[col_calls_future] = np.maximum(df[col_calls_future], 0)
    
    return df.drop(columns=['median', 'mad', 'k', 'cap'])


# =================================================================
# NUEVO HELPER: Creación de features residuales (de train_v7.py)
# =================================================================
def _make_tmo_residual_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features autoregresivas sobre la columna 'tmo_resid'.
    El DataFrame de entrada 'df' DEBE tener 'tmo_resid'.
    """
    d = df.copy()
    
    # Lags del residuo
    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        d[f"lag_resid_{lag}"] = d["tmo_resid"].shift(lag)
    
    resid_shift1 = d["tmo_resid"].shift(1)
    
    # MAs del residuo
    for w in [6, 12, 24, 72, 168]:
        d[f"ma_resid_{w}"] = resid_shift1.rolling(w, min_periods=1).mean()
    
    # EMAs del residuo
    for span in [6, 12, 24]:
        d[f"ema_resid_{span}"] = resid_shift1.ewm(span=span, adjust=False, min_periods=1).mean()
    
    # Volatilidad del residuo
    for w in [24, 72]:
        d[f"std_resid_{w}"] = resid_shift1.rolling(w, min_periods=2).std()
        d[f"max_resid_{w}"] = resid_shift1.rolling(w, min_periods=1).max()

    return d
# =================================================================


# (Helpers de Feriados - Asumimos que están aquí)
def compute_holiday_factors(df, holidays_set):
    #print("WARN: Usando placeholder para compute_holiday_factors")
    return {}, {}, 0, 0, {} 

def apply_holiday_adjustment(df_hourly, *args, **kwargs):
    #print("WARN: Usando placeholder para apply_holiday_adjustment")
    return df_hourly 

def apply_post_holiday_adjustment(df_hourly, *args, **kwargs):
    #print("WARN: Usando placeholder para apply_post_holiday_adjustment")
    return df_hourly


def _prepare_hist_df(df: pd.DataFrame, holidays_set: set) -> pd.DataFrame:
    """Prepara DF histórico: TS, time parts, y recorta ventana."""
    dfh = ensure_ts(df.copy(), TIMEZONE)
    
    # Asegurar feriados y pago (main.py ya los puso)
    if "feriados" not in dfh.columns:
         dfh["feriados"] = mark_holidays_index(dfh.index, holidays_set).values
    if "es_dia_de_pago" not in dfh.columns:
         dfh = add_es_dia_de_pago(dfh)

    dfh = add_time_parts(dfh)
    
    # Recortar histórico para eficiencia
    last_ts = dfh.index.max()
    # Requerimos al menos 168h (7 días) + 7 días extra = 14 días (336h) para los lags
    min_lag_hours = 168 + (7*24) 
    min_ts_lags = last_ts - pd.DateOffset(hours=min_lag_hours)
    
    # Y no más de HIST_WINDOW_DAYS (ej 90) para el baseline de outliers
    min_ts_hist = last_ts - pd.DateOffset(days=HIST_WINDOW_DAYS)
    
    # Usamos el más antiguo de los dos (para tener suficientes lags)
    min_ts_final = min(min_ts_hist, min_ts_lags) 
    
    dfh = dfh.loc[dfh.index >= min_ts_final].copy()
    
    # Re-asegurar que no haya NaNs en columnas clave
    # Esta es la línea que fallaba (KeyError: 'recibidos_nacional')
    dfh[TARGET_CALLS] = pd.to_numeric(dfh[TARGET_CALLS], errors='coerce').ffill().bfill()
    dfh[TARGET_TMO] = pd.to_numeric(dfh[TARGET_TMO], errors='coerce').ffill().bfill()
    dfh["feriados"] = dfh["feriados"].ffill().bfill()
    dfh["es_dia_de_pago"] = dfh["es_dia_de_pago"].ffill().bfill()

    return dfh.asfreq('h') # Asegurar frecuencia horaria


# --- PREDICCIÓN LLAMADAS (SIN CAMBIOS) ---
def _predict_calls(df_hist: pd.DataFrame, future_index: pd.DatetimeIndex) -> pd.DataFrame:
    print("--- Predicción Llamadas (Planner) ---")
    model = tf.keras.models.load_model(PLANNER_MODEL, compile=False)
    scaler = joblib.load(PLANNER_SCALER)
    model_cols = _load_cols(PLANNER_COLS)
    
    df_future = pd.DataFrame(index=future_index)
    
    for ts in tqdm(future_index, desc="Prediciendo Llamadas"):
        hist_loop = pd.concat([df_hist, df_future])
        
        # 1. Features de tiempo (para ts) - ya están en df_hist
        time_parts = df_hist.loc[[ts]]
        
        # 2. Features AR (sobre hist_loop)
        ar_features = add_lags_mas(hist_loop, TARGET_CALLS)
        
        # 3. Ensamblar fila
        current_features = pd.concat([
            time_parts.reset_index(drop=True),
            ar_features.iloc[[-1]].reset_index(drop=True) # Tomar última fila (features para ts)
        ], axis=1)
        current_features.index = [ts]
        
        # 4. Predecir
        X_row_pre = dummies_and_reindex(current_features, model_cols)
        X_row = scaler.transform(X_row_pre)
        pred = model.predict(X_row, verbose=0)[0, 0]
        
        # 5. Guardar (con guardrail)
        df_future.loc[ts, TARGET_CALLS] = max(0, pred)
        df_future.loc[ts, time_parts.columns] = time_parts.iloc[0] # Guardar time parts

    print(f"Predicción Llamadas completada. Mediana: {df_future[TARGET_CALLS].median():.1f}")
    return df_future[[TARGET_CALLS]]


# --- PREDICCIÓN TMO (REESCRITA PARA v7-RESIDUAL) ---
def _predict_tmo(df_hist_full: pd.DataFrame, future_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Predice TMO usando el modelo v7-residual.
    df_hist_full es el DF histórico completo (ya tiene time_parts, feriados, etc).
    """
    print("--- Predicción TMO v7-residual ---")
    
    # 1. Cargar artefactos del nuevo modelo
    try:
        model = tf.keras.models.load_model(TMO_MODEL, compile=False)
        scaler = joblib.load(TMO_SCALER)
        model_cols = _load_cols(TMO_COLS)
        df_baseline = pd.read_csv(TMO_BASELINE) # (dow, hour, tmo_baseline)
        with open(TMO_META, "r") as f:
            meta = json.load(f)
        RESID_MEAN = meta['resid_mean']
        RESID_STD = meta['resid_std']
    except Exception as e:
        print(f"ERROR: Faltan artefactos del modelo TMO v7-residual. Ejecuta download_release.py")
        print(f"Detalle: {e}")
        raise

    # 2. Preparar histórico (el DF completo)
    # df_hist_full ya tiene 'feriados', 'es_dia_de_pago', 'dow', 'hour', etc.
    df_hist = df_hist_full.copy()
    
    # a) Merge baseline
    df_hist = df_hist.merge(df_baseline, on=["dow", "hour"], how="left")
    
    # b) Rellenar baseline para horas/dows no vistos
    if df_hist["tmo_baseline"].isna().any():
        print("Rellenando 'tmo_baseline' faltante en histórico (ffill/bfill)...")
        df_hist["tmo_baseline"] = df_hist["tmo_baseline"].ffill().bfill()
    if df_hist["tmo_baseline"].isna().any(): # Si sigue habiendo (ej. al inicio)
        df_hist["tmo_baseline"] = df_hist["tmo_baseline"].fillna(df_hist["tmo_baseline"].mean())
        
    # c) Calcular residuo histórico
    df_hist[TARGET_TMO] = pd.to_numeric(df_hist[TARGET_TMO], errors='coerce')
    df_hist["tmo_resid"] = df_hist[TARGET_TMO] - df_hist["tmo_baseline"]
    
    # d) Llenar NaNs en el residuo histórico (clave para lags)
    df_hist["tmo_resid"] = df_hist["tmo_resid"].ffill().bfill() 
    if df_hist["tmo_resid"].isna().any():
         print("WARN: 'tmo_resid' histórico sigue con NaNs. Rellenando con 0.")
         df_hist["tmo_resid"] = df_hist["tmo_resid"].fillna(0)

    # 3. Dataframe para predicciones futuras
    df_future = pd.DataFrame(index=future_index)
    
    # 4. Loop autorregresivo
    for ts in tqdm(future_index, desc="Prediciendo TMO"):
        # Combinar histórico + futuro predicho hasta ahora
        hist_loop = pd.concat([df_hist, df_future])
        
        # a) Features de tiempo para 'ts' (ya están en df_hist_full)
        time_parts = df_hist_full.loc[[ts]]
        
        # b) Baseline para 'ts'
        current_dow = time_parts.iloc[0]['dow']
        current_hour = time_parts.iloc[0]['hour']
        
        baseline_values = df_baseline.loc[
            (df_baseline['dow'] == current_dow) & (df_baseline['hour'] == current_hour),
            'tmo_baseline'
        ].values
        
        if len(baseline_values) == 0:
            # Fallback: usar el baseline del histórico de la última hora (no debería pasar si tmo_baseline.csv es completo)
            baseline_now = hist_loop.iloc[-2]["tmo_baseline"] 
        else:
            baseline_now = baseline_values[0]

        # c) Features autoregresivas (calculadas sobre hist_loop)
        df_with_features = _make_tmo_residual_features(hist_loop)
        
        # d) Ensamblar fila de features para 'ts'
        ar_features = df_with_features.iloc[[-1]]
        
        current_features = pd.concat([
            time_parts.reset_index(drop=True),
            ar_features.reset_index(drop=True)
        ], axis=1)
        current_features.index = [ts]

        # e) Dummies, Reindex, Scale
        # Rellenar NaNs de features (ej. std_resid en las primeras horas)
        current_features_filled = current_features.ffill() 
        
        X_row_pre = dummies_and_reindex(current_features_filled, model_cols)
        
        # Rellenar NaNs *después* de dummify/reindex (si alguna columna faltaba o ffill falló)
        X_row_pre = X_row_pre.fillna(0) 

        X_row = scaler.transform(X_row_pre)
        
        # f) Predecir (z-score del residuo)
        pred_z = model.predict(X_row, verbose=0)[0, 0]
        
        # g) Reconstruir TMO en segundos
        pred_resid = (pred_z * RESID_STD) + RESID_MEAN
        pred_sec = baseline_now + pred_resid
        pred_sec = max(30, min(pred_sec, 900)) # Guardrail (ej: TMO 30s-900s)
        
        # h) Guardar predicción en df_future para el siguiente loop
        df_future.loc[ts, TARGET_TMO] = pred_sec
        df_future.loc[ts, "tmo_baseline"] = baseline_now
        df_future.loc[ts, "tmo_resid"] = pred_resid
        
        # Copiar time parts para que _make_tmo_residual_features funcione
        df_future.loc[ts, time_parts.columns] = time_parts.iloc[0]

    print(f"Predicción TMO v7-residual completada. Mediana: {df_future[TARGET_TMO].median():.1f}s")
    return df_future[[TARGET_TMO]]


# --- FUNCIÓN PRINCIPAL (MODIFICADA) ---
def forecast_120d(
    df: pd.DataFrame, 
    holidays_set: set,
    horizon_days: int = 120  # <-- ACEPTA HORIZONTE
) -> pd.DataFrame:
    
    # 1. Preparar histórico (DF unificado)
    df_hist_full = _prepare_hist_df(df, holidays_set)
    last_ts = df_hist_full.index.max()

    # 2. Definir rango futuro (usa horizon_days)
    future_index = pd.date_range(
        start=last_ts + pd.DateOffset(hours=1),
        periods=horizon_days * 24,  # <-- USA EL ARGUMENTO
        freq='h',
        tz=TIMEZONE
    )

    # 3. Añadir feriados y time parts al rango futuro (necesario para ambos modelos)
    df_future_base = pd.DataFrame(index=future_index)
    df_future_base["feriados"] = mark_holidays_index(future_index, holidays_set).values
    df_future_base = add_time_parts(df_future_base)
    df_future_base = add_es_dia_de_pago(df_future_base)

    # Concatenar base futura al histórico (para que _predict* tengan acceso)
    df_hist_full = pd.concat([df_hist_full, df_future_base])

    # 4. Predecir Llamadas
    df_calls_future = _predict_calls(df_hist_full, future_index)

    # 5. Predecir TMO
    df_tmo_future = _predict_tmo(df_hist_full, future_index)

    # 6. Combinar y post-procesar
    df_hourly = df_future_base.join(df_calls_future).join(df_tmo_future)
    df_hourly.rename(columns={TARGET_CALLS: "calls", TARGET_TMO: "tmo_s"}, inplace=True)
    
    # Rellenar NaNs si alguna predicción falló
    df_hourly["calls"] = df_hourly["calls"].fillna(0)
    df_hourly["tmo_s"] = df_hourly["tmo_s"].ffill().bfill()


    # ===== AJUSTE FERIADOS (opcional, pero recomendado) =====
    if holidays_set and len(holidays_set) > 0:
        print("Calculando y aplicando factores de ajuste por feriados...")
        (f_calls_by_hour, f_tmo_by_hour,
         g_calls, g_tmo, post_calls_by_hour) = compute_holiday_factors(df_hist_full, holidays_set) 

        df_hourly = apply_holiday_adjustment(
            df_hourly, holidays_set,
            f_calls_by_hour, f_tmo_by_hour,
            col_calls_future="calls", col_tmo_future="tmo_s"
        )
        df_hourly = apply_post_holiday_adjustment(
            df_hourly, holidays_set, post_calls_by_hour,
            col_calls_future="calls"
        )

    # ===== (OPCIONAL) CAP de OUTLIERS =====
    if ENABLE_OUTLIER_CAP:
        print("Aplicando cap de outliers a la predicción de llamadas...")
        base_mad = _baseline_median_mad(df_hist_full, col=TARGET_CALLS) 
        df_hourly = apply_outlier_cap(
            df_hourly, base_mad, holidays_set,
            col_calls_future="calls",
            k_weekday=K_WEEKDAY, k_weekend=K_WEEKEND
        )

    # ===== Erlang por hora =====
    print("Calculando agentes requeridos (Erlang)...")
    df_hourly["agents_prod"] = 0
    
    # Vectorizar Erlang si es posible (más rápido que el loop)
    try:
        df_hourly["agents_prod"] = np.vectorize(required_agents)(
            df_hourly["calls"].values, 
            df_hourly["tmo_s"].values
        )[0] # required_agents devuelve (agentes, occ)
        df_hourly["agents_prod"] = df_hourly["agents_prod"].astype(int)
    except Exception:
        print("Vectorización de Erlang falló, usando loop...")
        for ts in tqdm(df_hourly.index, desc="Calculando Erlang (loop)"):
            a, _ = required_agents(
                float(df_hourly.at[ts, "calls"]), 
                float(df_hourly.at[ts, "tmo_s"])
            )
            df_hourly.at[ts, "agents_prod"] = int(a)
    
    df_hourly["agents_sched"] = schedule_agents(df_hourly["agents_prod"])

    # ===== Guardar resultados =====
    print("Guardando resultados JSON...")
    os.makedirs(PUBLIC_DIR, exist_ok=True)
    write_hourly_json(
        os.path.join(PUBLIC_DIR, "forecast_hourly.json"),
        df_hourly, "calls", "tmo_s", "agents_sched"
    )
    write_daily_json(
        os.path.join(PUBLIC_DIR, "forecast_daily.json"),
        df_hourly, "calls", "tmo_s"
    )
    
    print("--- Inferencia completada exitosamente ---")
    return df_hourly
