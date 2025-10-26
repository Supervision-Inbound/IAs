# src/inferencia/inferencia_core.py
import json, sys
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from .features import ensure_ts, add_time_parts, add_lags_mas, dummies_and_reindex, add_es_dia_de_pago
from .erlang import required_agents, schedule_agents
from .utils_io import write_daily_json, write_hourly_json

TIMEZONE = "America/Santiago"
PUBLIC_DIR = "public"

# --- Modelos Planner (Llamadas) ---
PLANNER_MODEL = "models/modelo_planner.keras"
PLANNER_SCALER = "models/scaler_planner.pkl"
PLANNER_COLS = "models/training_columns_planner.json"

# --- Modelos TMO (NUEVO) ---
TMO_MODEL = "models/modelo_tmo.keras"
TMO_SCALER = "models/scaler_tmo.pkl"
TMO_COLS = "models/training_columns_tmo.json"
TMO_BASELINE = "models/tmo_baseline_dow_hour.csv" # <--- NUEVO
TMO_META = "models/tmo_residual_meta.json"       # <--- NUEVO

TARGET_CALLS = "contestados"
TARGET_TMO = "tmo (segundos)" # <--- MODIFICADO: Coincide con main.py

# Ventana reciente para lags/MA (no afecta last_ts)
HIST_WINDOW_DAYS = 90

# ======= NUEVO: Guardrail Outliers (config) =======
ENABLE_OUTLIER_CAP = True   # <- ponlo en False si quieres desactivarlo
K_WEEKDAY = 6.0             # techos +K*MAD en lun-vie
K_WEEKEND = 7.0             # techos +K*MAD en sáb-dom


def _load_cols(path: str):
    with open(path, "r") as f:
        return json.load(f)

# (Helpers de Outlier Cap: _baseline_median_mad, apply_outlier_cap - Sin cambios)
# ... (mantenemos las funciones _baseline_median_mad y apply_outlier_cap)
def _baseline_median_mad(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df_ = df.copy()
    df_['dow'] = df_.index.dayofweek
    df_['hour'] = df_.index.hour
    
    # Agrupar por (dow, hour) y calcular mediana (baseline) y MAD
    base_mad = df_.groupby(['dow', 'hour'])[col].agg(
        median='median',
        count='count',
        mad=lambda x: np.median(np.abs(x - np.median(x)))
    ).reset_index()
    
    # El MAD puede ser 0 si todos los valores son iguales. 
    # Usamos 1.4826 * MAD para estimar STD. Si MAD es 0, usamos 5% de la mediana como fallback.
    base_mad['mad_std'] = base_mad['mad'] * 1.4826
    base_mad['mad_std'] = base_mad['mad_std'].replace(0, base_mad['median'] * 0.05)
    base_mad['mad_std'] = base_mad['mad_std'].replace(0, 1.0) # Fallback final si mediana es 0
    return base_mad

def apply_outlier_cap(df_future: pd.DataFrame, base_mad: pd.DataFrame, 
                      holidays_set: set, col_calls_future: str,
                      k_weekday: float, k_weekend: float) -> pd.DataFrame:
    df_ = df_future.copy()
    df_['dow'] = df_.index.dayofweek
    df_['hour'] = df_.index.hour
    df_['date'] = df_.index.date

    df_ = df_.merge(base_mad[['dow', 'hour', 'median', 'mad_std']], on=['dow', 'hour'], how='left')

    # Definir K (límite)
    df_['k'] = np.where(df_['dow'].isin([5, 6]), k_weekend, k_weekday) # Sáb/Dom más alto
    
    # Calcular techo
    df_['techo'] = df_['median'] + (df_['k'] * df_['mad_std'])
    
    # No aplicar techo a feriados
    is_holiday = df_['date'].isin(holidays_set)
    df_.loc[is_holiday, 'techo'] = np.inf
    
    # Aplicar cap
    df_['original'] = df_[col_calls_future]
    df_[col_calls_future] = np.clip(df_['original'], a_min=0, a_max=df_['techo'])
    
    capped_count = (df_['original'] > df_['techo']).sum()
    if capped_count > 0:
        print(f"WARN: (Guardrail) Se aplicó techo a {capped_count} horas futuras.")

    return df_.drop(columns=['dow', 'hour', 'date', 'median', 'mad_std', 'k', 'techo', 'original'])


# (Helpers de Feriados: compute_holiday_factors, etc. - Sin cambios)
# ... (mantenemos las funciones compute_holiday_factors, apply_holiday_adjustment, apply_post_holiday_adjustment)
def compute_holiday_factors(df: pd.DataFrame, holidays_set: set):
    df_ = df.copy()
    df_['date'] = df_.index.date
    df_['dow'] = df_.index.dayofweek
    df_['hour'] = df_.index.hour

    df_['es_feriado'] = df_['date'].isin(holidays_set)
    
    # Baseline (mediana) por (dow, hour)
    baseline = df_.groupby(['dow', 'hour'])[[TARGET_CALLS, TARGET_TMO]].median()
    baseline = baseline.rename(columns={TARGET_CALLS: 'calls_base', TARGET_TMO: 'tmo_base'})
    
    df_ = df_.merge(baseline, on=['dow', 'hour'], how='left')
    
    # Calcular factores (Real / Baseline)
    df_['f_calls'] = df_[TARGET_CALLS] / (df_['calls_base'] + 1)
    df_['f_tmo'] = df_[TARGET_TMO] / (df_['tmo_base'] + 1)
    
    # Mediana del factor por (feriado, hora)
    factors = df_.groupby(['es_feriado', 'hour'])[['f_calls', 'f_tmo']].median()
    
    # Factores de Feriado (mediana del factor en feriados)
    f_calls_h = factors.loc[True]['f_calls'].rename("f_calls_holiday")
    f_tmo_h = factors.loc[True]['f_tmo'].rename("f_tmo_holiday")
    
    # Factores de No-Feriado (mediana del factor en NO feriados - cercano a 1.0)
    f_calls_g = factors.loc[False]['f_calls'].rename("f_calls_global")
    f_tmo_g = factors.loc[False]['f_tmo'].rename("f_tmo_global")

    # Post-feriado (efecto rebote)
    dias_post = set(h + pd.Timedelta(days=1) for h in holidays_set)
    df_['es_post_feriado'] = df_['date'].isin(dias_post) & (df_['dow'] != 6) # No domingos
    
    factors_post = df_.groupby(['es_post_feriado', 'hour'])[['f_calls']].median()
    f_calls_post_h = factors_post.loc[True]['f_calls'].rename("f_calls_post_holiday")

    return f_calls_h, f_tmo_h, f_calls_g, f_tmo_g, f_calls_post_h

def apply_holiday_adjustment(df_future: pd.DataFrame, holidays_set: set,
                             f_calls_by_hour: pd.Series, f_tmo_by_hour: pd.Series,
                             col_calls_future: str, col_tmo_future: str) -> pd.DataFrame:
    df_ = df_future.copy()
    df_['date'] = df_.index.date
    df_['hour'] = df_.index.hour
    
    is_holiday = df_['date'].isin(holidays_set)
    if is_holiday.sum() == 0:
        return df_.drop(columns=['date', 'hour']) # No hay feriados en el futuro
        
    df_ = df_.merge(f_calls_by_hour, left_on='hour', right_index=True, how='left')
    df_ = df_.merge(f_tmo_by_hour, left_on='hour', right_index=True, how='left')
    
    # Aplicar factor SOLO a feriados
    df_.loc[is_holiday, col_calls_future] *= df_.loc[is_holiday, 'f_calls_holiday']
    df_.loc[is_holiday, col_tmo_future]  *= df_.loc[is_holiday, 'f_tmo_holiday']

    return df_.drop(columns=['date', 'hour', 'f_calls_holiday', 'f_tmo_holiday'])

def apply_post_holiday_adjustment(df_future: pd.DataFrame, holidays_set: set, 
                                  post_calls_by_hour: pd.Series,
                                  col_calls_future: str) -> pd.DataFrame:
    df_ = df_future.copy()
    df_['date'] = df_.index.date
    df_['dow'] = df_.index.dayofweek
    df_['hour'] = df_.index.hour
    
    dias_post = set(h + pd.Timedelta(days=1) for h in holidays_set)
    is_post_holiday = df_['date'].isin(dias_post) & (df_['dow'] != 6) # No domingos
    
    if is_post_holiday.sum() == 0:
        return df_.drop(columns=['date', 'dow', 'hour'])

    df_ = df_.merge(post_calls_by_hour, left_on='hour', right_index=True, how='left')
    
    # Aplicar factor SOLO a post-feriados
    df_.loc[is_post_holiday, col_calls_future] *= df_.loc[is_post_holiday, 'f_calls_post_holiday']
    
    return df_.drop(columns=['date', 'dow', 'hour', 'f_calls_post_holiday'])


# --- Funciones de Predicción ---

def _predict_planner(df_hist: pd.DataFrame, df_future: pd.DataFrame) -> pd.Series:
    """Predice LLAMADAS (Planner)"""
    print("Ejecutando _predict_planner (Llamadas)...")
    model = tf.keras.models.load_model(PLANNER_MODEL, compile=False)
    scaler = joblib.load(PLANNER_SCALER)
    cols = _load_cols(PLANNER_COLS)
    
    target_col = TARGET_CALLS # Usamos 'recibidos_nacional'
    
    # 1. Tomar la serie histórica de la feature
    hist_vals = df_hist[target_col]
    future_idx = df_future.index
    
    # 2. Concatenar con el índice futuro
    full_series = pd.concat([hist_vals, pd.Series(index=future_idx, dtype=float)])
    full_series = full_series.asfreq(hist_vals.index.freq or 'H') # Asegurar freq
    
    # 3. Generar features autoregresivas (lags, MAs, etc.)
    df_features = pd.DataFrame(index=full_series.index)
    df_features[target_col] = full_series
    df_features = add_lags_mas(df_features, target_col) # <-- de features.py
    
    # 4. Añadir features de tiempo
    df_features = add_time_parts(df_features)
    
    # 5. Añadir features exógenas (feriados, pago) del futuro
    df_features = df_features.join(df_future[['feriados', 'es_dia_de_pago']])
    # Rellenar nulos en exógenas (para los días históricos)
    df_features['feriados'] = df_features['feriados'].fillna(0)
    df_features['es_dia_de_pago'] = df_features['es_dia_de_pago'].fillna(0)
    
    # 6. Filtrar solo el período de futuro
    X_future = df_features.loc[future_idx].copy()
    
    # 7. Dummies, Reindex y Scaler
    X_future, _ = dummies_and_reindex(X_future, cols)
    X_future_s = scaler.transform(X_future)
    
    # 8. Predecir
    y_pred = model.predict(X_future_s).flatten()
    y_pred = np.clip(y_pred, 0, None) # No llamadas negativas
    
    return pd.Series(y_pred, index=future_idx, name="calls")


def _predict_tmo(df_hist: pd.DataFrame, df_future: pd.DataFrame) -> pd.Series:
    """
    <--- MODIFICADO: Predice TMO (v8, Residual con features de LLAMADAS) ---
    """
    print("Ejecutando _predict_tmo (v8 - features de llamadas)...")
    
    # 1. Cargar TODOS los artefactos del nuevo modelo TMO
    try:
        model = tf.keras.models.load_model(TMO_MODEL, compile=False)
        scaler = joblib.load(TMO_SCALER)
        cols = _load_cols(TMO_COLS)
        baseline_df = pd.read_csv(TMO_BASELINE) # (dow, hour, tmo_baseline)
        with open(TMO_META, 'r') as f:
            meta = json.load(f)
        resid_mean = meta['resid_mean']
        resid_std = meta['resid_std']
    except FileNotFoundError as e:
        print(f"ERROR: Falta un artefacto del modelo TMO: {e.filename}", file=sys.stderr)
        print("Asegúrate de haber descargado 'tmo_baseline_dow_hour.csv' y 'tmo_residual_meta.json'", file=sys.stderr)
        raise e
        
    # 2. Definir la columna de FEATURES.
    # El modelo TMO v8 usa features de LLAMADAS.
    feature_col = TARGET_CALLS # <-- ¡Esta es la clave!
    
    # 3. Tomar la serie histórica de la feature (LLAMADAS)
    hist_vals = df_hist[feature_col]
    future_idx = df_future.index
    
    # 4. Concatenar con el índice futuro
    full_series = pd.concat([hist_vals, pd.Series(index=future_idx, dtype=float)])
    full_series = full_series.asfreq(hist_vals.index.freq or 'H')
    
    # 5. Generar features autoregresivas (lags, MAs, etc. DE LLAMADAS)
    df_features = pd.DataFrame(index=full_series.index)
    df_features[feature_col] = full_series
    # Usamos la *misma* función 'add_lags_mas' que usa el planner
    df_features = add_lags_mas(df_features, feature_col)
    
    # 6. Añadir features de tiempo
    df_features = add_time_parts(df_features)
    
    # 7. Añadir features exógenas (feriados, pago) del futuro
    df_features = df_features.join(df_future[['feriados', 'es_dia_de_pago']])
    df_features['feriados'] = df_features['feriados'].fillna(0)
    df_features['es_dia_de_pago'] = df_features['es_dia_de_pago'].fillna(0)
    
    # 8. Filtrar solo el período de futuro
    X_future = df_features.loc[future_idx].copy()
    
    # 9. Dummies, Reindex y Scaler
    X_future, _ = dummies_and_reindex(X_future, cols)
    X_future_s = scaler.transform(X_future)
    
    # 10. Predecir (Residuo Z-Score)
    y_pred_z = model.predict(X_future_s).flatten()
    
    # 11. Reconstruir TMO en segundos
    # 11.1. Invertir Z-Score -> Residuo
    y_pred_resid = (y_pred_z * resid_std) + resid_mean
    
    # 11.2. Traer el Baseline (mediana por dow, hour)
    df_pred = pd.DataFrame(y_pred_resid, index=X_future.index, columns=['tmo_resid_pred'])
    df_pred['dow'] = df_pred.index.dayofweek
    df_pred['hour'] = df_pred.index.hour
    
    # Juntar con el baseline de entrenamiento
    df_pred = df_pred.merge(baseline_df, on=['dow', 'hour'], how='left')
    
    # Fallback por si falta algún (dow, hour) en el baseline (ej. 3AM)
    median_baseline_global = baseline_df['tmo_baseline'].median()
    df_pred['tmo_baseline'] = df_pred['tmo_baseline'].fillna(median_baseline_global)
    
    # 11.3. Sumar: Predicción Final = Baseline + Residuo
    y_pred_final_sec = df_pred['tmo_baseline'] + df_pred['tmo_resid_pred']
    
    # 12. Aplicar un piso (ej. 30 segundos)
    y_pred_final_sec = y_pred_final_sec.clip(lower=30.0)
    
    return pd.Series(y_pred_final_sec.values, index=future_idx, name="tmo_s")


# --- Pipeline Principal ---

def forecast_120d(
    df_raw_reset: pd.DataFrame,
    # df_tmo_reset: pd.DataFrame | None, # <--- MODIFICADO: Eliminado
    holidays_set: set
) -> pd.DataFrame:
    
    # --- 1. Preparar Histórico (df) ---
    # df es el histórico de LLAMADAS (y TMO, ahora)
    df = ensure_ts(df_raw_reset, tz=TIMEZONE)
    df[TARGET_CALLS] = pd.to_numeric(df[TARGET_CALLS], errors='coerce').fillna(0)
    df[TARGET_TMO] = pd.to_numeric(df[TARGET_TMO], errors='coerce').ffill()
    
    # --- (Eliminada la preparación de df_tmo) ---
    
    # Acortar historial para lags (ej. últimos 90 días)
    last_ts = df.index.max()
    start_hist_window = last_ts - pd.Timedelta(days=HIST_WINDOW_DAYS)
    
    # df_hist se usará para generar features para AMBOS modelos
    df_hist = df.loc[df.index >= start_hist_window].copy()
    
    print(f"Histórico unificado desde {df_hist.index.min()} hasta {df_hist.index.max()}")
    
    # --- 2. Preparar Futuro (df_future) ---
    start_future = last_ts + pd.Timedelta(hours=1)
    end_future = last_ts + pd.Timedelta(days=120)
    
    future_index = pd.date_range(start=start_future, end=end_future, freq='H', tz=TIMEZONE)
    df_future = pd.DataFrame(index=future_index)
    
    # Añadir features exógenas al futuro
    df_future['feriados'] = mark_holidays_index(df_future.index, holidays_set).values
    df_future = add_time_parts(df_future) # add_time_parts también añade 'day'
    df_future['es_dia_de_pago'] = add_es_dia_de_pago(df_future).values
    
    # --- 3. Predicciones ---
    
    # 3.1. Predecir Planner (Llamadas)
    # _predict_planner usa df_hist[TARGET_CALLS] para sus features
    df_future["calls"] = _predict_planner(df_hist, df_future)

    # 3.2. Predecir TMO (Nuevo modelo)
    # _predict_tmo usa df_hist[TARGET_CALLS] para sus features
    df_future["tmo_s"] = _predict_tmo(df_hist, df_future)
    
    # --- 4. Post-Procesamiento (Ajustes Feriados, Outliers) ---
    df_hourly = df_future.copy()
    
    if holidays_set and len(holidays_set) > 0:
        print("Calculando y aplicando factores de ajuste por feriados...")
        (f_calls_by_hour, f_tmo_by_hour,
         g_calls, g_tmo, post_calls_by_hour) = compute_holiday_factors(df, holidays_set)

        df_hourly = apply_holiday_adjustment(
            df_hourly, holidays_set,
            f_calls_by_hour, f_tmo_by_hour,
            col_calls_future="calls", col_tmo_future="tmo_s"
        )
        df_hourly = apply_post_holiday_adjustment(
            df_hourly, holidays_set, post_calls_by_hour,
            col_calls_future="calls"
        )

    # (OPCIONAL) CAP de OUTLIERS (Solo sobre llamadas)
    if ENABLE_OUTLIER_CAP:
        print("Aplicando Guardrail (Cap) de outliers sobre llamadas...")
        base_mad = _baseline_median_mad(df, col=TARGET_CALLS)
        df_hourly = apply_outlier_cap(
            df_hourly, base_mad, holidays_set,
            col_calls_future="calls",
            k_weekday=K_WEEKDAY, k_weekend=K_WEEKEND
        )

    # --- 5. Erlang por hora ---
    print("Calculando agentes requeridos (Erlang)...")
    df_hourly["agents_prod"] = 0
    for ts in df_hourly.index:
        a, _ = required_agents(float(df_hourly.at[ts, "calls"]), float(df_hourly.at[ts, "tmo_s"]))
        df_hourly.at[ts, "agents_prod"] = int(a)
        
    # Agentes conectados (con Shrinkage)
    df_hourly["agents_conn"] = schedule_agents(df_hourly["agents_prod"])
    
    # --- 6. Guardar Resultados ---
    print("Guardando resultados en JSON...")
    os.makedirs(PUBLIC_DIR, exist_ok=True)
    write_hourly_json(
        f"{PUBLIC_DIR}/forecast_hourly.json", 
        df_hourly, 
        calls_col="calls", 
        tmo_col="tmo_s", 
        agentes_col="agents_conn"
    )
    write_daily_json(
        f"{PUBLIC_DIR}/forecast_daily.json",
        df_hourly,
        calls_col="calls",
        tmo_col="tmo_s"
    )

    print("--- Predicción de 120 días completada ---")
    
    return df_hourly
