# src/inferencia/inferencia_core.py
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys # Para imprimir versión

# Imprimir versiones al inicio
print(f"--- DEBUG: Python Version: {sys.version}")
print(f"--- DEBUG: TensorFlow Version: {tf.__version__}")
print(f"--- DEBUG: Pandas Version: {pd.__version__}")
print(f"--- DEBUG: Numpy Version: {np.__version__}")
print(f"--- DEBUG: Joblib Version: {joblib.__version__}")
# (Puedes añadir sklearn si es relevante, aunque el error es de TF)
# import sklearn; print(f"--- DEBUG: Sklearn Version: {sklearn.__version__}")


from .features import ensure_ts, add_time_parts, add_lags_mas, dummies_and_reindex
from .erlang import required_agents, schedule_agents
from .utils_io import write_daily_json, write_hourly_json
try:
    from src.main import add_es_dia_de_pago
except ImportError:
    print("WARN: No se pudo importar 'add_es_dia_de_pago' desde src.main. Definiendo fallback.")
    def add_es_dia_de_pago(df_idx: pd.DataFrame | pd.Index) -> pd.Series:
        idx = df_idx if isinstance(df_idx, pd.Index) else df_idx.index
        if not isinstance(idx, pd.DatetimeIndex):
            try: idx = pd.to_datetime(idx)
            except Exception: return pd.Series(0, index=idx, dtype=int, name="es_dia_de_pago")
        dias = [1,2,15,16,29,30,31]
        return pd.Series(idx.day.isin(dias).astype(int), index=idx, name="es_dia_de_pago")


TIMEZONE = "America/Santiago"
PUBLIC_DIR = "public"

PLANNER_MODEL = "models/modelo_planner.keras"
PLANNER_SCALER = "models/scaler_planner.pkl"
PLANNER_COLS = "models/training_columns_planner.json"

TMO_MODEL = "models/modelo_tmo.keras"
TMO_SCALER = "models/scaler_tmo.pkl"
TMO_COLS = "models/training_columns_tmo.json"

TARGET_CALLS = "recibidos_nacional"
TARGET_TMO = "tmo_general"

HIST_WINDOW_DAYS = 90
ENABLE_OUTLIER_CAP = True
K_WEEKDAY = 6.0
K_WEEKEND = 7.0


def _load_cols(path: str):
    """Carga columnas desde un archivo JSON."""
    print(f"--- DEBUG: Intentando cargar columnas desde: {path}")
    if not os.path.exists(path):
         print(f"ERROR: Archivo de columnas NO ENCONTRADO en {path}")
         raise FileNotFoundError(f"Archivo de columnas no encontrado: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            cols = json.load(f)
            print(f"--- DEBUG: Columnas cargadas ({len(cols)}): {cols[:5]}...") # Muestra las primeras 5
            return cols
    except json.JSONDecodeError:
        print(f"ERROR: El archivo de columnas {path} no es un JSON válido.")
        raise
    except Exception as e:
        print(f"ERROR: No se pudo cargar el archivo de columnas {path}: {e}")
        raise

# === [v31] Definición de _prepare_full_data ===
def _prepare_full_data(df_hist_joined):
    """Prepara y limpia el DataFrame completo para la predicción TMO v8/v29."""
    print("--- DEBUG: Entrando a _prepare_full_data...")
    df_input_copy = df_hist_joined.copy()
    try:
        df_full = ensure_ts(df_input_copy)
        print(f"--- DEBUG: ensure_ts completado para df_full. Shape: {df_full.shape}")
    except ValueError as e:
        print(f"ERROR: Falló ensure_ts al preparar datos completos para TMO: {e}")
        return None
    if TARGET_CALLS not in df_full.columns:
        print(f"ERROR: Falta columna {TARGET_CALLS} en datos para TMO. TMO se saltará.")
        return None
    if TARGET_TMO not in df_full.columns:
        print(f"WARN: Falta columna {TARGET_TMO} en datos para TMO. Se usará 0.")
        df_full[TARGET_TMO] = 0.0
    df_full = df_full.dropna(subset=[TARGET_CALLS])
    if df_full.empty:
        print(f"WARN: DataFrame vacío después de dropna({TARGET_CALLS}) para TMO.")
        return None
    df_full[TARGET_TMO] = pd.to_numeric(df_full[TARGET_TMO], errors='coerce').ffill().fillna(0.0)
    for aux in ["feriados", "es_dia_de_pago"]:
        if aux in df_full.columns:
            df_full[aux] = pd.to_numeric(df_full[aux], errors='coerce')
            df_full[aux] = df_full[aux].ffill()
        else:
            print(f"WARN: Columna '{aux}' faltante para TMO, se añadirá con ceros.")
            if isinstance(df_full.index, pd.DatetimeIndex):
                 if aux == "feriados": df_full[aux] = 0
                 elif aux == "es_dia_de_pago": df_full[aux] = add_es_dia_de_pago(df_full.index).values
            else: print(f"WARN: Índice no es DatetimeIndex, no se pudo añadir '{aux}' correctamente."); df_full[aux] = 0
        df_full[aux] = pd.to_numeric(df_full[aux], errors='coerce').fillna(0).astype(int)
    if df_full.empty: print("ERROR: df_full quedó vacío después del preprocesamiento para TMO."); return None
    print(f"--- DEBUG: _prepare_full_data completado. Shape final df_full: {df_full.shape}")
    return df_full
# === FIN DEFINICIÓN _prepare_full_data ===


# ========= Helpers (Corregidos v30/v31) =========
def _safe_ratio(num, den, fallback=1.0):
    num = float(num) if num is not None and not np.isnan(num) else np.nan
    den = float(den) if den is not None and not np.isnan(den) and den != 0 else np.nan
    if np.isnan(num) or np.isnan(den) or den == 0: return fallback
    return num / den

def _series_is_holiday(idx, holidays_set):
    if not isinstance(idx, pd.DatetimeIndex):
        try: idx = pd.to_datetime(idx)
        except Exception: return pd.Series(False, index=idx, dtype=bool)
    tz = getattr(idx, "tz", None)
    try: idx_dates = idx.tz_convert(TIMEZONE).date if tz is not None else idx.date
    except Exception: idx_dates = idx.date # type: ignore
    if holidays_set is None: holidays_set = set()
    return pd.Series([d in holidays_set for d in idx_dates], index=idx, dtype=bool)

def compute_holiday_factors(df_hist, holidays_set,
                            col_calls=TARGET_CALLS, col_tmo=TARGET_TMO):
    # print(f"--- DEBUG: compute_holiday_factors - df_hist shape: {df_hist.shape}, index type: {type(df_hist.index)}") # Debug
    cols = [col_calls]
    if col_tmo in df_hist.columns and (not pd.to_numeric(df_hist[col_tmo], errors='coerce').isnull().all()):
        cols.append(col_tmo)
    df_hist_dt_idx = df_hist.copy()
    if not isinstance(df_hist_dt_idx.index, pd.DatetimeIndex):
         try:
             df_hist_dt_idx.index = pd.to_datetime(df_hist_dt_idx.index)
             if getattr(df_hist_dt_idx.index, "tz", None) is None: df_hist_dt_idx.index = df_hist_dt_idx.index.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT').tz_convert(TIMEZONE)
             else: df_hist_dt_idx.index = df_hist_dt_idx.index.tz_convert(TIMEZONE)
         except Exception as e:
             print(f"WARN: No se pudo convertir índice en compute_holiday_factors: {e}"); default_factors = {h: 1.0 for h in range(24)}; return (default_factors.copy(), default_factors.copy(), 1.0, 1.0, default_factors.copy())

    if df_hist_dt_idx.empty: # Check if empty after index conversion
        print(f"WARN: df_hist vacío en compute_holiday_factors después de procesar índice.")
        default_factors = {h: 1.0 for h in range(24)}; return (default_factors.copy(), default_factors.copy(), 1.0, 1.0, default_factors.copy())

    dfh = add_time_parts(df_hist_dt_idx[cols].copy())
    dfh["is_holiday"] = _series_is_holiday(dfh.index, holidays_set)
    med_hol_calls = dfh[dfh["is_holiday"]].groupby("hour")[col_calls].median()
    med_nor_calls = dfh[~dfh["is_holiday"]].groupby("hour")[col_calls].median()
    if col_tmo in cols:
        med_hol_tmo = dfh[dfh["is_holiday"]].groupby("hour")[col_tmo].median(); med_nor_tmo = dfh[~dfh["is_holiday"]].groupby("hour")[col_tmo].median()
        g_hol_tmo = dfh[dfh["is_holiday"]][col_tmo].median(); g_nor_tmo = dfh[~dfh["is_holiday"]][col_tmo].median()
        global_tmo_factor = _safe_ratio(g_hol_tmo, g_nor_tmo, fallback=1.00)
    else: med_hol_tmo = med_nor_tmo = None; global_tmo_factor = 1.00
    g_hol_calls = dfh[dfh["is_holiday"]][col_calls].median(); g_nor_calls = dfh[~dfh["is_holiday"]][col_calls].median()
    global_calls_factor = _safe_ratio(g_hol_calls, g_nor_calls, fallback=0.75)
    factors_calls_by_hour = {int(h): _safe_ratio(med_hol_calls.get(h, np.nan), med_nor_calls.get(h, np.nan), fallback=global_calls_factor) for h in range(24)}
    if med_hol_tmo is not None: factors_tmo_by_hour = {int(h): _safe_ratio(med_hol_tmo.get(h, np.nan), med_nor_tmo.get(h, np.nan), fallback=global_tmo_factor) for h in range(24)}
    else: factors_tmo_by_hour = {int(h): 1.0 for h in range(24)}
    factors_calls_by_hour = {h: float(np.clip(v, 0.10, 1.60)) for h, v in factors_calls_by_hour.items()}
    factors_tmo_by_hour   = {h: float(np.clip(v, 0.70, 1.50)) for h, v in factors_tmo_by_hour.items()}
    dfh = dfh.copy(); dfh["is_post_hol"] = (~dfh["is_holiday"]) & (dfh["is_holiday"].shift(1).fillna(False))
    post_calls_group = dfh.loc[dfh["is_post_hol"], col_calls]
    if not post_calls_group.empty: med_post_calls = post_calls_group.groupby(dfh["hour"]).median()
    else: med_post_calls = pd.Series(dtype=float)
    post_calls_by_hour = {int(h): _safe_ratio(med_post_calls.get(h, np.nan), med_nor_calls.get(h, np.nan), fallback=1.05) for h in range(24)}
    post_calls_by_hour = {h: float(np.clip(v, 0.90, 1.80)) for h, v in post_calls_by_hour.items()}
    return (factors_calls_by_hour, factors_tmo_by_hour, global_calls_factor, global_tmo_factor, post_calls_by_hour)

def apply_holiday_adjustment(df_future, holidays_set,
                             factors_calls_by_hour, factors_tmo_by_hour,
                             col_calls_future="calls", col_tmo_future="tmo_s"):
    if df_future.empty: return df_future # Evitar errores si está vacío
    if not isinstance(df_future.index, pd.DatetimeIndex):
         try:
             df_future.index = pd.to_datetime(df_future.index)
             if getattr(df_future.index, "tz", None) is None: df_future.index = df_future.index.tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT')
             else: df_future.index = df_future.index.tz_convert(TIMEZONE)
         except Exception as e: print(f"WARN: No se pudo convertir índice en apply_holiday_adjustment: {e}"); return df_future
    d = add_time_parts(df_future.copy())
    is_hol = _series_is_holiday(d.index, holidays_set)
    hours = d["hour"].astype(int).values
    call_f = np.array([factors_calls_by_hour.get(int(h), 1.0) for h in hours])
    tmo_f  = np.array([factors_tmo_by_hour.get(int(h), 1.0) for h in hours])
    out = df_future.copy()
    mask = is_hol.values
    if col_calls_future in out.columns: out.loc[mask, col_calls_future] = np.round(out.loc[mask, col_calls_future].astype(float) * call_f[mask]).astype(int)
    if col_tmo_future in out.columns: out.loc[mask, col_tmo_future]   = np.round(out.loc[mask, col_tmo_future].astype(float)   * tmo_f[mask]).astype(int)
    return out

def apply_post_holiday_adjustment(df_future, holidays_set, post_calls_by_hour,
                                  col_calls_future="calls"):
    if df_future.empty: return df_future # Evitar errores si está vacío
    if not isinstance(df_future.index, pd.DatetimeIndex):
         try:
             df_future.index = pd.to_datetime(df_future.index)
             if getattr(df_future.index, "tz", None) is None: df_future.index = df_future.index.tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT')
             else: df_future.index = df_future.index.tz_convert(TIMEZONE)
         except Exception as e: print(f"WARN: No se pudo convertir índice en apply_post_holiday_adjustment: {e}"); return df_future
    idx = df_future.index
    prev_idx = (idx - pd.Timedelta(days=1))
    try: prev_dates = pd.to_datetime(prev_idx.astype(str)).tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT').date; curr_dates = pd.to_datetime(idx.astype(str)).tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT').date
    except Exception: prev_dates = prev_idx.date; curr_dates = idx.date # type: ignore
    if holidays_set is None: holidays_set = set()
    is_prev_hol = pd.Series([d in holidays_set for d in prev_dates], index=idx, dtype=bool)
    is_today_hol = pd.Series([d in holidays_set for d in curr_dates], index=idx, dtype=bool)
    is_post = (~is_today_hol) & (is_prev_hol)
    d = add_time_parts(df_future.copy())
    hours = d["hour"].astype(int).values
    ph_f = np.array([post_calls_by_hour.get(int(h), 1.0) for h in hours])
    out = df_future.copy()
    mask = is_post.values
    if col_calls_future in out.columns: out.loc[mask, col_calls_future] = np.round(out.loc[mask, col_calls_future].astype(float) * ph_f[mask]).astype(int)
    return out

def _baseline_median_mad(df_hist, col=TARGET_CALLS):
    if df_hist.empty: print("WARN: df_hist vacío en _baseline_median_mad."); return pd.DataFrame({'dow':[], 'hour':[], 'med':[], 'mad':[]})
    if not isinstance(df_hist.index, pd.DatetimeIndex):
         try:
             df_hist.index = pd.to_datetime(df_hist.index)
             if getattr(df_hist.index, "tz", None) is None: df_hist.index = df_hist.index.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT').tz_convert(TIMEZONE)
             else: df_hist.index = df_hist.index.tz_convert(TIMEZONE)
         except Exception as e: print(f"WARN: No se pudo convertir índice en _baseline_median_mad: {e}"); return pd.DataFrame({'dow':[], 'hour':[], 'med':[], 'mad':[]})
    if col not in df_hist.columns: print(f"WARN: Columna '{col}' no encontrada en _baseline_median_mad."); return pd.DataFrame({'dow':[], 'hour':[], 'med':[], 'mad':[]})
    df_hist_col = pd.to_numeric(df_hist[col], errors='coerce')
    if df_hist_col.isnull().all(): print(f"WARN: Columna '{col}' es toda NaN en _baseline_median_mad."); return pd.DataFrame({'dow':[], 'hour':[], 'med':[], 'mad':[]})
    d = add_time_parts(df_hist_col.to_frame(name=col).copy())
    g = d.groupby(["dow", "hour"], observed=False)[col] # Use observed=False
    base = g.median().rename("med").to_frame()
    def mad_robust(x):
        x_clean = x.dropna();
        if len(x_clean) == 0: return np.nan;
        med = np.median(x_clean);
        if not np.isscalar(med): return np.nan;
        return np.median(np.abs(x_clean - med))
    mad = g.apply(mad_robust).rename("mad")
    base = base.join(mad)
    if base.empty: print("WARN: Baseline MAD vacío después de groupby."); return base.reset_index() # Devuelve vacío con columnas
    if base["mad"].isna().all(): base["mad"] = 0.0
    median_mad_global = base["mad"].median()
    if pd.isna(median_mad_global) or median_mad_global == 0: median_mad_global = 1.0
    base["mad"] = base["mad"].replace(0, median_mad_global).fillna(median_mad_global)
    median_med_global = base["med"].median()
    if pd.isna(median_med_global): median_med_global = 0.0
    base["med"] = base["med"].fillna(median_med_global)
    return base.reset_index()


def apply_outlier_cap(df_future, base_median_mad, holidays_set,
                      col_calls_future="calls",
                      k_weekday=K_WEEKDAY, k_weekend=K_WEEKEND):
    if df_future.empty: return df_future
    if not isinstance(df_future.index, pd.DatetimeIndex):
         try:
             df_future.index = pd.to_datetime(df_future.index)
             if getattr(df_future.index, "tz", None) is None: df_future.index = df_future.index.tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT')
             else: df_future.index = df_future.index.tz_convert(TIMEZONE)
         except Exception as e: print(f"WARN: No se pudo convertir índice en apply_outlier_cap: {e}"); return df_future
    d = add_time_parts(df_future.copy())
    prev_idx = (d.index - pd.Timedelta(days=1))
    try: curr_dates = pd.to_datetime(d.index.astype(str)).tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT').date; prev_dates = pd.to_datetime(prev_idx.astype(str)).tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT').date
    except Exception: curr_dates = d.index.date; prev_dates = prev_idx.date # type: ignore
    if holidays_set is None: holidays_set = set()
    is_hol = pd.Series([dt in holidays_set for dt in curr_dates], index=d.index, dtype=bool)
    is_prev_hol = pd.Series([dt in holidays_set for dt in prev_dates], index=d.index, dtype=bool)
    is_post_hol = (~is_hol) & (is_prev_hol)
    if base_median_mad.empty: print("WARN: Baseline MAD vacío, no se aplicará cap de outliers."); return df_future
    base = base_median_mad.copy()
    capped = d.merge(base, on=["dow","hour"], how="left")
    median_med_global = base["med"].median(); median_mad_global = base["mad"].median()
    if pd.isna(median_med_global): median_med_global = 0.0
    if pd.isna(median_mad_global) or median_mad_global == 0: median_mad_global = 1.0
    capped["mad"] = capped["mad"].fillna(median_mad_global)
    capped["med"] = capped["med"].fillna(median_med_global)
    is_weekend = capped["dow"].isin([5,6]).values
    K = np.where(is_weekend, k_weekend, k_weekday).astype(float)
    upper = capped["med"].values + K * capped["mad"].values
    calls_numeric = pd.to_numeric(capped[col_calls_future], errors='coerce').fillna(0.0)
    mask = (~is_hol.values) & (~is_post_hol.values) & (calls_numeric.values > upper)
    capped.loc[mask, col_calls_future] = np.round(upper[mask]).astype(int)
    out = df_future.copy()
    out[col_calls_future] = capped[col_calls_future].astype(int).values
    return out

# --- Función _is_holiday CORREGIDA (v32) ---
def _is_holiday(ts, holidays_set: set | None) -> int:
    """Checks if a given timestamp falls on a holiday."""
    # print(f"--- DEBUG _is_holiday: Input ts={ts}, type={type(ts)}") # Optional debug
    # 1. Basic check for holiday set
    if not holidays_set:
        return 0

    # 2. Ensure ts is a valid Timestamp, handle potential NaT early
    if not isinstance(ts, pd.Timestamp):
        try:
            ts = pd.to_datetime(ts)
        except Exception:
            # print(f"--- DEBUG _is_holiday: Failed to convert to datetime: {ts}")
            return 0 # Cannot proceed if not convertible
    if pd.isna(ts): # Check if conversion resulted in NaT
        # print(f"--- DEBUG _is_holiday: Timestamp is NaT: {ts}")
        return 0

    # 3. Handle Timezone and get the date
    the_date = None
    try:
        # If timezone naive, localize it
        if getattr(ts, "tz", None) is None:
            ts_aware = ts.tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT')
        # If already timezone aware, convert it
        else:
            ts_aware = ts.tz_convert(TIMEZONE)

        # Check if localization/conversion resulted in NaT *again*
        if pd.isna(ts_aware):
            # print(f"--- DEBUG _is_holiday: Timestamp became NaT after tz handling: {ts}")
            # Fallback 1: Try getting date directly from original ts
            try:
                the_date = ts.date()
            except Exception:
                 # print(f"--- DEBUG _is_holiday: Failed fallback 1 (date from original): {ts}")
                 pass # Date remains None
        else:
            # Successfully got tz-aware timestamp
            the_date = ts_aware.date()

    except Exception as e:
        # print(f"--- DEBUG _is_holiday: Error during timezone handling for {ts}: {e}")
        # Fallback 2: Try getting date directly from original ts if any tz error
        try:
            the_date = ts.date()
        except Exception:
            # print(f"--- DEBUG _is_holiday: Failed fallback 2 (date from original): {ts}")
            pass # Date remains None

    # 4. Check against the holiday set
    if the_date is not None:
        # Ensure holidays_set is actually a set
        if holidays_set is None: holidays_set = set()
        is_h = 1 if the_date in holidays_set else 0
        # print(f"--- DEBUG _is_holiday: Date={the_date}, Is Holiday={is_h}") # Optional debug
        return is_h
    else:
        # print(f"--- DEBUG _is_holiday: Could not determine date for {ts}")
        return 0 # Failed to get a valid date
# --- FIN Función _is_holiday CORREGIDA ---


# --- ¡¡¡FUNCIÓN PRINCIPAL forecast_120d (v31)!!! ---
# (Mantiene firma v1, prepara datos TMO v8/v29 tarde)
def forecast_120d(df_hist_joined: pd.DataFrame, df_tmo_hist_only: pd.DataFrame | None, horizon_days: int = 120, holidays_set: set | None = None):
    """
    Combina la lógica v1 "perfecta" para llamadas con la v8/v29 para TMO.
    Mantiene la firma original v1 pero ignora df_tmo_hist_only internamente para TMO.
    Prepara datos de TMO v8/v29 DESPUÉS de la predicción de llamadas.
    """
    # === Artefactos ===
    print("INFO: Cargando artefactos de modelos...")
    try:
        # Verificar existencia de archivos antes de cargar
        if not os.path.exists(PLANNER_MODEL): raise FileNotFoundError(f"Planner model not found: {PLANNER_MODEL}")
        if not os.path.exists(PLANNER_SCALER): raise FileNotFoundError(f"Planner scaler not found: {PLANNER_SCALER}")
        if not os.path.exists(PLANNER_COLS): raise FileNotFoundError(f"Planner cols not found: {PLANNER_COLS}")
        if not os.path.exists(TMO_MODEL): raise FileNotFoundError(f"TMO model not found: {TMO_MODEL}")
        if not os.path.exists(TMO_SCALER): raise FileNotFoundError(f"TMO scaler not found: {TMO_SCALER}")
        if not os.path.exists(TMO_COLS): raise FileNotFoundError(f"TMO cols not found: {TMO_COLS}")

        print(f"--- DEBUG: Loading Planner model from: {os.path.abspath(PLANNER_MODEL)}")
        m_pl = tf.keras.models.load_model(PLANNER_MODEL, compile=False)
        print(f"--- DEBUG: Loading Planner scaler from: {os.path.abspath(PLANNER_SCALER)}")
        sc_pl = joblib.load(PLANNER_SCALER)
        cols_pl = _load_cols(PLANNER_COLS)

        print(f"--- DEBUG: Loading TMO model from: {os.path.abspath(TMO_MODEL)}")
        m_tmo = tf.keras.models.load_model(TMO_MODEL, compile=False)
        print(f"--- DEBUG: Loading TMO scaler from: {os.path.abspath(TMO_SCALER)}")
        sc_tmo = joblib.load(TMO_SCALER)
        cols_tmo = _load_cols(TMO_COLS)
        print("INFO: Artefactos cargados exitosamente.")
    except Exception as e:
        print(f"ERROR: No se pudieron cargar los artefactos del modelo: {e}")
        import traceback
        traceback.print_exc() # Imprimir traceback completo
        raise

    # === Base histórica (LÓGICA ORIGINAL v1 INTACTA) ===
    print("INFO: Procesando datos históricos (lógica v1)...")
    try:
        df = ensure_ts(df_hist_joined) # (v1, línea 239) - No usar copy aquí
    except ValueError as e:
        print(f"ERROR: Falló ensure_ts en la preparación v1: {e}")
        raise
    except Exception as e: # Capturar otros errores
        print(f"ERROR inesperado en ensure_ts (v1): {e}")
        raise

    df = df[[TARGET_CALLS, TARGET_TMO] if TARGET_TMO in df.columns else [TARGET_CALLS]].copy() # (v1, línea 241)
    df = df.dropna(subset=[TARGET_CALLS]) # (v1, línea 243)
    if df.empty:
        raise ValueError("DataFrame 'df' (v1) quedó vacío después de dropna(TARGET_CALLS).")
    print(f"--- DEBUG: 'df' (v1) shape after initial processing: {df.shape}")

    for c in ["feriados", "es_dia_de_pago",
              "proporcion_comercial", "proporcion_tecnica", "tmo_comercial", "tmo_tecnico"]:
        if c in df.columns: # <-- Siempre Falso en v1
            df[c] = df[c].ffill()

    last_ts = df.index.max()
    start_hist = last_ts - pd.Timedelta(days=HIST_WINDOW_DAYS)

    df_recent = df.loc[df.index >= start_hist].copy()
    if df_recent.empty:
        print("WARN: df_recent (v1) está vacío, usando df completo.")
        df_recent = df.copy()
    print(f"--- DEBUG: 'df_recent' (v1) shape: {df_recent.shape}")

    # ===== Horizonte futuro (Lógica v1) =====
    if pd.isna(last_ts):
        raise ValueError("No se pudo determinar la última fecha válida ('last_ts') desde los datos procesados v1.")

    future_ts = pd.date_range(
        last_ts + pd.Timedelta(hours=1),
        periods=horizon_days * 24,
        freq="h",
        tz=TIMEZONE
    )
    print(f"INFO: Horizonte futuro generado desde {future_ts.min()} hasta {future_ts.max()}")

    # ===== BLOQUE 1: PLANNER DE LLAMADAS (IDÉNTICO AL ORIGINAL 'PERFECTO' v1) =====
    print("Iniciando predicción de Llamadas (Lógica Original v1)...")
    if "feriados" in df_recent.columns: # <-- Siempre Falso en v1
        dfp = df_recent[[TARGET_CALLS, "feriados"]].copy()
    else:
        dfp = df_recent[[TARGET_CALLS]].copy() # <-- Se toma esta rama
    dfp[TARGET_CALLS] = pd.to_numeric(dfp[TARGET_CALLS], errors="coerce").ffill().fillna(0.0)

    for i, ts in enumerate(future_ts):
        if (i + 1) % (24*7) == 0: print(f"  Prediciendo llamadas: Semana { (i + 1) // (24*7) }", end='\r') # Mensaje menos frecuente
        if not isinstance(ts, pd.Timestamp):
             try: ts = pd.to_datetime(ts)
             except: print(f"WARN: Saltando ts inválido (no convertible): {ts}"); continue
        if getattr(ts, "tz", None) is None: ts = ts.tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT')
        if pd.isna(ts): print(f"WARN: Saltando ts NaT en bucle planner"); continue
        current_idx = pd.DatetimeIndex([ts])
        try:
            tmp = pd.concat([dfp, pd.DataFrame(index=current_idx)])
        except Exception as e_concat:
            print(f"ERROR en concat (planner) ts={ts}: {e_concat}"); continue

        tmp[TARGET_CALLS] = tmp[TARGET_CALLS].ffill()
        if "feriados" in tmp.columns: tmp.loc[ts, "feriados"] = _is_holiday(ts, holidays_set) # <-- Siempre Falso en v1
        if not isinstance(tmp.index, pd.DatetimeIndex):
            try:
                tmp.index = pd.to_datetime(tmp.index)
                if getattr(tmp.index, "tz", None) is None: tmp.index = tmp.index.tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT')
            except Exception as e: print(f"WARN: Error convirtiendo índice en bucle planner: {e}"); continue
        try:
            tmp_features = add_lags_mas(tmp, TARGET_CALLS)
            tmp_features = add_time_parts(tmp_features)
            X = dummies_and_reindex(tmp_features.tail(1), cols_pl)
            if X.isnull().values.any():
                nan_cols = X.columns[X.isnull().any()].tolist()
                print(f"WARN: NaNs encontrados en features X para planner en ts={ts}. Columnas: {nan_cols}. Usando ffill.")
                last_valid_call = dfp.iloc[-1][TARGET_CALLS] if not dfp.empty else 0.0
                # Asegurarse que el índice exista antes de asignar
                if ts not in dfp.index: dfp.loc[ts] = np.nan # Añadir fila si no existe
                dfp.loc[ts, TARGET_CALLS] = last_valid_call
            else:
                yhat = float(m_pl.predict(sc_pl.transform(X), verbose=0).flatten()[0])
                if ts not in dfp.index: dfp.loc[ts] = np.nan # Añadir fila si no existe
                dfp.loc[ts, TARGET_CALLS] = max(0.0, yhat)
        except Exception as e_pred:
             print(f"ERROR durante predicción de llamadas en ts={ts}: {e_pred}")
             last_valid_call = dfp.iloc[-1][TARGET_CALLS] if not dfp.empty else 0.0
             if ts not in dfp.index: dfp.loc[ts] = np.nan
             dfp.loc[ts, TARGET_CALLS] = last_valid_call
        if "feriados" in dfp.columns: # <-- Siempre Falso en v1
             if ts not in dfp.index: dfp.loc[ts] = np.nan
             dfp.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)
    print("\nPredicción de Llamadas (v1) completada.")
    pred_calls = dfp.reindex(future_ts).ffill().fillna(0.0)[TARGET_CALLS] # Asegurar índice completo


    # ===== [NUEVO v31] Preparación TARDÍA de datos TMO v8/v29 =====
    print("INFO: Preparando datos para TMO v8/v29...")
    df_full_tmo_v8 = _prepare_full_data(df_hist_joined)
    df_recent_tmo = None
    if df_full_tmo_v8 is not None:
        last_ts_full = df_full_tmo_v8.index.max()
        start_hist_tmo = last_ts_full - pd.Timedelta(days=HIST_WINDOW_DAYS)
        if isinstance(df_full_tmo_v8.index, pd.DatetimeIndex):
            df_recent_tmo = df_full_tmo_v8.loc[df_full_tmo_v8.index >= start_hist_tmo].copy()
            if df_recent_tmo.empty:
                print("WARN: df_recent_tmo vacío, usando df_full_tmo_v8 completo.")
                df_recent_tmo = df_full_tmo_v8.copy()
        else:
             print("WARN: Índice de df_full_tmo_v8 no es DatetimeIndex, no se puede crear df_recent_tmo.")
             df_full_tmo_v8 = None
    # ===== FIN BLOQUE NUEVO v31 =====

    # ===== [NUEVO v31] TMO iterativo v8/v29 =====
    pred_tmo = pd.Series(0.0, index=future_ts) # Fallback: TMO = 0
    if df_full_tmo_v8 is not None and df_recent_tmo is not None:
        try:
            print("Iniciando predicción de TMO (Lógica v8/v29 Autorregresiva)...")
            cols_tmo_hist = [TARGET_TMO]
            if "feriados" in df_recent_tmo.columns: cols_tmo_hist.append("feriados")
            if "es_dia_de_pago" in df_recent_tmo.columns: cols_tmo_hist.append("es_dia_de_pago")
            dft = df_recent_tmo[cols_tmo_hist].copy()
            if "feriados" not in dft.columns: dft["feriados"] = 0
            if "es_dia_de_pago" not in dft.columns: dft["es_dia_de_pago"] = 0
            dft[TARGET_TMO] = pd.to_numeric(dft[TARGET_TMO], errors="coerce").ffill().fillna(0.0)

            for i, ts in enumerate(future_ts):
                if (i + 1) % (24*7) == 0: print(f"  Prediciendo TMO: Semana { (i + 1) // (24*7) }", end='\r')
                if not isinstance(ts, pd.Timestamp):
                    try: ts = pd.to_datetime(ts)
                    except: print(f"WARN: Saltando ts inválido (no convertible TMO): {ts}"); continue
                if getattr(ts, "tz", None) is None: ts = ts.tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT')
                if pd.isna(ts): print(f"WARN: Saltando ts NaT en bucle TMO"); continue
                current_idx = pd.DatetimeIndex([ts])
                try:
                    tmp_t = pd.concat([dft, pd.DataFrame(index=current_idx)])
                except Exception as e_concat_tmo:
                     print(f"ERROR en concat (TMO) ts={ts}: {e_concat_tmo}"); continue

                tmp_t[TARGET_TMO] = tmp_t[TARGET_TMO].ffill()
                tmp_t.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)
                if "es_dia_de_pago" in cols_tmo: tmp_t.loc[ts, "es_dia_de_pago"] = 1 if ts.day in [1,2,15,16,29,30,31] else 0
                if not isinstance(tmp_t.index, pd.DatetimeIndex):
                    try:
                        tmp_t.index = pd.to_datetime(tmp_t.index)
                        if getattr(tmp_t.index, "tz", None) is None: tmp_t.index = tmp_t.index.tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT')
                    except Exception as e: print(f"WARN: Error convirtiendo índice en bucle TMO: {e}"); continue
                try:
                    tmp_t_features = add_lags_mas(tmp_t, TARGET_TMO)
                    tmp_t_features = add_time_parts(tmp_t_features)
                    Xt = dummies_and_reindex(tmp_t_features.tail(1), cols_tmo)
                    if Xt.isnull().values.any():
                        nan_cols_tmo = Xt.columns[Xt.isnull().any()].tolist()
                        print(f"WARN: NaNs encontrados en features Xt para TMO en ts={ts}. Columnas: {nan_cols_tmo}. Usando ffill.")
                        last_valid_tmo = dft.iloc[-1][TARGET_TMO] if not dft.empty else 0.0
                        if ts not in dft.index: dft.loc[ts] = np.nan
                        dft.loc[ts, TARGET_TMO] = last_valid_tmo
                    else:
                        yhat_t = float(m_tmo.predict(sc_tmo.transform(Xt), verbose=0).flatten()[0])
                        if ts not in dft.index: dft.loc[ts] = np.nan
                        dft.loc[ts, TARGET_TMO] = max(0.0, yhat_t)
                except Exception as e_pred_tmo:
                     print(f"ERROR durante predicción de TMO en ts={ts}: {e_pred_tmo}")
                     last_valid_tmo = dft.iloc[-1][TARGET_TMO] if not dft.empty else 0.0
                     if ts not in dft.index: dft.loc[ts] = np.nan
                     dft.loc[ts, TARGET_TMO] = last_valid_tmo
                if "feriados" in dft.columns:
                     if ts not in dft.index: dft.loc[ts] = np.nan
                     dft.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)
                if "es_dia_de_pago" in cols_tmo:
                     if ts not in dft.index: dft.loc[ts] = np.nan
                     dft.loc[ts, "es_dia_de_pago"] = 1 if ts.day in [1,2,15,16,29,30,31] else 0

            pred_tmo_calculated = dft.reindex(future_ts)[TARGET_TMO] # Asegurar índice completo
            pred_tmo = pred_tmo_calculated.ffill().fillna(0.0)
            print("\nPredicción de TMO (v8/v29) completada.")
        except Exception as e_tmo_loop:
            print(f"ERROR: Falló el bucle de predicción de TMO v8/v29: {e_tmo_loop}"); print("WARN: Usando TMO=0 como fallback.")
    else: print("WARN: Saltando predicción TMO v8/v29 debido a datos faltantes. Usando TMO=0.")
    # ===== FIN BLOQUE NUEVO v31 =====


    # ===== Curva base (sin ajuste) =====
    df_hourly = pd.DataFrame(index=future_ts)
    df_hourly["calls"] = np.round(pred_calls).astype(int)
    df_hourly["tmo_s"] = np.round(pred_tmo).astype(int) # <-- Usa la salida del bucle v8/v29 o el fallback

    # ===== AJUSTE POR FERIADOS (LÓGICA V1 INTACTA) =====
    print("Aplicando ajustes de feriados (lógica v1)...")
    if holidays_set and len(holidays_set) > 0:
        df_adj = df.copy() # Usa el df v1 roto
        if not isinstance(df_adj.index, pd.DatetimeIndex):
             try:
                 df_adj.index = pd.to_datetime(df_adj.index)
                 if getattr(df_adj.index, "tz", None) is None: df_adj.index = df_adj.index.tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT')
                 else: df_adj.index = df_adj.index.tz_convert(TIMEZONE)
             except Exception as e: print(f"WARN: No se pudo convertir índice de 'df' para ajustes: {e}"); pass
        (f_calls_by_hour, f_tmo_by_hour, g_calls, g_tmo, post_calls_by_hour) = compute_holiday_factors(df_adj, holidays_set)
        df_hourly = apply_holiday_adjustment(df_hourly, holidays_set, f_calls_by_hour, f_tmo_by_hour, col_calls_future="calls", col_tmo_future="tmo_s")
        df_hourly = apply_post_holiday_adjustment(df_hourly, holidays_set, post_calls_by_hour, col_calls_future="calls")

    # ===== (OPCIONAL) CAP de OUTLIERS (Solo para llamadas) (LÓGICA V1 INTACTA) =====
    if ENABLE_OUTLIER_CAP:
        print("Aplicando guardrail de outliers a llamadas (lógica v1)...")
        df_mad = df.copy() # Usa el df v1 roto
        base_mad = _baseline_median_mad(df_mad, col=TARGET_CALLS)
        df_hourly = apply_outlier_cap(df_hourly, base_mad, holidays_set, col_calls_future="calls", k_weekday=K_WEEKDAY, k_weekend=K_WEEKEND)

    # ===== Erlang por hora (Lógica v1 INTACTA) =====
    print("Calculando agentes requeridos (Erlang C)...")
    df_hourly["agents_prod"] = 0
    try: from .erlang import required_agents, schedule_agents
    except ImportError: print("ERROR: No se pudo importar 'erlang'. Saltando cálculo de agentes."); df_hourly["agents_sched"] = 0; # Continuar sin agentes
    else: # Ejecutar solo si la importación fue exitosa
        for ts in df_hourly.index:
            calls_val = float(df_hourly.at[ts, "calls"])
            tmo_val = float(df_hourly.at[ts, "tmo_s"])
            if calls_val >= 0 and tmo_val > 0:
                 try: a, _ = required_agents(calls_val, tmo_val); df_hourly.at[ts, "agents_prod"] = int(a)
                 except Exception as e_erlang: print(f"WARN: Error en required_agents para ts={ts}: {e_erlang}"); df_hourly.at[ts, "agents_prod"] = 0
            else: df_hourly.at[ts, "agents_prod"] = 0
        df_hourly["agents_sched"] = df_hourly["agents_prod"].apply(schedule_agents)


    # ===== Salidas (Lógica v1 INTACTA) =====
    print("Generando archivos JSON de salida...")
    try: from .utils_io import write_daily_json, write_hourly_json
    except ImportError: print("ERROR: No se pudo importar 'utils_io'. No se guardarán los JSON."); return df_hourly # Devolver df si falla escritura

    # Asegurarse que la columna de agentes exista antes de guardar
    if "agents_sched" not in df_hourly.columns:
        print("WARN: Columna 'agents_sched' no encontrada. Se añadirá con ceros.")
        df_hourly["agents_sched"] = 0

    write_hourly_json(f"{PUBLIC_DIR}/prediccion_horaria.json", df_hourly, "calls", "tmo_s", "agents_sched")
    write_daily_json(f"{PUBLIC_DIR}/prediccion_diaria.json", df_hourly, "calls", "tmo_s")

    print("--- Proceso de Inferencia Finalizado ---")
    return df_hourly
