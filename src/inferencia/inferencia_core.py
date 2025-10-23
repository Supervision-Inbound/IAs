# src/inferencia/inferencia_core.py
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import os

from .features import ensure_ts, add_time_parts, add_lags_mas, dummies_and_reindex
from .erlang import required_agents, schedule_agents
from .utils_io import write_daily_json, write_hourly_json
try:
    from src.main import add_es_dia_de_pago
except ImportError:
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
    with open(path, "r") as f:
        return json.load(f)

def _prepare_full_data(df_hist_joined):
    df_input_copy = df_hist_joined.copy()
    df_full = ensure_ts(df_input_copy)
    if TARGET_CALLS not in df_full.columns:
        print(f"ERROR: Falta columna {TARGET_CALLS} en datos para TMO. TMO se saltará.")
        return None
    else:
        if TARGET_TMO not in df_full.columns:
            print(f"WARN: Falta columna {TARGET_TMO} en datos para TMO. Se usará 0.")
            df_full[TARGET_TMO] = 0.0
        df_full = df_full.dropna(subset=[TARGET_CALLS])
        df_full[TARGET_TMO] = pd.to_numeric(df_full[TARGET_TMO], errors='coerce').ffill().fillna(0.0)
        for aux in ["feriados", "es_dia_de_pago"]:
            if aux in df_full.columns:
                df_full[aux] = df_full[aux].ffill()
            else:
                print(f"WARN: Columna '{aux}' faltante para TMO, se añadirá con ceros.")
                if isinstance(df_full.index, pd.DatetimeIndex):
                     if aux == "feriados": df_full[aux] = 0
                     elif aux == "es_dia_de_pago": df_full[aux] = add_es_dia_de_pago(df_full.index).values
                else: df_full[aux] = 0
        # Check final si el df quedó vacío
        if df_full.empty:
            print("ERROR: df_full quedó vacío después del preprocesamiento para TMO.")
            return None
        return df_full

# ========= Helpers (Función _is_holiday corregida v30) =========
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
    cols = [col_calls]
    if col_tmo in df_hist.columns and not df_hist[col_tmo].isnull().all():
        cols.append(col_tmo)
    df_hist_dt_idx = df_hist.copy()
    if not isinstance(df_hist_dt_idx.index, pd.DatetimeIndex):
         try:
             df_hist_dt_idx.index = pd.to_datetime(df_hist_dt_idx.index)
             if getattr(df_hist_dt_idx.index, "tz", None) is None: df_hist_dt_idx.index = df_hist_dt_idx.index.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT').tz_convert(TIMEZONE)
             else: df_hist_dt_idx.index = df_hist_dt_idx.index.tz_convert(TIMEZONE)
         except Exception as e:
             print(f"WARN: No se pudo convertir índice en compute_holiday_factors: {e}"); default_factors = {h: 1.0 for h in range(24)}; return (default_factors.copy(), default_factors.copy(), 1.0, 1.0, default_factors.copy())
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
    med_post_calls = dfh[dfh["is_post_hol"]].groupby("hour")[col_calls].median()
    post_calls_by_hour = {int(h): _safe_ratio(med_post_calls.get(h, np.nan), med_nor_calls.get(h, np.nan), fallback=1.05) for h in range(24)}
    post_calls_by_hour = {h: float(np.clip(v, 0.90, 1.80)) for h, v in post_calls_by_hour.items()}
    return (factors_calls_by_hour, factors_tmo_by_hour, global_calls_factor, global_tmo_factor, post_calls_by_hour)

def apply_holiday_adjustment(df_future, holidays_set,
                             factors_calls_by_hour, factors_tmo_by_hour,
                             col_calls_future="calls", col_tmo_future="tmo_s"):
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
    g = d.groupby(["dow", "hour"])[col]
    base = g.median().rename("med").to_frame()
    def mad_robust(x):
        x_clean = x.dropna()
        if len(x_clean) == 0: return np.nan
        med = np.median(x_clean)
        if not np.isscalar(med): return np.nan
        return np.median(np.abs(x_clean - med))
    mad = g.apply(mad_robust).rename("mad")
    base = base.join(mad)
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

# --- Función _is_holiday CORREGIDA (v30) ---
def _is_holiday(ts, holidays_set: set) -> int:
    if not holidays_set:
        return 0

    # 1. Ensure ts is a valid Timestamp
    if not isinstance(ts, pd.Timestamp):
        try:
            ts = pd.to_datetime(ts)
        except Exception:
            # print(f"DEBUG: Failed to convert {ts} to datetime") # Optional debug
            return 0 # Cannot proceed if not convertible
    if pd.isna(ts): # Check if conversion resulted in NaT
        # print("DEBUG: Timestamp is NaT") # Optional debug
        return 0

    # 2. Ensure ts is timezone-aware (in the target timezone)
    ts_aware = None
    try:
        if getattr(ts, "tz", None) is None:
            # Localize if naive
            ts_aware = ts.tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT')
        else:
            # Convert if already aware but possibly different timezone
            ts_aware = ts.tz_convert(TIMEZONE)
        # Check again if localization/conversion resulted in NaT
        if pd.isna(ts_aware):
             # print(f"DEBUG: Timestamp became NaT after tz handling: {ts}") # Optional debug
             return 0
    except Exception as e:
        # print(f"DEBUG: Error handling timezone for {ts}: {e}") # Optional debug
        # Fallback: Try using date without timezone if conversion fails
        try:
            d = ts.date()
            # Asegurarse que holidays_set sea un set aquí también
            if holidays_set is None: holidays_set = set()
            return 1 if d in holidays_set else 0
        except Exception:
             # print(f"DEBUG: Failed to get date even without tz for {ts}") # Optional debug
             return 0 # Cannot get date at all

    # 3. Get the date from the timezone-aware timestamp
    try:
        d = ts_aware.date()
        # Asegurarse que holidays_set sea un set
        if holidays_set is None: holidays_set = set()
        return 1 if d in holidays_set else 0
    except Exception as e:
         # print(f"DEBUG: Failed to get date from tz-aware timestamp {ts_aware}: {e}") # Optional debug
         return 0 # Should not happen if ts_aware is valid, but safeguard
# --- FIN Función _is_holiday CORREGIDA ---


# --- ¡¡¡FIRMA ORIGINAL v1 RESTAURADA (con df_tmo_hist_only)!!! ---
def forecast_120d(df_hist_joined: pd.DataFrame, df_tmo_hist_only: pd.DataFrame | None, horizon_days: int = 120, holidays_set: set | None = None):
    """
    Combina la lógica v1 "perfecta" para llamadas con la v8/v29 para TMO.
    Mantiene la firma original v1 pero ignora df_tmo_hist_only internamente para TMO.
    """
    # === Artefactos ===
    m_pl = tf.keras.models.load_model(PLANNER_MODEL, compile=False)
    sc_pl = joblib.load(PLANNER_SCALER)
    cols_pl = _load_cols(PLANNER_COLS)

    m_tmo = tf.keras.models.load_model(TMO_MODEL, compile=False)
    sc_tmo = joblib.load(TMO_SCALER)
    cols_tmo = _load_cols(TMO_COLS)

    # === [NUEVO v29] Pre-cálculo de datos completos para TMO v8/v29 ===
    df_full_tmo_v8 = _prepare_full_data(df_hist_joined)
    df_recent_tmo = None # Inicializar
    if df_full_tmo_v8 is not None:
        last_ts_full = df_full_tmo_v8.index.max()
        start_hist_tmo = last_ts_full - pd.Timedelta(days=HIST_WINDOW_DAYS)
        df_recent_tmo = df_full_tmo_v8.loc[df_full_tmo_v8.index >= start_hist_tmo].copy()
        if df_recent_tmo.empty:
            df_recent_tmo = df_full_tmo_v8.copy()
    # === FIN BLOQUE NUEVO v29 ===


    # === Base histórica (LÓGICA ORIGINAL v1 INTACTA) ===
    # 'df' se usará para el bucle de llamadas y los helpers de ajuste v1.
    df = ensure_ts(df_hist_joined) # (v1, línea 239) - No usar copy aquí
    df = df[[TARGET_CALLS, TARGET_TMO] if TARGET_TMO in df.columns else [TARGET_CALLS]].copy() # (v1, línea 241)
    df = df.dropna(subset=[TARGET_CALLS]) # (v1, línea 243)

    # (v1, líneas 246-249)
    # (TARGET_TMO NO está en esta lista)
    for c in ["feriados", "es_dia_de_pago",
              "proporcion_comercial", "proporcion_tecnica", "tmo_comercial", "tmo_tecnico"]:
        if c in df.columns: # <-- Siempre Falso en v1
            df[c] = df[c].ffill()

    last_ts = df.index.max() # <-- Usa el índice del df v1
    start_hist = last_ts - pd.Timedelta(days=HIST_WINDOW_DAYS)

    # (v1, línea 254)
    # 'df_recent' se crea desde 'df' (v1 "roto")
    df_recent = df.loc[df.index >= start_hist].copy()
    if df_recent.empty:
        df_recent = df.copy()
    # ===============================================

    # ===== Horizonte futuro (Lógica v1) =====
    # Asegurar que last_ts sea válido antes de crear el rango
    if pd.isna(last_ts):
        raise ValueError("No se pudo determinar la última fecha válida ('last_ts') desde los datos procesados v1.")

    future_ts = pd.date_range(
        last_ts + pd.Timedelta(hours=1),
        periods=horizon_days * 24,
        freq="h",
        tz=TIMEZONE
    )

    # ===== BLOQUE 1: PLANNER DE LLAMADAS (IDÉNTICO AL ORIGINAL 'PERFECTO' v1) =====
    # --- ¡¡¡Este bloque usa 'df_recent' (datos v1 "rotos")!!! ---
    print("Iniciando predicción de Llamadas (Lógica Original v1)...")

    # (v1, líneas 273-277)
    if "feriados" in df_recent.columns:
        dfp = df_recent[[TARGET_CALLS, "feriados"]].copy()
    else:
        dfp = df_recent[[TARGET_CALLS]].copy() # <-- Se toma esta rama

    dfp[TARGET_CALLS] = pd.to_numeric(dfp[TARGET_CALLS], errors="coerce").ffill().fillna(0.0)

    for ts in future_ts:
        if not isinstance(ts, pd.Timestamp): ts = pd.to_datetime(ts)
        if getattr(ts, "tz", None) is None: ts = ts.tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT')
        if pd.isna(ts): continue
        current_idx = pd.DatetimeIndex([ts])

        tmp = pd.concat([dfp, pd.DataFrame(index=current_idx)])
        tmp[TARGET_CALLS] = tmp[TARGET_CALLS].ffill()
        if "feriados" in tmp.columns: # <-- Siempre Falso en v1
            tmp.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)

        if not isinstance(tmp.index, pd.DatetimeIndex):
            try:
                tmp.index = pd.to_datetime(tmp.index)
                if getattr(tmp.index, "tz", None) is None: tmp.index = tmp.index.tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT')
            except Exception as e: print(f"WARN: Error convirtiendo índice en bucle planner: {e}"); continue

        tmp = add_lags_mas(tmp, TARGET_CALLS)
        tmp = add_time_parts(tmp)

        X = dummies_and_reindex(tmp.tail(1), cols_pl)
        yhat = float(m_pl.predict(sc_pl.transform(X), verbose=0).flatten()[0])
        dfp.loc[ts, TARGET_CALLS] = max(0.0, yhat)
        if "feriados" in dfp.columns: # <-- Siempre Falso en v1
            dfp.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)

    pred_calls = dfp.loc[future_ts, TARGET_CALLS]
    pred_calls = pred_calls.ffill().fillna(0.0)
    print("Predicción de Llamadas (v1) completada.")
    # --- FIN DEL BLOQUE DE LLAMADAS v1 ---


    # ===== [NUEVO v29 - Cambio 2] TMO iterativo v8/v29 (reemplaza cálculo original v1) =====
    pred_tmo = pd.Series(0.0, index=future_ts) # Fallback: TMO = 0
    if df_full_tmo_v8 is not None and df_recent_tmo is not None: # Ejecutar solo si los datos están listos
        try:
            print("Iniciando predicción de TMO (Lógica v8/v29 Autorregresiva)...")

            cols_tmo_hist = [TARGET_TMO]
            if "feriados" in df_recent_tmo.columns: cols_tmo_hist.append("feriados")
            if "es_dia_de_pago" in df_recent_tmo.columns: cols_tmo_hist.append("es_dia_de_pago")

            # Usar df_recent_tmo (derivado de df_full_tmo_v8)
            dft = df_recent_tmo[cols_tmo_hist].copy()

            if "feriados" not in dft.columns: dft["feriados"] = 0
            if "es_dia_de_pago" not in dft.columns: dft["es_dia_de_pago"] = 0

            dft[TARGET_TMO] = pd.to_numeric(dft[TARGET_TMO], errors="coerce").ffill().fillna(0.0)

            for ts in future_ts: # future_ts ya está definido por la lógica v1
                if not isinstance(ts, pd.Timestamp): ts = pd.to_datetime(ts)
                if getattr(ts, "tz", None) is None: ts = ts.tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT')
                if pd.isna(ts): continue
                current_idx = pd.DatetimeIndex([ts])

                tmp_t = pd.concat([dft, pd.DataFrame(index=current_idx)])
                tmp_t[TARGET_TMO] = tmp_t[TARGET_TMO].ffill()
                tmp_t.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)

                if "es_dia_de_pago" in cols_tmo:
                     tmp_t.loc[ts, "es_dia_de_pago"] = 1 if ts.day in [1,2,15,16,29,30,31] else 0

                if not isinstance(tmp_t.index, pd.DatetimeIndex):
                    try:
                        tmp_t.index = pd.to_datetime(tmp_t.index)
                        if getattr(tmp_t.index, "tz", None) is None: tmp_t.index = tmp_t.index.tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT')
                    except Exception as e: print(f"WARN: Error convirtiendo índice en bucle TMO: {e}"); continue

                tmp_t = add_lags_mas(tmp_t, TARGET_TMO)
                tmp_t = add_time_parts(tmp_t)

                Xt = dummies_and_reindex(tmp_t.tail(1), cols_tmo)
                yhat_t = float(m_tmo.predict(sc_tmo.transform(Xt), verbose=0).flatten()[0])
                dft.loc[ts, TARGET_TMO] = max(0.0, yhat_t)
                dft.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)
                if "es_dia_de_pago" in cols_tmo:
                     dft.loc[ts, "es_dia_de_pago"] = 1 if ts.day in [1,2,15,16,29,30,31] else 0

            pred_tmo_calculated = dft.loc[future_ts, TARGET_TMO]
            pred_tmo = pred_tmo_calculated.ffill().fillna(0.0) # Actualizar pred_tmo si el cálculo fue exitoso
            print("Predicción de TMO (v8/v29) completada.")

        except Exception as e_tmo:
            print(f"ERROR: Falló la predicción de TMO v8/v29: {e_tmo}")
            print("WARN: Usando TMO=0 como fallback.")
            # pred_tmo ya está inicializado a 0.0
    else:
         print("WARN: Saltando predicción TMO v8/v29 debido a datos faltantes. Usando TMO=0.")
    # ===== FIN BLOQUE NUEVO v29 =====


    # ===== Curva base (sin ajuste) =====
    df_hourly = pd.DataFrame(index=future_ts)
    df_hourly["calls"] = np.round(pred_calls).astype(int)
    # --- [NUEVO v29 - Cambio 3] ---
    df_hourly["tmo_s"] = np.round(pred_tmo).astype(int) # <-- Usa la salida del bucle v8/v29 o el fallback

    # ===== AJUSTE POR FERIADOS (LÓGICA V1 INTACTA) =====
    # (v1, línea 369)
    # ¡IMPORTANTE! Usa 'df' (el "roto" v1) para calcular los factores.
    print("Aplicando ajustes de feriados (lógica v1)...")
    if holidays_set and len(holidays_set) > 0:
        df_adj = df.copy()
        if not isinstance(df_adj.index, pd.DatetimeIndex):
             try:
                 df_adj.index = pd.to_datetime(df_adj.index)
                 if getattr(df_adj.index, "tz", None) is None: df_adj.index = df_adj.index.tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT')
                 else: df_adj.index = df_adj.index.tz_convert(TIMEZONE)
             except Exception as e: print(f"WARN: No se pudo convertir índice de 'df' para ajustes: {e}"); pass

        (f_calls_by_hour, f_tmo_by_hour,
         g_calls, g_tmo, post_calls_by_hour) = compute_holiday_factors(df_adj, holidays_set)

        df_hourly = apply_holiday_adjustment(
            df_hourly, holidays_set,
            f_calls_by_hour, f_tmo_by_hour,
            col_calls_future="calls", col_tmo_future="tmo_s"
        )
        df_hourly = apply_post_holiday_adjustment(
            df_hourly, holidays_set, post_calls_by_hour,
            col_calls_future="calls"
        )

    # ===== (OPCIONAL) CAP de OUTLIERS (Solo para llamadas) (LÓGICA V1 INTACTA) =====
    if ENABLE_OUTLIER_CAP:
        print("Aplicando guardrail de outliers a llamadas (lógica v1)...")
        # (v1, línea 385)
        # ¡IMPORTANTE! Usa 'df' (el "roto" v1) para calcular el MAD.
        df_mad = df.copy()
        base_mad = _baseline_median_mad(df_mad, col=TARGET_CALLS)
        df_hourly = apply_outlier_cap(
            df_hourly, base_mad, holidays_set,
            col_calls_future="calls",
            k_weekday=K_WEEKDAY, k_weekend=K_WEEKEND
        )

    # ===== Erlang por hora (Lógica v1 INTACTA) =====
    print("Calculando agentes requeridos (Erlang C)...")
    df_hourly["agents_prod"] = 0
    # Importar localmente para asegurar disponibilidad
    from .erlang import required_agents, schedule_agents
    for ts in df_hourly.index:
        calls_val = float(df_hourly.at[ts, "calls"])
        tmo_val = float(df_hourly.at[ts, "tmo_s"])
        if calls_val >= 0 and tmo_val > 0:
             a, _ = required_agents(calls_val, tmo_val)
             df_hourly.at[ts, "agents_prod"] = int(a)
        else:
             df_hourly.at[ts, "agents_prod"] = 0

    df_hourly["agents_sched"] = df_hourly["agents_prod"].apply(schedule_agents)


    # ===== Salidas (Lógica v1 INTACTA) =====
    print("Generando archivos JSON de salida...")
    # Importar localmente para asegurar disponibilidad
    from .utils_io import write_daily_json, write_hourly_json
    write_hourly_json(f"{PUBLIC_DIR}/prediccion_horaria.json",
                      df_hourly, "calls", "tmo_s", "agents_sched")
    write_daily_json(f"{PUBLIC_DIR}/prediccion_diaria.json",
                     df_hourly, "calls", "tmo_s")

    print("--- Proceso de Inferencia Finalizado ---")
    return df_hourly
