# src/inferencia/inferencia_core.py
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import os # Necesario para PUBLIC_DIR

from .features import ensure_ts, add_time_parts, add_lags_mas, dummies_and_reindex
# --- ¡Importar write_json directamente! ---
from .utils_io import write_json
# --- Erlang y schedule_agents ya no se usan aquí ---

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

# ========= Helpers (Idénticos a v21/v20) =========
# (_safe_ratio, _series_is_holiday, compute_holiday_factors,
#  apply_holiday_adjustment, apply_post_holiday_adjustment,
#  _baseline_median_mad, apply_outlier_cap, _is_holiday)

def _safe_ratio(num, den, fallback=1.0):
    num = float(num) if num is not None and not np.isnan(num) else np.nan
    den = float(den) if den is not None and not np.isnan(den) and den != 0 else np.nan
    if np.isnan(num) or np.isnan(den) or den == 0:
        return fallback
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
             if getattr(df_hist_dt_idx.index, "tz", None) is None:
                 df_hist_dt_idx.index = df_hist_dt_idx.index.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT').tz_convert(TIMEZONE)
             else:
                 df_hist_dt_idx.index = df_hist_dt_idx.index.tz_convert(TIMEZONE)
         except Exception as e:
             print(f"WARN: No se pudo convertir índice en compute_holiday_factors: {e}")
             default_factors = {h: 1.0 for h in range(24)}
             return (default_factors.copy(), default_factors.copy(), 1.0, 1.0, default_factors.copy())
    dfh = add_time_parts(df_hist_dt_idx[cols].copy())
    dfh["is_holiday"] = _series_is_holiday(dfh.index, holidays_set)
    med_hol_calls = dfh[dfh["is_holiday"]].groupby("hour")[col_calls].median()
    med_nor_calls = dfh[~dfh["is_holiday"]].groupby("hour")[col_calls].median()
    if col_tmo in cols:
        med_hol_tmo = dfh[dfh["is_holiday"]].groupby("hour")[col_tmo].median()
        med_nor_tmo = dfh[~dfh["is_holiday"]].groupby("hour")[col_tmo].median()
        g_hol_tmo = dfh[dfh["is_holiday"]][col_tmo].median()
        g_nor_tmo = dfh[~dfh["is_holiday"]][col_tmo].median()
        global_tmo_factor = _safe_ratio(g_hol_tmo, g_nor_tmo, fallback=1.00)
    else: med_hol_tmo = med_nor_tmo = None; global_tmo_factor = 1.00
    g_hol_calls = dfh[dfh["is_holiday"]][col_calls].median()
    g_nor_calls = dfh[~dfh["is_holiday"]][col_calls].median()
    global_calls_factor = _safe_ratio(g_hol_calls, g_nor_calls, fallback=0.75)
    factors_calls_by_hour = {int(h): _safe_ratio(med_hol_calls.get(h, np.nan), med_nor_calls.get(h, np.nan), fallback=global_calls_factor) for h in range(24)}
    if med_hol_tmo is not None: factors_tmo_by_hour = {int(h): _safe_ratio(med_hol_tmo.get(h, np.nan), med_nor_tmo.get(h, np.nan), fallback=global_tmo_factor) for h in range(24)}
    else: factors_tmo_by_hour = {int(h): 1.0 for h in range(24)}
    factors_calls_by_hour = {h: float(np.clip(v, 0.10, 1.60)) for h, v in factors_calls_by_hour.items()}
    factors_tmo_by_hour   = {h: float(np.clip(v, 0.70, 1.50)) for h, v in factors_tmo_by_hour.items()}
    dfh = dfh.copy()
    dfh["is_post_hol"] = (~dfh["is_holiday"]) & (dfh["is_holiday"].shift(1).fillna(False))
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

    # --- FUNCIÓN mad_robust CON INDENTACIÓN CORRECTA ---
    def mad_robust(x):
        x_clean = x.dropna()
        if len(x_clean) == 0:
            return np.nan
        med = np.median(x_clean)
        # Asegurarse que med sea escalar antes de restar
        if not np.isscalar(med):
            return np.nan
        return np.median(np.abs(x_clean - med))
    # --- FIN FUNCIÓN ---

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
    K = np.where(is_weekend, k_weekday, k_weekday).astype(float)
    upper = capped["med"].values + K * capped["mad"].values
    calls_numeric = pd.to_numeric(capped[col_calls_future], errors='coerce').fillna(0.0)
    mask = (~is_hol.values) & (~is_post_hol.values) & (calls_numeric.values > upper)
    capped.loc[mask, col_calls_future] = np.round(upper[mask]).astype(int)
    out = df_future.copy()
    out[col_calls_future] = capped[col_calls_future].astype(int).values
    return out

def _is_holiday(ts, holidays_set: set) -> int:
    if not holidays_set:
        return 0
    if not isinstance(ts, pd.Timestamp):
        try:
            ts = pd.to_datetime(ts)
            # --- CORRECCIÓN AQUÍ ---
            if getattr(ts, "tz", None) is None:
                ts = ts.tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT')
            # --- FIN CORRECCIÓN ---
        except Exception:
            return 0
    if pd.isna(ts):
        return 0
    try:
        # --- CORRECCIÓN AQUÍ ---
        ts_aware = ts.tz_convert(TIMEZONE) if getattr(ts, "tz", None) is not None else ts.tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT')
        if pd.isna(ts_aware):
            return 0
        d = ts_aware.date()
        # --- FIN CORRECCIÓN ---
    except Exception:
        try:
            d = ts.date()
        except Exception:
            return 0
    return 1 if d in holidays_set else 0
# ===========================================================

# ==================================================================
# --- FUNCIÓN 1: PREDICCIÓN DE LLAMADAS (LÓGICA v1 "PERFECTA") ---
# ==================================================================
def forecast_calls_v1(df_hist_joined: pd.DataFrame, horizon_days: int = 120, holidays_set: set | None = None):
    """
    Replica exacta del flujo de predicción de llamadas de v1.
    - Usa el "bug beneficioso" (ignorar feriados/día de pago en el bucle).
    - Usa los helpers de ajuste (calculados sobre el df "roto" v1).
    - Devuelve un df solo con 'calls' ajustadas.
    """
    # === Artefactos (Solo Planner) ===
    m_pl = tf.keras.models.load_model(PLANNER_MODEL, compile=False)
    sc_pl = joblib.load(PLANNER_SCALER)
    cols_pl = _load_cols(PLANNER_COLS)

    # === Base histórica (LÓGICA v1) ===

    # 1. 'df' - REPLICA EXACTA DE LA LÓGICA V1
    df = ensure_ts(df_hist_joined)
    df = df[[TARGET_CALLS, TARGET_TMO] if TARGET_TMO in df.columns else [TARGET_CALLS]].copy()
    df = df.dropna(subset=[TARGET_CALLS])

    # (TARGET_TMO NO está en esta lista, replicando el v1 original)
    for c in ["feriados", "es_dia_de_pago",
              "proporcion_comercial", "proporcion_tecnica", "tmo_comercial", "tmo_tecnico"]:
        if c in df.columns: # <-- Siempre Falso
            df[c] = df[c].ffill()

    last_ts = df.index.max()
    start_hist = last_ts - pd.Timedelta(days=HIST_WINDOW_DAYS)

    df_recent = df.loc[df.index >= start_hist].copy()
    if df_recent.empty:
        df_recent = df.copy()

    # ===== Horizonte futuro =====
    future_ts = pd.date_range(
        last_ts + pd.Timedelta(hours=1),
        periods=horizon_days * 24,
        freq="h",
        tz=TIMEZONE
    )

    # ===== BLOQUE 1: PLANNER DE LLAMADAS (IDÉNTICO AL ORIGINAL 'PERFECTO') =====
    print("Iniciando predicción de Llamadas (Lógica Original v1)...")

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
        if "feriados" in tmp.columns:
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
        if "feriados" in dfp.columns:
            dfp.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)

    pred_calls = dfp.loc[future_ts, TARGET_CALLS]
    pred_calls = pred_calls.ffill().fillna(0.0)
    print("Predicción de Llamadas (v1) completada.")

    # ===== Curva base (sin ajuste) =====
    df_hourly = pd.DataFrame(index=future_ts)
    df_hourly["calls"] = np.round(pred_calls).astype(int)

    # ===== AJUSTE POR FERIADOS (LÓGICA V1) =====
    print("Aplicando ajustes de feriados (lógica v1)...")
    if holidays_set and len(holidays_set) > 0:
        # ¡IMPORTANTE! Usa 'df' (el "roto" v1) para calcular los factores
        (f_calls_by_hour, f_tmo_by_hour,
         g_calls, g_tmo, post_calls_by_hour) = compute_holiday_factors(df, holidays_set)

        df_hourly = apply_holiday_adjustment(
            df_hourly, holidays_set,
            f_calls_by_hour, f_tmo_by_hour,
            col_calls_future="calls", col_tmo_future="tmo_s" # tmo_s no existe aquí
        )
        df_hourly = apply_post_holiday_adjustment(
            df_hourly, holidays_set, post_calls_by_hour,
            col_calls_future="calls"
        )

    # ===== (OPCIONAL) CAP de OUTLIERS (LÓGICA V1) =====
    if ENABLE_OUTLIER_CAP:
        print("Aplicando guardrail de outliers a llamadas (lógica v1)...")
        # ¡IMPORTANTE! Usa 'df' (el "roto" v1) para calcular el MAD
        base_mad = _baseline_median_mad(df, col=TARGET_CALLS)
        df_hourly = apply_outlier_cap(
            df_hourly, base_mad, holidays_set,
            col_calls_future="calls",
            k_weekday=K_WEEKDAY, k_weekend=K_WEEKEND
        )

    # Devuelve solo el df de llamadas
    return df_hourly


# ==================================================================
# --- FUNCIÓN 2: PREDICCIÓN DE TMO (LÓGICA v8 AUTORREGRESIVA) ---
# ==================================================================
def forecast_tmo_v8(df_hist_joined: pd.DataFrame, future_ts: pd.DatetimeIndex, holidays_set: set | None = None):
    """
    Flujo de predicción de TMO v8 (autorregresivo).
    - Usa el 'df_hist_joined' COMPLETO (rellenado por main.py v21).
    - Devuelve una pd.Series 'pred_tmo'.
    """
    # === Artefactos (Solo TMO) ===
    m_tmo = tf.keras.models.load_model(TMO_MODEL, compile=False)
    sc_tmo = joblib.load(TMO_SCALER)
    cols_tmo = _load_cols(TMO_COLS)

    # === Base histórica (LÓGICA V8) ===
    # 1. 'df_full' - Usa la versión completa (con feriados, etc.)
    df_full = ensure_ts(df_hist_joined) # df_hist_joined ya tiene TMO ffill de main.py v21

    if TARGET_CALLS not in df_full.columns: df_full[TARGET_CALLS] = 0
    if TARGET_TMO not in df_full.columns: df_full[TARGET_TMO] = 0

    df_full = df_full.dropna(subset=[TARGET_CALLS])
    df_full[TARGET_TMO] = pd.to_numeric(df_full[TARGET_TMO], errors='coerce').fillna(0.0)
    # ffill de calendario ya hecho en main.py v21

    last_ts = df_full.index.max()
    start_hist = last_ts - pd.Timedelta(days=HIST_WINDOW_DAYS)

    # 2. 'df_recent_tmo' - Creado desde 'df_full'
    df_recent_tmo = df_full.loc[df_full.index >= start_hist].copy()
    if df_recent_tmo.empty:
        df_recent_tmo = df_full.copy()

    # ===== BLOQUE 2: TMO ITERATIVO (LÓGICA v8 QUE FUNCIONA) =====
    print("Iniciando predicción de TMO (Lógica v8 Autorregresiva)...")

    cols_tmo_hist = [TARGET_TMO]
    if "feriados" in df_recent_tmo.columns:
        cols_tmo_hist.append("feriados")
    if "es_dia_de_pago" in df_recent_tmo.columns:
         cols_tmo_hist.append("es_dia_de_pago")

    dft = df_recent_tmo[cols_tmo_hist].copy()

    if "feriados" not in dft.columns: dft["feriados"] = 0
    if "es_dia_de_pago" not in dft.columns: dft["es_dia_de_pago"] = 0

    dft[TARGET_TMO] = pd.to_numeric(dft[TARGET_TMO], errors="coerce").ffill().fillna(0.0)

    for ts in future_ts:
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
        tmp_t = add_time_parts(tmp_t) # add_time_parts espera índice DatetimeIndex

        Xt = dummies_and_reindex(tmp_t.tail(1), cols_tmo)
        yhat_t = float(m_tmo.predict(sc_tmo.transform(Xt), verbose=0).flatten()[0])
        dft.loc[ts, TARGET_TMO] = max(0.0, yhat_t)
        dft.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)
        if "es_dia_de_pago" in cols_tmo:
             dft.loc[ts, "es_dia_de_pago"] = 1 if ts.day in [1,2,15,16,29,30,31] else 0

    pred_tmo = dft.loc[future_ts, TARGET_TMO]
    pred_tmo = pred_tmo.ffill().fillna(0.0)
    print("Predicción de TMO (v8) completada.")

    # Devuelve solo la serie de TMO
    return pred_tmo

# ==================================================================
# --- FUNCIÓN 3: ORQUESTADOR PARA SALIDAS SEPARADAS (NUEVO v25) ---
# ==================================================================
def forecast_separate_outputs(df_hist_joined: pd.DataFrame, horizon_days: int = 120, holidays_set: set | None = None):
    """
    Orquesta las predicciones v1 (llamadas) y v8 (TMO) y genera 4 JSONs separados.
    """
    # 1. Ejecutar predicción de llamadas v1
    #    (Esta función ya aplica ajustes y outliers con lógica v1)
    df_hourly_calls = forecast_calls_v1(
        df_hist_joined.copy(), # Usar copia para asegurar aislamiento
        horizon_days=horizon_days,
        holidays_set=holidays_set
    )
    future_ts = df_hourly_calls.index # Obtener el índice futuro

    # 2. Ejecutar predicción de TMO v8
    #    (main.py v21 ya rellena el TMO histórico antes de llamar aquí)
    pred_tmo = forecast_tmo_v8(
        df_hist_joined.copy(), # Usar copia para asegurar aislamiento
        future_ts,
        holidays_set
    )
    df_hourly_tmo = pd.DataFrame({'tmo_s': np.round(pred_tmo).astype(int)}, index=future_ts)

    # 3. Aplicar ajuste de feriados SOLO al TMO v8 (las llamadas ya están ajustadas)
    print("Aplicando ajustes de feriados (solo TMO v8)...")
    if holidays_set and len(holidays_set) > 0:
        # ¡IMPORTANTE! Calcular factores usando el 'df' v1 (roto) para consistencia con llamadas
        df_v1_hist = ensure_ts(df_hist_joined) # Recrear el df v1 aquí
        df_v1_hist = df_v1_hist[[TARGET_CALLS, TARGET_TMO] if TARGET_TMO in df_v1_hist.columns else [TARGET_CALLS]].copy()
        df_v1_hist = df_v1_hist.dropna(subset=[TARGET_CALLS])
        for c in ["feriados", "es_dia_de_pago","proporcion_comercial", "proporcion_tecnica", "tmo_comercial", "tmo_tecnico"]:
            if c in df_v1_hist.columns: df_v1_hist[c] = df_v1_hist[c].ffill() # Falla silenciosamente

        (f_calls_by_hour, f_tmo_by_hour, # Usamos f_tmo_by_hour
         g_calls, g_tmo, post_calls_by_hour) = compute_holiday_factors(df_v1_hist, holidays_set)

        # Aplicar solo el ajuste de TMO
        df_hourly_tmo = apply_holiday_adjustment(
            df_hourly_tmo, holidays_set,
            {}, # No aplicar factor de llamadas aquí
            f_tmo_by_hour, # Aplicar factor TMO
            col_calls_future="calls", # No existe
            col_tmo_future="tmo_s"
        )
        # Post-holiday no afecta TMO

    # 4. Generar salidas JSON separadas
    print("Generando archivos JSON separados...")
    os.makedirs(PUBLIC_DIR, exist_ok=True) # Asegurar que existe el directorio

    # --- Salidas LLAMADAS v1 ---
    # Horario
    write_separate_hourly_json(
        os.path.join(PUBLIC_DIR, "prediccion_horaria_llamadas.json"),
        df_hourly_calls, calls_col="calls"
    )
    # Diario
    write_separate_daily_json(
        os.path.join(PUBLIC_DIR, "prediccion_diaria_llamadas.json"),
        df_hourly_calls, value_col="calls", output_col_name="llamadas_diarias"
    )

    # --- Salidas TMO v8 ---
     # Horario
    write_separate_hourly_json(
        os.path.join(PUBLIC_DIR, "prediccion_horaria_tmo.json"),
        df_hourly_tmo, tmo_col="tmo_s"
    )
    # Diario
    write_separate_daily_json(
        os.path.join(PUBLIC_DIR, "prediccion_diaria_tmo.json"),
        df_hourly_tmo, value_col="tmo_s", output_col_name="tmo_diario", agg_func="mean"
    )

    print("--- Proceso de Inferencia (Salidas Separadas) Finalizado ---")
    # No se devuelve nada, la función escribe los archivos directamente

# ======================================================================
# --- NUEVAS FUNCIONES DE ESCRITURA JSON (Adaptadas de utils_io.py) ---
# ======================================================================
# (Estas reemplazan la necesidad de modificar utils_io.py)

def write_separate_hourly_json(path: str, df_hourly: pd.DataFrame, calls_col: str | None = None, tmo_col: str | None = None):
    """Escribe JSON horario solo con las columnas presentes."""
    # Asegurar índice datetime
    if not isinstance(df_hourly.index, pd.DatetimeIndex):
        try: df_hourly.index = pd.to_datetime(df_hourly.index)
        except Exception: print(f"WARN: Índice inválido en write_separate_hourly_json para {path}"); return

    out = df_hourly.reset_index().rename(columns={"index":"ts"})
    # Asegurar ts es datetime antes de strftime
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
    out = out.dropna(subset=["ts"])
    out["ts"] = out["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    records = []
    for _, row in out.iterrows():
        record = {"ts": row["ts"]}
        if calls_col and calls_col in row and pd.notna(row[calls_col]):
            record["llamadas_hora"] = int(row[calls_col])
        if tmo_col and tmo_col in row and pd.notna(row[tmo_col]):
             # Convertir a float para consistencia, incluso si es int
            record["tmo_hora"] = float(row[tmo_col])
        # Ignorar agentes
        records.append(record)

    write_json(path, records)

def write_separate_daily_json(path: str, df_hourly: pd.DataFrame, value_col: str, output_col_name: str, agg_func: str = "sum"):
    """Escribe JSON diario para una columna específica (llamadas o TMO)."""
     # Asegurar índice datetime
    if not isinstance(df_hourly.index, pd.DatetimeIndex):
        try: df_hourly.index = pd.to_datetime(df_hourly.index)
        except Exception: print(f"WARN: Índice inválido en write_separate_daily_json para {path}"); return

    tmp = df_hourly.reset_index().rename(columns={"index": "ts"})
    tmp["ts"] = pd.to_datetime(tmp["ts"], errors="coerce")
    tmp = tmp.dropna(subset=["ts"])
    tmp["fecha"] = tmp["ts"].dt.date.astype(str)

    if value_col not in tmp.columns:
        print(f"WARN: Columna '{value_col}' no encontrada para el JSON diario '{path}'. Se generará vacío.")
        daily = pd.DataFrame(columns=["fecha", output_col_name])
    else:
        # Asegurar que la columna sea numérica antes de agregar
        tmp[value_col] = pd.to_numeric(tmp[value_col], errors='coerce')
        # Filtrar NaNs introducidos por coerce antes de groupby
        tmp = tmp.dropna(subset=[value_col])
        if tmp.empty:
             print(f"WARN: No hay datos válidos en '{value_col}' para el JSON diario '{path}'. Se generará vacío.")
             daily = pd.DataFrame(columns=["fecha", output_col_name])
        else:
            daily = (tmp.groupby("fecha", as_index=False)
                        .agg(agg_value=(value_col, agg_func))) # Usar nombre temporal
            daily = daily.rename(columns={"agg_value": output_col_name}) # Renombrar

            # Asegurar tipos serializables
            if agg_func == "sum":
                daily[output_col_name] = daily[output_col_name].fillna(0).astype(int)
            else: # mean u otro
                daily[output_col_name] = daily[output_col_name].fillna(0.0).astype(float)

    write_json(path, daily.to_dict(orient="records"))
