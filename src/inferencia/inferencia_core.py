# src/inferencia/inferencia_core.py
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from .features import ensure_ts, add_time_parts, add_lags_mas, dummies_and_reindex
from .erlang import required_agents, schedule_agents
from .utils_io import write_daily_json, write_hourly_json

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

# Ventana reciente para lags/MA (no afecta last_ts)
HIST_WINDOW_DAYS = 90

# ======= NUEVO: Guardrail Outliers (config) =======
ENABLE_OUTLIER_CAP = True   # <- ponlo en False si quieres desactivarlo
K_WEEKDAY = 6.0             # techos +K*MAD en lun-vie
K_WEEKEND = 7.0             # techos +K*MAD en sáb-dom

# === INICIO CAMBIO: Días de pago (para alinear con entrenamiento) ===
DIAS_DE_PAGO = {1, 2, 15, 16, 29, 30, 31}
# === FIN CAMBIO ===

def _load_cols(path: str):
    with open(path, "r") as f:
        return json.load(f)


# ========= Helpers de FERIADOS (PORTADOS + EXTENDIDOS) =========
def _safe_ratio(num, den, fallback=1.0):
    num = float(num) if num is not None and not np.isnan(num) else np.nan
    den = float(den) if den is not None and not np.isnan(den) and den != 0 else np.nan
    if np.isnan(num) or np.isnan(den) or den == 0:
        return fallback
    return num / den


def _series_is_holiday(idx, holidays_set):
    tz = getattr(idx, "tz", None)
    idx_dates = idx.tz_convert(TIMEZONE).date if tz is not None else idx.date
    return pd.Series([d in holidays_set for d in idx_dates], index=idx, dtype=bool)


def compute_holiday_factors(df_hist, holidays_set,
                            col_calls=TARGET_CALLS, col_tmo=TARGET_TMO):
    """
    Calcula factores por HORA (mediana feriado vs normal) + factores globales,
    y además factores para el DÍA POST-FERIADO por hora.
    Basado en tu forecast3m.py.
    """
    cols = []
    if col_calls in df_hist.columns:
        cols.append(col_calls)
    if col_tmo in df_hist.columns:
        cols.append(col_tmo)
    
    if not cols: # Si ninguna columna objetivo está, retornar defaults
        factors_default = {int(h): 1.0 for h in range(24)}
        return (factors_default, factors_default, 1.0, 1.0, factors_default)

    dfh = add_time_parts(df_hist[cols].copy())
    dfh["is_holiday"] = _series_is_holiday(dfh.index, holidays_set)

    # Medianas por hora (feriado vs normal)
    med_hol_calls = dfh[dfh["is_holiday"]].groupby("hour")[col_calls].median() if col_calls in dfh.columns else pd.Series(dtype=float)
    med_nor_calls = dfh[~dfh["is_holiday"]].groupby("hour")[col_calls].median() if col_calls in dfh.columns else pd.Series(dtype=float)

    if col_tmo in dfh.columns:
        med_hol_tmo = dfh[dfh["is_holiday"]].groupby("hour")[col_tmo].median()
        med_nor_tmo = dfh[~dfh["is_holiday"]].groupby("hour")[col_tmo].median()
        g_hol_tmo = dfh[dfh["is_holiday"]][col_tmo].median()
        g_nor_tmo = dfh[~dfh["is_holiday"]][col_tmo].median()
        global_tmo_factor = _safe_ratio(g_hol_tmo, g_nor_tmo, fallback=1.00)
    else:
        med_hol_tmo = med_nor_tmo = None
        global_tmo_factor = 1.00

    if col_calls in dfh.columns:
        g_hol_calls = dfh[dfh["is_holiday"]][col_calls].median()
        g_nor_calls = dfh[~dfh["is_holiday"]][col_calls].median()
        global_calls_factor = _safe_ratio(g_hol_calls, g_nor_calls, fallback=0.75)
    else:
        global_calls_factor = 1.00 # Fallback si no hay datos de llamadas

    factors_calls_by_hour = {
        int(h): _safe_ratio(med_hol_calls.get(h, np.nan),
                            med_nor_calls.get(h, np.nan),
                            fallback=global_calls_factor)
        for h in range(24)
    }

    if med_hol_tmo is not None:
        factors_tmo_by_hour = {
            int(h): _safe_ratio(med_hol_tmo.get(h, np.nan),
                                med_nor_tmo.get(h, np.nan),
                                fallback=global_tmo_factor)
            for h in range(24)
        }
    else:
        factors_tmo_by_hour = {int(h): 1.0 for h in range(24)}

    # Límites (más permisivo en llamadas, para no cortar picos reales)
    factors_calls_by_hour = {h: float(np.clip(v, 0.10, 1.60)) for h, v in factors_calls_by_hour.items()}
    
    # TMO es más estable, usamos un clip más conservador (este es el que ajustamos antes)
    factors_tmo_by_hour   = {h: float(np.clip(v, 0.85, 1.25)) for h, v in factors_tmo_by_hour.items()}


    # ---- NEW: factores del DÍA POST-FERIADO por hora ----
    dfh = dfh.copy()
    dfh["is_post_hol"] = (~dfh["is_holiday"]) & (dfh["is_holiday"].shift(1).fillna(False))
    
    if col_calls in dfh.columns:
        med_post_calls = dfh[dfh["is_post_hol"]].groupby("hour")[col_calls].median()
        post_calls_by_hour = {
            int(h): _safe_ratio(med_post_calls.get(h, np.nan),
                                med_nor_calls.get(h, np.nan),
                                fallback=1.05)  # leve alza por defecto
            for h in range(24)
        }
    else:
        post_calls_by_hour = {int(h): 1.0 for h in range(24)} # Fallback

    # Más margen en horas punta del rebote
    post_calls_by_hour = {h: float(np.clip(v, 0.90, 1.80)) for h, v in post_calls_by_hour.items()}

    return (factors_calls_by_hour, factors_tmo_by_hour,
            global_calls_factor, global_tmo_factor, post_calls_by_hour)


def apply_holiday_adjustment(df_future, holidays_set,
                             factors_calls_by_hour, factors_tmo_by_hour,
                             col_calls_future="calls", col_tmo_future="tmo_s"):
    """
    Aplica factores por hora SOLO en horas/fechas feriado (idéntico al original).
    """
    d = add_time_parts(df_future.copy())
    is_hol = _series_is_holiday(d.index, holidays_set)

    hours = d["hour"].astype(int).values
    call_f = np.array([factors_calls_by_hour.get(int(h), 1.0) for h in hours])
    tmo_f  = np.array([factors_tmo_by_hour.get(int(h), 1.0) for h in hours])

    out = df_future.copy()
    mask = is_hol.values
    
    # El ajuste de llamadas se mantiene (funciona "perfecto" según el usuario)
    out.loc[mask, col_calls_future] = np.round(out.loc[mask, col_calls_future].astype(float) * call_f[mask]).astype(int)
    
    # === INICIO CAMBIO ===
    # El ajuste de TMO se DESACTIVA (se comenta la línea) porque el modelo ya lo aprendió
    # out.loc[mask, col_tmo_future]   = np.round(out.loc[mask, col_tmo_future].astype(float)   * tmo_f[mask]).astype(int)
    # === FIN CAMBIO ===
    
    return out


def apply_post_holiday_adjustment(df_future, holidays_set, post_calls_by_hour,
                                  col_calls_future="calls"):
    """
    Ajuste para el DÍA POST-FERIADO: si el día anterior fue feriado, aplicar factor por hora.
    """
    idx = df_future.index
    prev_idx = (idx - pd.Timedelta(days=1))
    try:
        prev_dates = prev_idx.tz_convert(TIMEZONE).date
        curr_dates = idx.tz_convert(TIMEZONE).date
    except Exception:
        prev_dates = prev_idx.date
        curr_dates = idx.date

    is_prev_hol = pd.Series([d in holidays_set for d in prev_dates], index=idx, dtype=bool)
    is_today_hol = pd.Series([d in holidays_set for d in curr_dates], index=idx, dtype=bool)
    is_post = (~is_today_hol) & (is_prev_hol)

    d = add_time_parts(df_future.copy())
    hours = d["hour"].astype(int).values
    ph_f = np.array([post_calls_by_hour.get(int(h), 1.0) for h in hours])

    out = df_future.copy()
    mask = is_post.values
    
    # El ajuste post-feriado (solo para llamadas) se mantiene
    out.loc[mask, col_calls_future] = np.round(out.loc[mask, col_calls_future].astype(float) * ph_f[mask]).astype(int)
    return out
# ===========================================================

# ========= NUEVO: Guardrail de outliers por (dow,hour) ======
def _baseline_median_mad(df_hist, col=TARGET_CALLS):
    """
    Baseline robusto por (dow,hour): mediana y MAD.
    """
    d = add_time_parts(df_hist[[col]].copy())
    g = d.groupby(["dow", "hour"])[col]
    base = g.median().rename("med").to_frame()
    mad = g.apply(lambda x: np.median(np.abs(x - np.median(x)))).rename("mad")
    base = base.join(mad)
    # fallback si alguna combinación no tiene MAD
    if base["mad"].isna().all():
        base["mad"] = 0
    base["mad"] = base["mad"].replace(0, base["mad"].median() if not np.isnan(base["mad"].median()) else 1.0)
    return base.reset_index()  # columnas: dow, hour, med, mad


def apply_outlier_cap(df_future, base_median_mad, holidays_set,
                      col_calls_future="calls",
                      k_weekday=K_WEEKDAY, k_weekend=K_WEEKEND):
    """
    Capa picos: pred <= mediana + K*MAD (K diferente en finde).
    No actúa en feriados ni post-feriados.
    """
    if df_future.empty:
        return df_future

    d = add_time_parts(df_future.copy())
    # flags feriado/post-feriado
    prev_idx = (d.index - pd.Timedelta(days=1))
    try:
        curr_dates = d.index.tz_convert(TIMEZONE).date
        prev_dates = prev_idx.tz_convert(TIMEZONE).date
    except Exception:
        curr_dates = d.index.date
        prev_dates = prev_idx.date
    is_hol = pd.Series([dt in holidays_set for dt in curr_dates], index=d.index, dtype=bool) if holidays_set else pd.Series(False, index=d.index)
    is_prev_hol = pd.Series([dt in holidays_set for dt in prev_dates], index=d.index, dtype=bool) if holidays_set else pd.Series(False, index=d.index)
    is_post_hol = (~is_hol) & (is_prev_hol)

    # merge (dow,hour) -> med, mad
    base = base_median_mad.copy()
    capped = d.merge(base, on=["dow","hour"], how="left")
    capped["mad"] = capped["mad"].fillna(capped["mad"].median() if not np.isnan(capped["mad"].median()) else 1.0)
    capped["med"] = capped["med"].fillna(capped["med"].median() if not np.isnan(capped["med"].median()) else 0.0)

    # K por día de semana
    is_weekend = capped["dow"].isin([5,6]).values
    K = np.where(is_weekend, k_weekend, k_weekday).astype(float)

    # techo
    upper = capped["med"].values + K * capped["mad"].values

    # máscara: solo cuando NO es feriado ni post-feriado
    mask = (~is_hol.values) & (~is_post_hol.values) & (capped[col_calls_future].astype(float).values > upper)
    capped.loc[mask, col_calls_future] = np.round(upper[mask]).astype(int)

    out = df_future.copy()
    out[col_calls_future] = capped[col_calls_future].astype(int).values
    return out
# ===========================================================


def _is_holiday(ts, holidays_set: set) -> int:
    if not holidays_set:
        return 0
    try:
        d = ts.tz_convert(TIMEZONE).date()
    except Exception:
        d = ts.date()
    return 1 if d in holidays_set else 0

# === INICIO CAMBIO: Helper para `es_dia_de_pago` ===
def _is_payday(ts) -> int:
    return 1 if ts.day in DIAS_DE_PAGO else 0
# === FIN CAMBIO ===


# --- ¡¡¡INICIO FUNCIÓN MODIFICADA!!! ---
def forecast_120d(df_hist_joined: pd.DataFrame, df_hist_tmo_only: pd.DataFrame | None, horizon_days: int = 120, holidays_set: set | None = None):
    """
    Versión Autorregresiva (v10 - Alineada 100% con Training)
    - Planner (Llamadas) iterativo usando df_hist_joined.
    - TMO (TMO) iterativo usando df_hist_tmo_only como fuente de historia.
    - 'es_dia_de_pago' se calcula correctamente.
    - Ajuste de feriados de TMO (post-proc) DESACTIVADO para evitar double-dipping.
    - Ajuste de feriados de Llamadas (post-proc) MANTENIDO.
    """
    # === Artefactos ===
    m_pl = tf.keras.models.load_model(PLANNER_MODEL, compile=False)
    sc_pl = joblib.load(PLANNER_SCALER)
    cols_pl = _load_cols(PLANNER_COLS)

    m_tmo = tf.keras.models.load_model(TMO_MODEL, compile=False)
    sc_tmo = joblib.load(TMO_SCALER)
    cols_tmo = _load_cols(TMO_COLS)

    # === Base histórica LLAMADAS (Planner) ===
    df_calls = ensure_ts(df_hist_joined)
    
    if TARGET_CALLS not in df_calls.columns:
        raise ValueError(f"Falta columna {TARGET_CALLS} en historical_data.csv")
    
    # === INICIO CAMBIO: Calcular 'es_dia_de_pago' correctamente ===
    if "es_dia_de_pago" not in df_calls.columns:
        df_calls['day'] = df_calls.index.day
        df_calls['es_dia_de_pago'] = df_calls['day'].isin(DIAS_DE_PAGO).astype(int)
    # === FIN CAMBIO ===
    
    # ffill final para llamadas
    cols_to_ffill_calls = [TARGET_CALLS, "feriados", "es_dia_de_pago"]
    for c in cols_to_ffill_calls:
        if c in df_calls.columns:
            df_calls[c] = df_calls[c].ffill()
    
    df_calls = df_calls.dropna(subset=[TARGET_CALLS])
    
    last_ts_calls = df_calls.index.max()
    start_hist_calls = last_ts_calls - pd.Timedelta(days=HIST_WINDOW_DAYS)
    df_recent_calls = df_calls.loc[df_calls.index >= start_hist_calls].copy()
    if df_recent_calls.empty:
        df_recent_calls = df_calls.copy()

    # === Base histórica TMO (Analista) ===
    tmo_static_features = {"proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico"}
    df_tmo = None

    if df_hist_tmo_only is not None and not df_hist_tmo_only.empty:
        try:
            df_tmo = ensure_ts(df_hist_tmo_only)
            print("INFO: Usando HISTORICO_TMO.csv como base para TMO.")
        except Exception as e:
            print(f"WARN: Error procesando df_hist_tmo_only ({e}), usando fallback).")
            df_tmo = None

    if df_tmo is None:
        print("WARN: HISTORICO_TMO.csv no disponible. Usando fallback de TMO desde historical_data.csv")
        df_tmo = df_calls.copy() 

    if TARGET_TMO not in df_tmo.columns:
        df_tmo[TARGET_TMO] = np.nan
    for c in tmo_static_features:
        if c not in df_tmo.columns:
            df_tmo[c] = np.nan
    
    cols_to_ffill_tmo = [TARGET_TMO] + list(tmo_static_features)
    for c in cols_to_ffill_tmo:
        if c in df_tmo.columns:
            df_tmo[c] = df_tmo[c].ffill()
    
    if "feriados" not in df_tmo.columns and holidays_set:
         df_tmo["feriados"] = _series_is_holiday(df_tmo.index, holidays_set).astype(int)
    elif "feriados" not in df_tmo.columns:
        df_tmo["feriados"] = 0
            
    last_ts_tmo = df_tmo.index.max()
    start_hist_tmo = last_ts_tmo - pd.Timedelta(days=HIST_WINDOW_DAYS)
    df_recent_tmo = df_tmo.loc[df_tmo.index >= start_hist_tmo].copy()
    if df_recent_tmo.empty:
        df_recent_tmo = df_tmo.copy()


    # === Valores Estáticos (Mediana Robusta) ===
    df_for_tmo_features = df_tmo.ffill() 
    
    last_ts_features = df_for_tmo_features.index.max()
    recent_data = df_for_tmo_features.loc[
        (df_for_tmo_features.index >= last_ts_features - pd.Timedelta(days=14)) &
        (df_for_tmo_features.index.hour >= 8) &
        (df_for_tmo_features.index.hour <= 20)
    ]
    features_to_agg = list(tmo_static_features.intersection(df_for_tmo_features.columns))
    
    if recent_data.empty or not features_to_agg or recent_data[features_to_agg].isnull().all().all():
        print("WARN: No se encontraron datos TMO robustos. Usando iloc[-1].")
        last_vals_agg = df_for_tmo_features.iloc[[-1]]
    else:
        last_vals_agg_series = recent_data[features_to_agg].median()
        last_vals_agg = pd.DataFrame(last_vals_agg_series).T
        print(f"INFO: Usando valores TMO robustos (mediana 14d): {last_vals_agg.to_dict('records')[0]}")
    
    static_tmo_cols_dict = {}
    for c in tmo_static_features:
        static_tmo_cols_dict[c] = float(last_vals_agg[c].iloc[0]) if c in last_vals_agg.columns and not pd.isna(last_vals_agg[c].iloc[0]) else 0.0

    # ===== Horizonte futuro =====
    last_ts = max(last_ts_calls, last_ts_tmo)
    
    future_ts = pd.date_range(
        last_ts + pd.Timedelta(hours=1),
        periods=horizon_days * 24,
        freq="h",
        tz=TIMEZONE
    )

    # ===== Bucle Iterativo (Llamadas + TMO) =====
    
    # dfp_calls (DataFrame de predicción de llamadas)
    cols_iter_calls = [TARGET_CALLS]
    if "feriados" in df_recent_calls.columns:
        cols_iter_calls.append("feriados")
    if "es_dia_de_pago" in df_recent_calls.columns:
        cols_iter_calls.append("es_dia_de_pago")
            
    dfp_calls = df_recent_calls[cols_iter_calls].copy()
    
    # dfp_tmo (DataFrame de predicción de TMO)
    cols_iter_tmo = [TARGET_TMO] + list(tmo_static_features)
    dfp_tmo = df_recent_tmo[cols_iter_tmo].copy()

    # dfp_full (DataFrame para la iteración)
    dfp_full = dfp_calls.join(dfp_tmo, how='outer')

    if "feriados" not in dfp_full.columns and "feriados" in df_calls.columns:
        dfp_full["feriados"] = df_calls["feriados"]
    if "es_dia_de_pago" not in dfp_full.columns and "es_dia_de_pago" in df_calls.columns:
        dfp_full["es_dia_de_pago"] = df_calls["es_dia_de_pago"]

    for c, val in static_tmo_cols_dict.items():
        if c not in dfp_full.columns:
            dfp_full[c] = val
        else:
            dfp_full[c] = dfp_full[c].fillna(val)

    # Rellenar los targets (ffill)
    dfp_full[TARGET_CALLS] = dfp_full[TARGET_CALLS].ffill().fillna(0.0)
    dfp_full[TARGET_TMO] = dfp_full[TARGET_TMO].ffill().fillna(0.0)
    
    # === INICIO CAMBIO: Asegurar que 'es_dia_de_pago' esté lleno ===
    if "es_dia_de_pago" in dfp_full.columns:
         dfp_full["es_dia_de_pago"] = dfp_full["es_dia_de_pago"].ffill().fillna(0)
    # === FIN CAMBIO ===

    print("Iniciando predicción iterativa (Llamadas + TMO)...")
    for ts in future_ts:
        # 1. Crear el 'slice' temporal para esta hora (historia + 'ts' vacío)
        tmp = pd.concat([dfp_full, pd.DataFrame(index=[ts])])
        
        # 2. ffill de los targets y features estáticas
        tmp[TARGET_CALLS] = tmp[TARGET_CALLS].ffill()
        tmp[TARGET_TMO] = tmp[TARGET_TMO].ffill()
        for c in static_tmo_cols_dict.keys():
            tmp.loc[ts, c] = static_tmo_cols_dict[c] # Propagar valor estático
        
        if "feriados" in tmp.columns:
            tmp.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)
        
        # === INICIO CAMBIO: Calcular 'es_dia_de_pago' para el futuro ===
        if "es_dia_de_pago" in tmp.columns:
            tmp.loc[ts, "es_dia_de_pago"] = _is_payday(ts)
        # === FIN CAMBIO ===

        # 3. Crear TODOS los features (Lags, MAs, Tiempo)
        tmp_with_feats = add_lags_mas(tmp, TARGET_CALLS) 
        
        for lag in [24, 48, 72, 168]:
            tmp_with_feats[f'tmo_lag_{lag}'] = tmp_with_feats[TARGET_TMO].shift(lag)
        for window in [24, 72, 168]:
            tmp_with_feats[f'tmo_ma_{window}'] = tmp_with_feats[TARGET_TMO].rolling(window, min_periods=1).mean()
        
        tmp_with_feats = add_time_parts(tmp_with_feats)
        
        # 4. PREDECIR LLAMADAS (PLANNER)
        current_row = tmp_with_feats.tail(1)
        
        X_pl = dummies_and_reindex(current_row, cols_pl)
        yhat_calls = float(m_pl.predict(sc_pl.transform(X_pl), verbose=0).flatten()[0])
        yhat_calls = max(0.0, yhat_calls)
        
        dfp_full.loc[ts, TARGET_CALLS] = yhat_calls
        
        # 5. PREDECIR TMO (ANALISTA)
        current_row.loc[ts, TARGET_CALLS] = yhat_calls 
        
        X_tmo = dummies_and_reindex(current_row, cols_tmo)
        yhat_tmo = float(m_tmo.predict(sc_tmo.transform(X_tmo), verbose=0).flatten()[0])
        yhat_tmo = max(0.0, yhat_tmo)
        
        dfp_full.loc[ts, TARGET_TMO] = yhat_tmo
        
        # 6. Guardar feriado/dia_pago (ya hecho)
        if "feriados" in dfp_full.columns:
            dfp_full.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)
        if "es_dia_de_pago" in dfp_full.columns:
            dfp_full.loc[ts, "es_dia_de_pago"] = _is_payday(ts)

    print("Predicción iterativa completada.")

    # ===== Curva base (extraída del bucle) =====
    df_hourly = pd.DataFrame(index=future_ts)
    df_hourly["calls"] = np.round(dfp_full.loc[future_ts, TARGET_CALLS]).astype(int)
    df_hourly["tmo_s"] = np.round(dfp_full.loc[future_ts, TARGET_TMO]).astype(int)

    # ===== AJUSTE POR FERIADOS =====
    if holidays_set and len(holidays_set) > 0:
        
        # Factores de Llamadas (usa la historia de llamadas: df_calls)
        (f_calls_by_hour, f_tmo_by_hour_calc, _, _, post_calls_by_hour) = compute_holiday_factors(
            df_calls, holidays_set, col_calls=TARGET_CALLS, col_tmo="col_tmo_fake"
        )

        # === INICIO CAMBIO: Desactivar 'double-dipping' para TMO ===
        # El modelo de TMO ya fue entrenado con 'feriados', no necesita un 2do ajuste.
        # El modelo de LLAMADAS (que funciona "perfecto") SÍ MANTIENE su ajuste.
        print("INFO: Ajuste TMO post-proc DESACTIVADO (modelo ya entrenado con 'feriados').")
        f_tmo_by_hour = {int(h): 1.0 for h in range(24)} # Forzar factor a 1.0
        # === FIN CAMBIO ===
        
        df_hourly = apply_holiday_adjustment(
            df_hourly, holidays_set,
            f_calls_by_hour, f_tmo_by_hour, # Pasamos los factores de TMO en 1.0
            col_calls_future="calls", col_tmo_future="tmo_s"
        )
        # El ajuste post-feriado (solo para llamadas) se mantiene
        df_hourly = apply_post_holiday_adjustment(
            df_hourly, holidays_set, post_calls_by_hour,
            col_calls_future="calls"
        )

    # ===== (OPCIONAL) CAP de OUTLIERS =====
    if ENABLE_OUTLIER_CAP:
        base_mad = _baseline_median_mad(df_calls, col=TARGET_CALLS)
        df_hourly = apply_outlier_cap(
            df_hourly, base_mad, holidays_set,
            col_calls_future="calls",
            k_weekday=K_WEEKDAY, k_weekend=K_WEEKEND 
        )

    # ===== Erlang por hora =====
    df_hourly["agents_prod"] = 0
    for ts in df_hourly.index:
        a, _ = required_agents(float(df_hourly.at[ts, "calls"]), float(df_hourly.at[ts, "tmo_s"]))
        df_hourly.at[ts, "agents_prod"] = int(a)
    df_hourly["agents_sched"] = df_hourly["agents_prod"].apply(schedule_agents)

    # ===== Salidas =====
    write_hourly_json(f"{PUBLIC_DIR}/prediccion_horaria.json",
                      df_hourly, "calls", "tmo_s", "agents_sched")
    write_daily_json(f"{PUBLIC_DIR}/prediccion_diaria.json",
                     df_hourly, "calls", "tmo_s")

    return df_hourly
# --- ¡¡¡FIN FUNCIÓN MODIFICADA!!! ---
