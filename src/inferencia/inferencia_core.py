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

# Guardrail específico para TMO (usa distribución histórica)
ENABLE_TMO_CAP = True
TMO_CAP_K = 6.0             # +K*MAD por (dow,hour)
TMO_CAP_P95_FACTOR = 1.10   # 110% del p95 horario como límite adicional

# === Días de pago (para alinear con entrenamiento) ===
DIAS_DE_PAGO = {1, 2, 15, 16, 29, 30, 31}

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
    """
    cols = []
    if col_calls in df_hist.columns:
        cols.append(col_calls)
    if col_tmo in df_hist.columns:
        cols.append(col_tmo)
    
    if not cols: 
        factors_default = {int(h): 1.0 for h in range(24)}
        return (factors_default, factors_default, 1.0, 1.0, factors_default)

    dfh = add_time_parts(df_hist[cols].copy())
    dfh["is_holiday"] = _series_is_holiday(dfh.index, holidays_set)

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
        global_calls_factor = 1.00

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

    factors_calls_by_hour = {h: float(np.clip(v, 0.10, 1.60)) for h, v in factors_calls_by_hour.items()}
    factors_tmo_by_hour   = {h: float(np.clip(v, 0.85, 1.25)) for h, v in factors_tmo_by_hour.items()}

    dfh = dfh.copy()
    dfh["is_post_hol"] = (~dfh["is_holiday"]) & (dfh["is_holiday"].shift(1).fillna(False))
    
    if col_calls in dfh.columns:
        med_post_calls = dfh[dfh["is_post_hol"]].groupby("hour")[col_calls].median()
        post_calls_by_hour = {
            int(h): _safe_ratio(med_post_calls.get(h, np.nan),
                                med_nor_calls.get(h, np.nan),
                                fallback=1.05)
            for h in range(24)
        }
    else:
        post_calls_by_hour = {int(h): 1.0 for h in range(24)}

    post_calls_by_hour = {h: float(np.clip(v, 0.90, 1.80)) for h, v in post_calls_by_hour.items()}

    return (factors_calls_by_hour, factors_tmo_by_hour,
            global_calls_factor, global_tmo_factor, post_calls_by_hour)


def apply_holiday_adjustment(df_future, holidays_set,
                             factors_calls_by_hour, factors_tmo_by_hour,
                             col_calls_future="calls", col_tmo_future="tmo_s"):
    """
    Aplica factores por hora SOLO en horas/fechas feriado.
    """
    d = add_time_parts(df_future.copy())
    is_hol = _series_is_holiday(d.index, holidays_set)

    hours = d["hour"].astype(int).values
    call_f = np.array([factors_calls_by_hour.get(int(h), 1.0) for h in hours])
    tmo_f  = np.array([factors_tmo_by_hour.get(int(h), 1.0) for h in hours])

    out = df_future.copy()
    mask = is_hol.values
    
    out.loc[mask, col_calls_future] = np.round(out.loc[mask, col_calls_future].astype(float) * call_f[mask]).astype(int)
    
    # El ajuste de TMO se DESACTIVA (se comenta la línea) porque el modelo ya lo aprendió (v10)
    # out.loc[mask, col_tmo_future]   = np.round(out.loc[mask, col_tmo_future].astype(float)   * tmo_f[mask]).astype(int)
    
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
    
    out.loc[mask, col_calls_future] = np.round(out.loc[mask, col_calls_future].astype(float) * ph_f[mask]).astype(int)
    return out
# ===========================================================

# ========= NUEVO: Guardrail de outliers por (dow,hour) ======
def _baseline_median_mad(df_hist, col=TARGET_CALLS):
    d = add_time_parts(df_hist[[col]].copy())
    g = d.groupby(["dow", "hour"])[col]
    base = g.median().rename("med").to_frame()
    mad = g.apply(lambda x: np.median(np.abs(x - np.median(x)))).rename("mad")
    base = base.join(mad)
    if base["mad"].isna().all():
        base["mad"] = 0
    base["mad"] = base["mad"].replace(0, base["mad"].median() if not np.isnan(base["mad"].median()) else 1.0)
    return base.reset_index()


def apply_outlier_cap(df_future, base_median_mad, holidays_set,
                      col_calls_future="calls",
                      k_weekday=K_WEEKDAY, k_weekend=K_WEEKEND):
    if df_future.empty:
        return df_future

    d = add_time_parts(df_future.copy())
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

    base = base_median_mad.copy()
    capped = d.merge(base, on=["dow","hour"], how="left")
    capped["mad"] = capped["mad"].fillna(capped["mad"].median() if not np.isnan(capped["mad"].median()) else 1.0)
    capped["med"] = capped["med"].fillna(capped["med"].median() if not np.isnan(capped["med"].median()) else 0.0)

    is_weekend = capped["dow"].isin([5,6]).values
    K = np.where(is_weekend, k_weekend, k_weekday).astype(float)
    upper = capped["med"].values + K * capped["mad"].values

    mask = (~is_hol.values) & (~is_post_hol.values) & (capped[col_calls_future].astype(float).values > upper)
    capped.loc[mask, col_calls_future] = np.round(upper[mask]).astype(int)

    out = df_future.copy()
    out[col_calls_future] = capped[col_calls_future].astype(int).values
    return out
# ===========================================================


def _baseline_tmo_guardrail(df_hist, col=TARGET_TMO):
    if df_hist is None or df_hist.empty or col not in df_hist.columns:
        return None

    d = df_hist[[col]].dropna().copy()
    if d.empty:
        return None

    d = add_time_parts(d)
    g = d.groupby(["dow", "hour"])[col]

    med = g.median().rename("med")
    mad = g.apply(lambda x: float(np.median(np.abs(x - np.median(x)))))
    p95 = g.quantile(0.95).rename("p95")

    base = pd.concat([med, mad.rename("mad"), p95], axis=1)

    fallback_med = float(d[col].median())
    fallback_mad = float(np.median(np.abs(d[col] - fallback_med)))
    if np.isnan(fallback_mad) or fallback_mad <= 0:
        fallback_mad = 1.0
    fallback_p95 = float(d[col].quantile(0.95))
    if np.isnan(fallback_p95) or fallback_p95 <= 0:
        fallback_p95 = fallback_med

    base["mad"] = base["mad"].replace(0, np.nan).fillna(fallback_mad)
    base["med"] = base["med"].fillna(fallback_med)
    base["p95"] = base["p95"].fillna(fallback_p95)

    base = base.reset_index()
    return {
        "by_dow_hour": base,
        "fallback_med": fallback_med,
        "fallback_mad": fallback_mad,
        "fallback_p95": fallback_p95,
    }


def apply_tmo_guardrail(df_future, guardrail_data, col_tmo_future="tmo_s",
                        k=TMO_CAP_K, p95_factor=TMO_CAP_P95_FACTOR):
    if guardrail_data is None or df_future.empty or col_tmo_future not in df_future.columns:
        return df_future

    base = guardrail_data.get("by_dow_hour")
    if base is None or base.empty:
        return df_future

    fallback_med = guardrail_data.get("fallback_med", 0.0)
    fallback_mad = guardrail_data.get("fallback_mad", 1.0)
    fallback_p95 = guardrail_data.get("fallback_p95", fallback_med)

    fallback_mad = float(fallback_mad) if not np.isnan(fallback_mad) and fallback_mad > 0 else 1.0
    fallback_med = float(fallback_med) if not np.isnan(fallback_med) else 0.0
    fallback_p95 = float(fallback_p95) if not np.isnan(fallback_p95) and fallback_p95 > 0 else fallback_med

    d = add_time_parts(df_future.copy())
    merged = d.merge(base[["dow", "hour", "med", "mad", "p95"]], on=["dow", "hour"], how="left")

    merged["med"] = merged["med"].fillna(fallback_med)
    merged["mad"] = merged["mad"].replace(0, np.nan).fillna(fallback_mad)
    merged["p95"] = merged["p95"].fillna(fallback_p95)

    caps = np.minimum(merged["med"].values + k * merged["mad"].values,
                      merged["p95"].values * p95_factor)
    global_cap = min(fallback_med + k * fallback_mad, fallback_p95 * p95_factor)
    caps = np.where(np.isnan(caps), global_cap, caps)

    out = df_future.copy()
    values = out[col_tmo_future].astype(float).values
    mask = values > caps
    if np.any(mask):
        capped_vals = np.round(caps[mask]).astype(int)
        out.loc[out.index[mask], col_tmo_future] = capped_vals
    return out


def _is_holiday(ts, holidays_set: set) -> int:
    if not holidays_set:
        return 0
    try:
        d = ts.tz_convert(TIMEZONE).date()
    except Exception:
        d = ts.date()
    return 1 if d in holidays_set else 0

def _is_payday(ts) -> int:
    return 1 if ts.day in DIAS_DE_PAGO else 0


# --- ¡¡¡INICIO FUNCIÓN MODIFICADA!!! ---
def forecast_120d(df_hist_joined: pd.DataFrame, df_hist_tmo_only: pd.DataFrame | None, horizon_days: int = 120, holidays_set: set | None = None):
    """
    Versión (v14.1 - Lógica Prototipo v7 + Fix TypeError)
    - SOLUCIÓN:
        - 1. LLAMADAS: Se mantiene el bucle autorregresivo (funciona bien).
        - 2. TMO: Se elimina el bucle. Se usa la lógica VECTORIZADA
          del prototipo 'v_main' Fase 6.
    - FIX (v14.1): Se cambia 'tmo_static_features' de set a list
      para evitar TypeError en pandas.
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
    
    if "es_dia_de_pago" not in df_calls.columns:
        df_calls['day'] = df_calls.index.day
        df_calls['es_dia_de_pago'] = df_calls['day'].isin(DIAS_DE_PAGO).astype(int)
    
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
    
    # === INICIO CAMBIO (v14.1): Corregir TypeError cambiando set {} por list [] ===
    tmo_static_features = ["proporcion_comercial", "proporcion_tecnica", "tmo_comercial", "tmo_tecnico"]
    # === FIN CAMBIO ===
    
    df_tmo = None
    is_fallback = False # Flag para saber si usamos la data mala
    tmo_guardrail_data = None

    if df_hist_tmo_only is not None and not df_hist_tmo_only.empty:
        try:
            df_tmo = ensure_ts(df_hist_tmo_only)
            # --- Carga de features TMO desde el loader ---
            # (Asumimos que loader_tmo.py ya se ejecutó en main.py y las columnas están)
            if "proporcion_comercial" not in df_tmo.columns:
                 print("WARN: 'proporcion_comercial' no encontrada en df_tmo_hist_only. Calculando...")
                 # Lógica de fallback del loader
                 if 'q_llamadas_comercial' in df_tmo.columns and 'q_llamadas_general' in df_tmo.columns:
                     df_tmo['proporcion_comercial'] = df_tmo['q_llamadas_comercial'] / (df_tmo['q_llamadas_general'] + 1e-6)
                     df_tmo['proporcion_tecnica'] = df_tmo['q_llamadas_tecnico'] / (df_tmo['q_llamadas_general'] + 1e-6)
                 else:
                     df_tmo['proporcion_comercial'] = 0.0
                     df_tmo['proporcion_tecnica'] = 0.0

            print("INFO: Usando HISTORICO_TMO.csv como base para TMO.")
        except Exception as e:
            print(f"WARN: Error procesando df_hist_tmo_only ({e}), usando fallback).")
            df_tmo = None

    if df_tmo is None:
        print("WARN: HISTORICO_TMO.csv no disponible. Usando fallback de TMO desde historical_data.csv")
        df_tmo = df_calls.copy() # Usar la base de llamadas como fallback
        is_fallback = True

    # Asegurar que existan todas las columnas TMO
    if TARGET_TMO not in df_tmo.columns:
        df_tmo[TARGET_TMO] = np.nan
    for c in tmo_static_features:
        if c not in df_tmo.columns:
            # Si es el fallback, es posible que no tenga estas columnas
            # (ej. tmo_comercial no está en historical_data.csv)
            if is_fallback:
                print(f"WARN: Columna estática '{c}' no encontrada en fallback. Rellenando con 0.")
            df_tmo[c] = np.nan
    
    # ffill de la historia TMO
    cols_to_ffill_tmo = [TARGET_TMO] + tmo_static_features
    for c in cols_to_ffill_tmo:
        if c in df_tmo.columns:
            df_tmo[c] = df_tmo[c].ffill()

    if not is_fallback:
        tmo_guardrail_data = _baseline_tmo_guardrail(df_tmo, col=TARGET_TMO)

    # Usar el ÚLTIMO valor histórico como estático (lógica v_main)
    if df_tmo.empty:
        print("WARN: No hay datos históricos de TMO. Usando 0.0 para features estáticas.")
        last_tmo_data_static = {c: 0.0 for c in tmo_static_features}
    else:
        # === INICIO CAMBIO (v14.1): Asegurar que todas las columnas existan antes de indexar ===
        # El fallback (df_calls) podría no tener 'tmo_comercial', etc.
        valid_static_features = [c for c in tmo_static_features if c in df_tmo.columns]
        last_tmo_data_static = df_tmo.iloc[-1][valid_static_features].to_dict()
        
        # Añadir las que faltaron (si alguna) con 0.0
        for c in tmo_static_features:
            if c not in last_tmo_data_static:
                last_tmo_data_static[c] = 0.0
            elif pd.isna(last_tmo_data_static[c]):
                last_tmo_data_static[c] = 0.0
        # === FIN CAMBIO ===
                
    print(f"INFO: Usando valores TMO estáticos (prototipo v_main): {last_tmo_data_static}")
            
    if "feriados" not in df_tmo.columns and holidays_set:
         df_tmo["feriados"] = _series_is_holiday(df_tmo.index, holidays_set).astype(int)
    elif "feriados" not in df_tmo.columns:
        df_tmo["feriados"] = 0

    # ===== Horizonte futuro =====
    last_ts = last_ts_calls # El TMO histórico ya no importa para el start
    
    future_ts = pd.date_range(
        last_ts + pd.Timedelta(hours=1),
        periods=horizon_days * 24,
        freq="h",
        tz=TIMEZONE
    )

    # ===== Bucle Iterativo (SÓLO LLAMADAS) =====
    
    cols_iter_calls = [TARGET_CALLS]
    if "feriados" in df_recent_calls.columns:
        cols_iter_calls.append("feriados")
    if "es_dia_de_pago" in df_recent_calls.columns:
        cols_iter_calls.append("es_dia_de_pago")
            
    dfp_calls = df_recent_calls[cols_iter_calls].copy()
    
    dfp_full = dfp_calls.copy()
    
    dfp_full[TARGET_CALLS] = dfp_full[TARGET_CALLS].ffill().fillna(0.0)
    if "es_dia_de_pago" in dfp_full.columns:
         dfp_full["es_dia_de_pago"] = dfp_full["es_dia_de_pago"].ffill().fillna(0)

    print("Iniciando predicción iterativa (SÓLO Llamadas)...")
    for ts in future_ts:
        tmp = pd.concat([dfp_full, pd.DataFrame(index=[ts])])
        tmp[TARGET_CALLS] = tmp[TARGET_CALLS].ffill()
        
        if "feriados" in tmp.columns:
            tmp.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)
        
        if "es_dia_de_pago" in tmp.columns:
            tmp.loc[ts, "es_dia_de_pago"] = _is_payday(ts)

        tmp_with_feats = add_lags_mas(tmp, TARGET_CALLS) 
        tmp_with_feats = add_time_parts(tmp_with_feats)
        
        current_row = tmp_with_feats.tail(1)
        
        X_pl = dummies_and_reindex(current_row, cols_pl)
        yhat_calls = float(m_pl.predict(sc_pl.transform(X_pl), verbose=0).flatten()[0])
        yhat_calls = max(0.0, yhat_calls)
        
        dfp_full.loc[ts, TARGET_CALLS] = yhat_calls
        
        if "feriados" in dfp_full.columns:
            dfp_full.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)
        if "es_dia_de_pago" in dfp_full.columns:
            dfp_full.loc[ts, "es_dia_de_pago"] = _is_payday(ts)

    print("Predicción iterativa de llamadas completada.")

    # ===== Predicción TMO (VECTORIZADA, Lógica v_main Fase 6) =====
    
    print("Iniciando predicción vectorizada (TMO)...")
    
    df_future = dfp_full.loc[future_ts].copy()
    df_future_tmo_features = add_time_parts(df_future)
    df_future_tmo_features[TARGET_CALLS] = df_future_tmo_features[TARGET_CALLS]

    for col_name, static_value in last_tmo_data_static.items():
        df_future_tmo_features[col_name] = static_value
        
    X_tmo = dummies_and_reindex(df_future_tmo_features, cols_tmo)
    
    X_tmo_s = sc_tmo.transform(X_tmo)
    yhat_tmo_vector = m_tmo.predict(X_tmo_s, verbose=0).flatten()

    df_hourly = pd.DataFrame(index=future_ts)
    df_hourly["calls"] = np.round(df_future[TARGET_CALLS]).astype(int)
    df_hourly["tmo_s"] = np.round(np.clip(yhat_tmo_vector, 0, None)).astype(int)
    
    print("Predicción TMO vectorizada completada.")
    
    # ===== AJUSTE POR FERIADOS =====
    if holidays_set and len(holidays_set) > 0:
        
        (f_calls_by_hour, _, _, _, post_calls_by_hour) = compute_holiday_factors(
            df_calls, holidays_set, col_calls=TARGET_CALLS, col_tmo="col_tmo_fake"
        )

        f_tmo_by_hour = {int(h): 1.0 for h in range(24)}
        
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
        base_mad = _baseline_median_mad(df_calls, col=TARGET_CALLS)
        df_hourly = apply_outlier_cap(
            df_hourly, base_mad, holidays_set,
            col_calls_future="calls",
            k_weekday=K_WEEKDAY, k_weekend=K_WEEKEND
        )

    if ENABLE_TMO_CAP and not is_fallback and tmo_guardrail_data is not None:
        df_hourly = apply_tmo_guardrail(
            df_hourly, tmo_guardrail_data,
            col_tmo_future="tmo_s",
            k=TMO_CAP_K,
            p95_factor=TMO_CAP_P95_FACTOR
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
