# src/inferencia/inferencia_core.py (¡LÓGICA v7 - LSTM Multi-Step!)
import os
import glob
import pathlib
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from .features import ensure_ts, add_time_parts, dummies_and_reindex
from .erlang import required_agents, schedule_agents
from .utils_io import write_daily_json, write_hourly_json

TIMEZONE = "America/Santiago"
PUBLIC_DIR = "public"

# --- NUEVO: Constantes LSTM deben coincidir con el entrenamiento ---
LOOKBACK_STEPS = 168 # 7 días de historia
HORIZON_STEPS = 24   # 1 día de predicción

# --- Resolver flexible (Sin cambios) ---
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

# (Funciones _load_cols, _safe_ratio, _series_is_holiday, compute_holiday_factors, 
#  apply_holiday_adjustment, _baseline_median_mad, apply_outlier_cap, _is_holiday
#  son idénticas a la v5, pegadas por completitud)
def _load_cols(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

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
    cols = [col_calls]
    med_hol_tmo = med_nor_tmo = None
    if col_tmo in df_hist.columns:
        cols.append(col_tmo)
    dfh = add_time_parts(df_hist[cols].copy())
    dfh["is_holiday"] = _series_is_holiday(dfh.index, holidays_set)

    med_hol_calls = dfh[dfh["is_holiday"]].groupby("hour")[col_calls].median()
    med_nor_calls = dfh[~dfh["is_holiday"]].groupby("hour")[col_calls].median()

    if col_tmo in dfh.columns:
        med_hol_tmo = dfh[dfh["is_holiday"]].groupby("hour")[col_tmo].median()
        med_nor_tmo = dfh[~dfh["is_holiday"]].groupby("hour")[col_tmo].median()
        g_hol_tmo = dfh[dfh["is_holiday"]][col_tmo].median()
        g_nor_tmo = dfh[~dfh["is_holiday"]][col_tmo].median()
        global_tmo_factor = _safe_ratio(g_hol_tmo, g_nor_tmo, fallback=1.00)
    else:
        global_tmo_factor = 1.00

    g_hol_calls = dfh[dfh["is_holiday"]][col_calls].median()
    g_nor_calls = dfh[~dfh["is_holiday"]][col_calls].median()
    global_calls_factor = _safe_ratio(g_hol_calls, g_nor_calls, fallback=0.75)

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
    factors_tmo_by_hour   = {h: float(np.clip(v, 0.70, 1.50)) for h, v in factors_tmo_by_hour.items()}

    # Lógica post-feriado eliminada del 'factor'
    return (factors_calls_by_hour, factors_tmo_by_hour, global_calls_factor, global_tmo_factor)

def apply_holiday_adjustment(df_future, holidays_set,
                             factors_calls_by_hour, factors_tmo_by_hour,
                             col_calls_future="calls", col_tmo_future="tmo_s"):
    d = add_time_parts(df_future.copy())
    is_hol = _series_is_holiday(d.index, holidays_set)
    hours = d["hour"].astype(int).values
    call_f = np.array([factors_calls_by_hour.get(int(h), 1.0) for h in hours])
    tmo_f  = np.array([factors_tmo_by_hour.get(int(h), 1.0) for h in hours])
    out = df_future.copy()
    mask = is_hol.values
    out.loc[mask, col_calls_future] = np.round(out.loc[mask, col_calls_future].astype(float) * call_f[mask]).astype(int)
    out.loc[mask, col_tmo_future]   = np.round(out.loc[mask, col_tmo_future].astype(float)   * tmo_f[mask]).astype(int)
    return out

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

def _is_holiday(ts, holidays_set: set) -> int:
    if not holidays_set:
        return 0
    try:
        d = ts.tz_convert(TIMEZONE).date()
    except Exception:
        d = ts.date()
    return 1 if d in holidays_set else 0

# --- NUEVO: Función para generar features en un dataframe
# Esta función replica la lógica de features del entrenamiento
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

# ---------- Núcleo (MODIFICADO PARA LSTM) ----------
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
    
    # Generar *todos* los features en el histórico
    dfp = generate_features(df, TARGET_CALLS, TARGET_TMO, "feriados")
    
    # ffill inicial (para lags y MAs al inicio)
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
        input_features_pl = pd.get_dummies(input_df[cols_pl], columns=['dow', 'month', 'hour'])
        input_features_pl = input_features_pl.reindex(columns=cols_pl, fill_value=0.0)
        input_scaled_pl = sc_pl.transform(input_features_pl)
        input_lstm_pl = input_scaled_pl.reshape((1, LOOKBACK_STEPS, len(cols_pl)))
        
        # 3. Preparar Input del TMO
        input_features_tmo = pd.get_dummies(input_df[cols_tmo], columns=['dow', 'month', 'hour'])
        input_features_tmo = input_features_tmo.reindex(columns=cols_tmo, fill_value=0.0)
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
        df_future_day[TARGET_CALLS] = np.maximum(0, yhat_calls_24h) # No llamadas negativas
        df_future_day[TARGET_TMO] = np.maximum(0, yhat_tmo_24h)   # No TMO negativo
        
        # 6. Generar "Known Future Features" (Calendario) para este nuevo día
        df_future_day["ts"] = df_future_day.index
        df_future_day = add_time_parts(df_future_day)
        df_future_day["feriados"] = df_future_day.index.to_series().apply(lambda ts: _is_holiday(ts, holidays_set))
        
        # 7. Apendizar al histórico (dfp) para la *siguiente* iteración
        # Necesitamos generar los lags/MAs para este nuevo día
        dfp_with_future = pd.concat([dfp, df_future_day])
        dfp_with_future = generate_features(dfp_with_future, TARGET_CALLS, TARGET_TMO, "feriados")
        
        # Actualizar dfp para el siguiente bucle
        dfp = dfp_with_future
        
        # Guardar las predicciones de este día
        future_predictions.append(dfp_with_future.iloc[-HORIZON_STEPS:])

    print("Predicción iterativa completada.")

    # ===== Salida horaria =====
    df_hourly = pd.concat(future_predictions)
    
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
