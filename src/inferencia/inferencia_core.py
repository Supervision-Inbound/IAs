# =========================================================================
# src/inferencia/inferencia_core.py (¡Actualizado para v10.2!)
# - AJUSTE: Nombres de columnas (TARGET_CALLS, TARGET_TMO)
# - AJUSTE: Eliminados CONTEXT_FEATURES (v8)
# - AJUSTE: Añadidos features de Volatilidad (v10.2) en el Paso 2
# - SIN CAMBIOS: La lógica del Paso 1 (Planner) sigue intacta.
# =========================================================================
import os
import glob
import pathlib
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

# ---------- Resolver flexible (Sin cambios) ----------
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

# ---------- Planner (dejamos rutas por defecto) ----------
PLANNER_MODEL = "models/modelo_planner.keras"
PLANNER_SCALER = "models/scaler_planner.pkl"
PLANNER_COLS   = "models/training_columns_planner.json"

# ---------- TMO (usamos lo que encuentre el release) ----------
TMO_MODEL    = _paths.get("keras")    or "models/modelo_tmo.keras"
TMO_SCALER   = _paths.get("scaler")   or "models/scaler_tmo.pkl"
TMO_COLS     = _paths.get("cols")     or "models/training_columns_tmo.json"
# (Lógica residual eliminada)

# ---------- Negocio ----------
# <-- AJUSTE v10.2: Alineado con los nombres de tu 'Hosting ia.xlsx'
TARGET_CALLS = "recibidos"
TARGET_TMO   = "tmo (segundos)"
# --- FIN AJUSTE ---

HIST_WINDOW_DAYS = 90
ENABLE_OUTLIER_CAP = True
K_WEEKDAY = 6.0
K_WEEKEND = 7.0

# <-- AJUSTE v10.2: Eliminados features de contexto
CONTEXT_FEATURES = []
# --- FIN AJUSTE ---


# ---------- Utils feriados / outliers (Sin cambios) ----------
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

    dfh = dfh.copy()
    dfh["is_post_hol"] = (~dfh["is_holiday"]) & (dfh["is_holiday"].shift(1).fillna(False))
    med_post_calls = dfh[dfh["is_post_hol"]].groupby("hour")[col_calls].median()
    post_calls_by_hour = {
        int(h): _safe_ratio(med_post_calls.get(h, np.nan),
                            med_nor_calls.get(h, np.nan),
                            fallback=1.05)
        for h in range(24)
    }
    post_calls_by_hour = {h: float(np.clip(v, 0.90, 1.80)) for h, v in post_calls_by_hour.items()}
    return (factors_calls_by_hour, factors_tmo_by_hour, global_calls_factor, global_tmo_factor, post_calls_by_hour)

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

def apply_post_holiday_adjustment(df_future, holidays_set, post_calls_by_hour,
                                  col_calls_future="calls"):
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

# (Funciones residuales eliminadas)


# ---------- Núcleo ----------
def forecast_120d(df_hist_joined: pd.DataFrame, 
                  horizon_days: int = 120, holidays_set: set | None = None):
    """
    - Estrategia: Directa (v10.2) para TMO, igual que Planner.
    - Volumen iterativo (con planner).
    - TMO iterativo (con modelo directo v10.2).
    - AMBOS modelos reciben "pistas" del otro (Llamadas <-> TMO).
    """
    # Artefactos
    m_pl = tf.keras.models.load_model(PLANNER_MODEL, compile=False)
    sc_pl = joblib.load(PLANNER_SCALER)
    cols_pl = _load_cols(PLANNER_COLS)

    m_tmo = tf.keras.models.load_model(TMO_MODEL, compile=False)
    sc_tmo = joblib.load(TMO_SCALER)
    cols_tmo = _load_cols(TMO_COLS)

    # Base histórica única
    # Asumimos que ensure_ts maneja la carga de df_hist_joined
    df = ensure_ts(df_hist_joined)

    # --- GUARD: asegurar que el índice es DatetimeIndex TZ-aware ---
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
        df.index = df.index.tz_convert(TIMEZONE)
    elif df.index.tz is None:
        df.index = df.index.tz_localize(TIMEZONE)
    else:
        df.index = df.index.tz_convert(TIMEZONE)
    # ---------------------------------------------------------------

    # <-- AJUSTE v10.2: TARGET_CALLS ahora es "recibidos"
    if TARGET_CALLS not in df.columns:
        raise ValueError(f"Falta columna '{TARGET_CALLS}' en historical_data.csv")
    if "feriados" not in df.columns:
        df["feriados"] = 0
    if "es_dia_de_pago" not in df.columns:
        df["es_dia_de_pago"] = 0
    else:
        df["es_dia_de_pago"] = 0 # Forzado a 0

    # (Lógica residual eliminada)

    # <-- AJUSTE v10.2: CONTEXT_FEATURES ahora está vacía
    static_context_features = {}
    for c in CONTEXT_FEATURES: # Este bucle ahora no hará nada, lo cual es correcto
        if c not in df.columns:
            df[c] = np.nan
        else:
            last_val = df[c].ffill().iloc[-1]
            static_context_features[c] = float(last_val) if pd.notna(last_val) else 0.0

    # Ventana reciente
    keep_cols = [TARGET_CALLS, TARGET_TMO, "feriados", "es_dia_de_pago"] + CONTEXT_FEATURES
    keep_cols_exist = [c for c in keep_cols if c in df.columns]
    
    idx = df.index
    last_ts = pd.to_datetime(idx.max())
    mask_recent = idx >= (last_ts - pd.Timedelta(days=HIST_WINDOW_DAYS))
    dfp = df.loc[mask_recent, keep_cols_exist].copy()
    if dfp.empty:
        dfp = df.loc[:, keep_cols_exist].copy()

    # ffill inicial de datos históricos
    dfp[TARGET_CALLS] = pd.to_numeric(dfp[TARGET_CALLS], errors="coerce").ffill().fillna(0.0)
    dfp[TARGET_TMO] = pd.to_numeric(dfp[TARGET_TMO], errors="coerce").ffill().fillna(0.0)
    
    static_context_vals = {}
    for c in CONTEXT_FEATURES: # Este bucle tampoco hará nada
        if c in dfp.columns:
            dfp[c] = pd.to_numeric(dfp[c], errors="coerce").ffill()
            last_val = dfp[c].iloc[-1]
            static_context_vals[c] = float(last_val) if pd.notna(last_val) else 0.0
        else:
            static_context_vals[c] = 0.0
    
    # Imprimirá un diccionario vacío, lo cual es correcto
    print(f"INFO: Usando valores de contexto estáticos: {static_context_vals}")

    # ===== Horizonte futuro =====
    future_ts = pd.date_range(
        last_ts + pd.Timedelta(hours=1),
        periods=horizon_days * 24,
        freq="h",
        tz=TIMEZONE
    )

    # ===== Bucle iterativo: Volumen + TMO (ambos Directos) =====
    print("Iniciando predicción iterativa (Llamadas Directas + TMO Directo)...")
    for ts in future_ts:
        
        # --- 1. PREDECIR LLAMADAS (PLANNER) ---
        #
        # (¡SIN CAMBIOS! Esta lógica es la original)
        #
        cols_planner = [TARGET_CALLS, TARGET_TMO, "feriados", "es_dia_de_pago"]
        cols_planner_exist = [c for c in cols_planner if c in dfp.columns]
        tmp_planner = dfp[cols_planner_exist].copy()
        tmp_planner = pd.concat([tmp_planner, pd.DataFrame(index=[ts])])
        tmp_planner[TARGET_CALLS] = tmp_planner[TARGET_CALLS].ffill()
        if TARGET_TMO in tmp_planner.columns:
            tmp_planner[TARGET_TMO] = tmp_planner[TARGET_TMO].ffill()
        if "feriados" in tmp_planner.columns:
            tmp_planner.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)
        if "es_dia_de_pago" in tmp_planner.columns:
            tmp_planner.loc[ts, "es_dia_de_pago"] = 0
        tmp_planner = add_lags_mas(tmp_planner, TARGET_CALLS) 
        if TARGET_TMO in tmp_planner.columns:
            for lag in [24, 48, 72, 168]:
                tmp_planner[f'tmo_lag_{lag}'] = tmp_planner[TARGET_TMO].shift(lag)
            for window in [24, 72, 168]:
                tmp_planner[f'tmo_ma_{window}'] = tmp_planner[TARGET_TMO].rolling(window, min_periods=1).mean()
        tmp_planner = add_time_parts(tmp_planner)
        X_pl = dummies_and_reindex(tmp_planner.tail(1), cols_pl)
        yhat_calls = float(m_pl.predict(sc_pl.transform(X_pl), verbose=0).flatten()[0])
        yhat_calls = max(0.0, yhat_calls)


        # --- 2. PREDECIR TMO (DIRECTO v10.2) ---
        
        # 2b. Crear Dataframe temporal para TMO
        cols_tmo_model = [
            TARGET_TMO,     # Pista 1 (TMO Total)
            TARGET_CALLS,   # Pista 2 (Volumen)
            "feriados", 
            "es_dia_de_pago"
        ]
        cols_tmo_model_exist = [c for c in cols_tmo_model if c in dfp.columns]
        tmp_tmo = dfp[cols_tmo_model_exist].copy()
        
        tmp_tmo = pd.concat([tmp_tmo, pd.DataFrame(index=[ts])])
        tmp_tmo[TARGET_TMO] = tmp_tmo[TARGET_TMO].ffill()
        tmp_tmo[TARGET_CALLS] = tmp_tmo[TARGET_CALLS].ffill()
        
        # Añadir features de contexto (estará vacío, correcto)
        for c, val in static_context_vals.items():
            tmp_tmo[c] = val 

        tmp_tmo.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)
        tmp_tmo.loc[ts, "es_dia_de_pago"] = 0 # Forzado a 0

        # <-- AJUSTE v10.2: Generar los mismos features que el entrenamiento
        # 2c. Crear Pistas de Lags/MAs (¡Debe coincidir con el Entrenamiento v10.2!)

        # Pista 1: TMO Total (Lags "rápidos" + Volatilidad)
        s_tmo_total = tmp_tmo[TARGET_TMO]
        for lag in [1, 2, 3, 6, 12, 24, 168]:
            tmp_tmo[f"lag_tmo_total_{lag}"] = s_tmo_total.shift(lag)
        s_tmo_total_s1 = s_tmo_total.shift(1)
        for w in [6, 12, 24, 72]:
            tmp_tmo[f"ma_tmo_total_{w}"] = s_tmo_total_s1.rolling(w, min_periods=1).mean()

        # --- INICIO AJUSTE v10.2: AÑADIR FEATURES DE VOLATILIDAD ---
        for w in [6, 12, 24]:
            tmp_tmo[f"std_tmo_total_{w}"] = s_tmo_total_s1.rolling(w, min_periods=2).std()

        tmp_tmo["roc_tmo_total_1"] = s_tmo_total_s1.diff(1)
        tmp_tmo["roc_tmo_total_6"] = s_tmo_total_s1.diff(6)
        
        if "ma_tmo_total_12" in tmp_tmo.columns:
            tmp_tmo["diff_ma_tmo_total_12"] = s_tmo_total_s1 - tmp_tmo["ma_tmo_total_12"]
        else:
            tmp_tmo["diff_ma_tmo_total_12"] = tmp_tmo["roc_tmo_total_1"] # Fallback
        # --- FIN AJUSTE v10.2 ---

        # Pista 2: Volumen (Lags "rápidos" - Alineado con "recibidos")
        s_contest = tmp_tmo[TARGET_CALLS] # Usa "recibidos" (TARGET_CALLS)
        for lag in [1, 24, 48, 168]:
             tmp_tmo[f"lag_contest_{lag}"] = s_contest.shift(lag)
        s_contest_s1 = s_contest.shift(1)
        for w in [6, 24, 72]:
            tmp_tmo[f"ma_contest_{w}"] = s_contest_s1.rolling(w, min_periods=1).mean()
        # --- FIN AJUSTES v10.2 ---

        # 2d. Crear Pistas de Tiempo
        tmp_tmo = add_time_parts(tmp_tmo)
        
        # 2e. Predecir TMO (Directo)
        X_tmo = dummies_and_reindex(tmp_tmo.tail(1), cols_tmo)
        yhat_tmo = float(m_tmo.predict(sc_tmo.transform(X_tmo), verbose=0).flatten()[0])
        yhat_tmo = max(0.0, yhat_tmo) # TMO no puede ser negativo


        # --- 3. ACTUALIZACIÓN ITERATIVA ---
        dfp.loc[ts, TARGET_CALLS] = yhat_calls
        dfp.loc[ts, TARGET_TMO] = yhat_tmo # <-- Guardar predicción directa
        
        dfp.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)
        dfp.loc[ts, "es_dia_de_pago"] = 0 # Forzado a 0
        
        for c, val in static_context_vals.items():
            dfp.loc[ts, c] = val


    print("Predicción iterativa completada.")

    # ===== Salida horaria =====
    df_hourly = pd.DataFrame(index=future_ts)
    df_hourly["calls"] = np.round(dfp.loc[future_ts, TARGET_CALLS]).astype(int)
    df_hourly["tmo_s"] = np.round(dfp.loc[future_ts, TARGET_TMO]).astype(int)

    # ===== Ajustes feriados / post-feriados =====
    if holidays_set and len(holidays_set) > 0:
        (f_calls_by_hour, f_tmo_by_hour, _gc, _gt, post_calls_by_hour) = compute_holiday_factors(
            df, holidays_set, col_calls=TARGET_CALLS, col_tmo=TARGET_TMO
        )
        df_hourly = apply_holiday_adjustment(df_hourly, holidays_set, f_calls_by_hour, f_tmo_by_hour,
                                             col_calls_future="calls", col_tmo_future="tmo_s")
        df_hourly = apply_post_holiday_adjustment(df_hourly, holidays_set, post_calls_by_hour,
                                                col_calls_future="calls")

    # ===== CAP outliers (llamadas) =====
    if ENABLE_OUTLIER_CAP:
        base_mad = _baseline_median_mad(df, col=TARGET_CALLS)
        df_hourly = apply_outlier_cap(df_hourly, base_mad, holidays_set,
                                      col_calls_future="calls",
                                      k_weekday=K_WEEKDAY, k_weekend=K_WEEKEND)

    # ===== Erlang =====
    df_hourly["agents_prod"] = 0
    for ts in df_hourly.index:
        a, _ = required_agents(float(df_hourly.at[ts, "calls"]), float(df_hourly.at[ts, "tmo_s"]))
        df_hourly.at[ts, "agents_prod"] = int(a)
    df_hourly["agents_sched"] = df_hourly["agents_prod"].apply(schedule_agents)

    # ===== Salidas JSON =====
    write_hourly_json(f"{PUBLIC_DIR}/prediccion_horaria.json", df_hourly, "calls", "tmo_s", "agents_sched")
    write_daily_json(f"{PUBLIC_DIR}/prediccion_diaria.json", df_hourly, "calls", "tmo_s",
                     weights_col="calls")

    return df_hourly
