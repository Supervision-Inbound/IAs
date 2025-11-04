# src/inferencia/inferencia_core.py (¡LÓGICA v8 - LSTM EN CRUDO!)
import os
import glob
import pathlib
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from .features import ensure_ts, dummies_and_reindex
from .erlang import required_agents, schedule_agents
from .utils_io import write_daily_json, write_hourly_json

TIMEZONE = "America/Santiago"
PUBLIC_DIR = "public"

# --- Constantes LSTM (deben coincidir con el entrenamiento) ---
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
FERIADOS_COL = "feriados"
ENABLE_OUTLIER_CAP = True
K_WEEKDAY = 6.0
K_WEEKEND = 7.0

# --- Utils feriados / outliers ---
def _load_cols(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _safe_ratio(num, den, fallback=1.0):
    num = float(num) if num is not None and not np.isnan(num) else np.nan
    den = float(den) if den is not None and not np.isnan(den) and den != 0 else np.nan
    if np.isnan(num) or np.isnan(den) or den == 0:
        return fallback
    return num / den

# --- MODIFICADO: add_time_parts (v7) ahora es local ---
def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    
    if isinstance(d.index, pd.DatetimeIndex):
        idx = d.index
    else:
        idx = pd.to_datetime(d.get('ts'), errors='coerce') 

    if idx is None or idx.isna().all():
        raise ValueError("Error en add_time_parts: No se pudo encontrar un índice de tiempo o columna 'ts' válida.")

    if isinstance(idx, pd.Series):
        d["dow"]   = idx.dt.dayofweek
        d["month"] = idx.dt.month
        d["hour"]  = idx.dt.hour
        d["day"]   = idx.dt.day
        d["days_in_month"] = idx.dt.days_in_month
    else: # Is a DatetimeIndex
        d["dow"]   = idx.dayofweek
        d["month"] = idx.month
        d["hour"]  = idx.hour
        d["day"]   = idx.day
        d["days_in_month"] = idx.days_in_month

    if d["hour"].isna().any():
        d["hour"] = d["hour"].fillna(method='ffill').fillna(method='bfill')
        if d["hour"].isna().any():
            raise ValueError("Error en add_time_parts: 'hour' no se pudo calcular.")
        
    d["sin_hour"] = np.sin(2*np.pi*d["hour"]/24.0)
    d["cos_hour"] = np.cos(2*np.pi*d["hour"]/24.0)
    d["sin_dow"]  = np.sin(2*np.pi*d["dow"]/7.0)
    d["cos_dow"]  = np.cos(2*np.pi*d["dow"]/7.0)
    
    # Ciclo del día del mes
    d["sin_day"] = np.sin(2*np.pi*d["day"]/d["days_in_month"])
    d["cos_day"] = np.cos(2*np.pi*d["day"]/d["days_in_month"])
    
    if 'days_in_month' in d.columns:
        d = d.drop(columns=['days_in_month'])
    
    return d

def _series_is_holiday(idx, holidays_set: set) -> pd.Series:
    if not isinstance(idx, (pd.DatetimeIndex, pd.Series)):
        raise TypeError(f"Se esperaba DatetimeIndex o Series, pero se obtuvo {type(idx)}")

    idx_para_convertir = idx
    if isinstance(idx, pd.Series):
        idx_para_convertir = pd.to_datetime(idx, errors='coerce')

    try:
        idx_local = idx_para_convertir.tz_convert(TIMEZONE)
    except TypeError:
        idx_local = idx_para_convertir.tz_localize("UTC").tz_convert(TIMEZONE)
    
    idx_dates = idx_local.date
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
        
        g_hol_tmo = float(dfh[dfh["is_holiday"]][col_tmo].median())
        g_nor_tmo = float(dfh[~dfh["is_holiday"]][col_tmo].median())

        global_tmo_factor = _safe_ratio(g_hol_tmo, g_nor_tmo, fallback=1.00)
    else:
        global_tmo_factor = 1.00

    g_hol_calls = float(dfh[dfh["is_holiday"]][col_calls].median())
    g_nor_tmo = float(dfh[~dfh["is_holiday"]][col_calls].median())
    global_calls_factor = _safe_ratio(g_hol_calls, g_nor_tmo, fallback=0.75)

    factors_calls_by_hour = {
        int(h): _safe_ratio(med_hol_calls.get(h, np.nan),
                            med_nor_calls.get(h, np.nan),
                            fallback=global_calls_factor)
        for h in range(24)
    }
    if med_hol_tmo is not None and med_nor_tmo is not None:
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
    capped.loc[mask, col_calls_futu
