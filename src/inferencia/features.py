# src/inferencia/features.py
import re
import numpy as np
import pandas as pd

TIMEZONE = "America/Santiago"

# ------------------------------------------------------------
# Utils de parseo temporal
# (Sin cambios)
# ------------------------------------------------------------
def _coerce_ts_series(s: pd.Series) -> pd.Series:
    if s.dtype == "datetime64[ns]" or np.issubdtype(s.dtype, np.datetime64):
        dt = pd.to_datetime(s, errors="coerce", utc=True)
    else:
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True, utc=True)
        if dt.isna().any():
            dt2 = pd.to_datetime(s, errors="coerce", dayfirst=False, utc=True)
            dt = dt.fillna(dt2)
    try:
        dt = dt.dt.tz_convert(TIMEZONE)
    except Exception:
        dt = dt.dt.tz_localize("UTC").dt.tz_convert(TIMEZONE)
    return dt

def ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if isinstance(d.index, pd.DatetimeIndex):
        try:
            idx = d.index.tz_convert(TIMEZONE)
        except Exception:
            idx = d.index.tz_localize("UTC").tz_convert(TIMEZONE)
        d.index = idx
        if "ts" in d.columns:
            d = d.drop(columns=["ts"])
        d = d.sort_index()
        d.index.name = "ts"
        return d

    cols = {c.lower().strip(): c for c in d.columns}
    ts_col = None
    for cand in ["ts"]:
        if cand in cols:
            ts_col = cols[cand]; break
    fecha_col = None
    hora_col = None
    if ts_col is None:
        for cand in ["fecha", "date", "dia", "día"]:
            if cand in cols:
                fecha_col = cols[cand]; break
        for cand in ["hora", "hour"]:
            if cand in cols:
                hora_col = cols[cand]; break

    if ts_col is not None:
        ts = _coerce_ts_series(d[ts_col].astype(str))
    elif fecha_col is not None and hora_col is not None:
        s = (d[fecha_col].astype(str).str.strip() + " " + d[hora_col].astype(str).str.strip()).str.strip()
        ts = _coerce_ts_series(s)
    else:
        raise ValueError("No se pudo construir 'ts'. Aporta 'ts' o 'fecha'+'hora'.")

    d["ts"] = ts
    if isinstance(d.index, pd.MultiIndex) and "ts" in d.index.names:
        d.index = d.index.droplevel(d.index.names.index("ts"))
    d = d.dropna(subset=["ts"]).sort_values("ts").set_index("ts")
    return d

# ------------------------------------------------------------
# Partes de tiempo
# ------------------------------------------------------------
# --- MODIFICADO: add_time_parts con .dt y sin_day/cos_day ---
def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    
    # 1. Get the datetime object, whether it's index or column
    if isinstance(d.index, pd.DatetimeIndex):
        idx = d.index
    else:
        # Asumimos que si no es índice, debe existir la columna 'ts'
        idx = pd.to_datetime(d['ts'], errors='coerce')

    # 2. Get properties. DatetimeIndex has them directly,
    #    Series has them under .dt
    if isinstance(idx, pd.Series):
        d["dow"]   = idx.dt.dayofweek
        d["month"] = idx.dt.month
        d["hour"]  = idx.dt.hour
        d["day"]   = idx.dt.day
        # <-- NUEVO: Obtenemos el número de días en el mes para normalizar
        d["days_in_month"] = idx.dt.days_in_month
    else: # Is a DatetimeIndex
        d["dow"]   = idx.dayofweek
        d["month"] = idx.month
        d["hour"]  = idx.hour
        d["day"]   = idx.day
        # <-- NUEVO: Obtenemos el número de días en el mes para normalizar
        d["days_in_month"] = idx.days_in_month

    if d["hour"].isna().any():
        raise ValueError("Error en add_time_parts: 'hour' no se pudo calcular. Verifique el índice de tiempo.")
        
    d["sin_hour"] = np.sin(2*np.pi*d["hour"]/24.0)
    d["cos_hour"] = np.cos(2*np.pi*d["hour"]/24.0)
    d["sin_dow"]  = np.sin(2*np.pi*d["dow"]/7.0)
    d["cos_dow"]  = np.cos(2*np.pi*d["dow"]/7.0)
    
    # <-- NUEVO: Ciclo del día del mes
    d["sin_day"] = np.sin(2*np.pi*d["day"]/d["days_in_month"])
    d["cos_day"] = np.cos(2*np.pi*d["day"]/d["days_in_month"])
    
    # Limpiamos la columna auxiliar
    d = d.drop(columns=['days_in_month'])
    
    return d

# ------------------------------------------------------------
# Features de lags y medias móviles (Función v5, ya no se usa en v6)
# ------------------------------------------------------------
def add_lags_mas(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    d = df.copy()
    if target_col not in d.columns:
        d[target_col] = 0.0
    s = pd.to_numeric(d[target_col], errors="coerce")
    for k in [24, 48, 72, 168]:
        d[f"lag_{k}"] = s.shift(k)
    s1 = s.shift(1)
    for w in [24, 72, 168]:
        d[f"ma_{w}"] = s1.rolling(w, min_periods=1).mean()
    for c in [f"lag_{k}" for k in [24,48,72,168]] + [f"ma_{w}" for w in [24,72,168]]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

# ------------------------------------------------------------
# Dummies + reindex (Sin cambios, sigue siendo necesario)
# ------------------------------------------------------------
def dummies_and_reindex(df: pd.DataFrame, training_cols: list) -> pd.DataFrame:
    d = df.copy()

    cat_cols = []
    for c in ["dow", "month", "hour"]:
        if c in d.columns:
            cat_cols.append(c)

    if cat_cols:
        d = pd.get_dummies(d, columns=cat_cols, drop_first=False)

    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    X = d.reindex(columns=training_cols, fill_value=0.0)
    X = X.ffill().fillna(0.0)
    return X
