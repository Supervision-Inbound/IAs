import re
import numpy as np
import pandas as pd

TIMEZONE = "America/Santiago"

# ------------------------------------------------------------
# Utils de parseo temporal
# ------------------------------------------------------------
def _coerce_ts_series(s: pd.Series) -> pd.Series:
    """Parsea una serie de strings a datetime CONSCIENTE (UTC inicial) y la convierte a TIMEZONE."""
    if s.dtype == "datetime64[ns]" or np.issubdtype(s.dtype, np.datetime64):
        dt = pd.to_datetime(s, errors="coerce", utc=True)
    else:
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True, utc=True)
        if dt.isna().any():
            dt2 = pd.to_datetime(s, errors="coerce", dayfirst=False, utc=True)
            dt = dt.fillna(dt2)
    
    # Convierte a la zona horaria final.
    return dt.dt.tz_convert(TIMEZONE)

def ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura un índice DatetimeIndex redondeado, tz-aware y sin duplicados."""
    d = df.copy()
    
    # Caso 1: El índice ya es DatetimeIndex
    if isinstance(d.index, pd.DatetimeIndex):
        idx = d.index
        # 1. Convertir a naive (UTC-based)
        if idx.tz is not None:
             idx_naive = idx.tz_convert('UTC').tz_localize(None) 
        else:
             idx_naive = idx
        
        # 2. Redondear el tiempo ingenuo a la hora más cercana
        idx_naive_rounded = idx_naive.round('h') 
        
        # 3. Localizar a UTC y CONVERTIR a TIMEZONE
        idx_aware = idx_naive_rounded.tz_localize('UTC').tz_convert(TIMEZONE)
        
        d.index = idx_aware
        if "ts" in d.columns: d = d.drop(columns=["ts"])
        d = d.sort_index()
        d.index.name = "ts"
        # Limpieza de duplicados en el índice
        d = d[~d.index.duplicated(keep='last')] 
        return d

    # Caso 2: El índice no es DatetimeIndex (Construir desde columnas)
    cols = {c.lower().strip(): c for c in d.columns}
    ts_col = None
    for cand in ["ts"]:
        if cand in cols: ts_col = cols[cand]; break
    fecha_col = None
    hora_col = None
    if ts_col is None:
        for cand in ["fecha", "date", "dia", "día"]:
            if cand in cols: fecha_col = cols[cand]; break
        for cand in ["hora", "hour"]:
            if cand in cols: hora_col = cols[cand]; break

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
    
    # 1. Convertir 'ts' a naive (UTC-based)
    ts_naive = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    
    # 2. Redondear (naive)
    ts_naive_rounded = ts_naive.round('h')
    
    # 3. Localizar a UTC y CONVERTIR a TIMEZONE
    ts_rounded_aware = ts_naive_rounded.dt.tz_localize('UTC').dt.tz_convert(TIMEZONE)
    
    # Establecer índice y limpiar
    d = d.dropna(subset=["ts"]).sort_values("ts").set_index(ts_rounded_aware) 
    # Limpieza de duplicados en el índice
    d = d[~d.index.duplicated(keep='last')] 
    return d

# ------------------------------------------------------------
# Partes de tiempo (Corregido el manejo de TZ en caso de naive)
# ------------------------------------------------------------
def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "ts" in df.columns:
            df = df.set_index(pd.to_datetime(df["ts"], errors="coerce", utc=True)).drop(columns=["ts"], errors="ignore")
        else:
            raise ValueError("add_time_parts requiere un índice datetime o una columna 'ts'.")

    d = df.copy()
    
    idx = d.index
    if idx.tz is None:
        try: 
            # Asumir UTC y convertir
            idx = idx.tz_localize('UTC').tz_convert(TIMEZONE)
        except Exception: 
            # Manejo de DST en caso de ambigüedad
            idx = idx.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward').tz_convert(TIMEZONE)
    else:
        idx = idx.tz_convert(TIMEZONE)

    d["dow"] = idx.weekday
    d["month"] = idx.month
    d["hour"] = idx.hour
    d["day"] = idx.day

    d["sin_hour"] = np.sin(2*np.pi*d["hour"]/24.0)
    d["cos_hour"] = np.cos(2*np.pi*d["hour"]/24.0)
    d["sin_dow"] = np.sin(2*np.pi*d["dow"]/7.0)
    d["cos_dow"] = np.cos(2*np.pi*d["dow"]/7.0)
    
    return d

# ------------------------------------------------------------
# Features de lags y medias móviles (sin cambios)
# ------------------------------------------------------------
def add_lags_mas(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    d = df.copy()
    if target_col not in d.columns: d[target_col] = 0.0
    s = pd.to_numeric(d[target_col], errors="coerce")
    for k in [24, 48, 72, 168]: d[f"lag_{k}"] = s.shift(k)
    s1 = s.shift(1)
    for w in [24, 72, 168]: d[f"ma_{w}"] = s1.rolling(w, min_periods=1).mean()
    for c in [f"lag_{k}" for k in [24,48,72,168]] + [f"ma_{w}" for w in [24,72,168]]: d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

# ------------------------------------------------------------
# Dummies + reindex (Corregida la lista de columnas duplicadas)
# ------------------------------------------------------------
def dummies_and_reindex(df: pd.DataFrame, training_cols: list) -> pd.DataFrame:
    d = df.copy()
    cat_cols = []
    for c in ["dow", "month", "hour"]:
        if c in d.columns: cat_cols.append(c)
    if cat_cols: d = pd.get_dummies(d, columns=cat_cols, drop_first=False)
    for c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce")
    
    # CORRECCIÓN: Eliminar duplicados de la lista de columnas esperadas
    unique_training_cols = list(dict.fromkeys(training_cols))
    
    X = d.reindex(columns=unique_training_cols, fill_value=0.0)
    X = X.ffill().fillna(0.0)
    return X
