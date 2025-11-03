# src/inferencia/features.py
import re
import numpy as np
import pandas as pd

TIMEZONE = "America/Santiago"

# ------------------------------------------------------------
# Utils de parseo temporal (Sin cambios)
# ------------------------------------------------------------
def _coerce_ts_series(s: pd.Series) -> pd.Series:
    """Intenta parsear a datetime (UTC) de forma robusta y retorna tz-aware (America/Santiago)."""
    if s.dtype == "datetime64[ns]" or np.issubdtype(s.dtype, np.datetime64):
        dt = pd.to_datetime(s, errors="coerce", utc=True)
    else:
        # Intento 1: dayfirst
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True, utc=True)
        # Intento 2: si hay NaN, probar dayfirst=False
        if dt.isna().any():
            dt2 = pd.to_datetime(s, errors="coerce", dayfirst=False, utc=True)
            dt = dt.fillna(dt2)
    # Asegurar tz
    try:
        dt = dt.dt.tz_convert(TIMEZONE)
    except Exception:
        # Si no tiene tz (naive), lo localizo
        dt = dt.dt.tz_localize("UTC").dt.tz_convert(TIMEZONE)
    return dt

def ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura índice temporal 'ts' (tz-aware en America/Santiago), sin ambigüedad.
    Soporta:
      - DataFrame ya indexado por datetime (nombre del índice puede ser 'ts')
      - Columna 'ts'
      - Pareja 'fecha' + 'hora'
    """
    d = df.copy()

    # --- CASO A: ya tenemos un DatetimeIndex ---
    if isinstance(d.index, pd.DatetimeIndex):
        # Normalizar TZ
        try:
            idx = d.index.tz_convert(TIMEZONE)
        except Exception:
            idx = d.index.tz_localize("UTC").tz_convert(TIMEZONE)
        d.index = idx

        # Si también existe una columna 'ts', elimínala para evitar ambigüedad
        if "ts" in d.columns:
            d = d.drop(columns=["ts"])

        # Ordenar por índice y devolver
        d = d.sort_index()
        d.index.name = "ts"
        return d

    # --- CASO B: no es DatetimeIndex -> buscar columnas ---
    cols = {c.lower().strip(): c for c in d.columns}

    # B1) columna 'ts'
    ts_col = None
    for cand in ["ts"]:
        if cand in cols:
            ts_col = cols[cand]; break

    # B2) 'fecha' + 'hora'
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
        # Adaptado para Clima (puede no tener 'hora')
        date_col = next((c for c in cols if 'time' in c), None)
        if date_col:
             ts = _coerce_ts_series(d[date_col].astype(str))
        else:
            raise ValueError("No se pudo construir 'ts'. Aporta 'ts' o 'fecha'+'hora' o 'time'.")

    d["ts"] = ts
    if isinstance(d.index, pd.MultiIndex) and "ts" in d.index.names:
        d.index = d.index.droplevel(d.index.names.index("ts"))
    d = d.dropna(subset=["ts"]).sort_values("ts").set_index("ts")
    return d

# ------------------------------------------------------------
# Partes de tiempo (Sin cambios)
# ------------------------------------------------------------
def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Requiere índice datetime. Agrega dow, month, hour, day y sen/cos.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        if "ts" in df.columns:
            df = df.set_index(pd.to_datetime(df["ts"], errors="coerce", utc=True)).drop(columns=["ts"], errors="ignore")
        else:
            raise ValueError("add_time_parts requiere un índice datetime o una columna 'ts'.")

    d = df.copy()
    try:
        idx = d.index.tz_convert(TIMEZONE)
    except Exception:
        idx = d.index.tz_localize("UTC").tz_convert(TIMEZONE)

    d["dow"]   = idx.weekday
    d["month"] = idx.month
    d["hour"]  = idx.hour
    d["day"]   = idx.day

    d["sin_hour"] = np.sin(2*np.pi*d["hour"]/24.0)
    d["cos_hour"] = np.cos(2*np.pi*d["hour"]/24.0)
    d["sin_dow"]  = np.sin(2*np.pi*d["dow"]/7.0)
    d["cos_dow"]  = np.cos(2*np.pi*d["dow"]/7.0)

    if "es_dia_de_pago" not in d.columns:
        d["es_dia_de_pago"] = 0

    return d

# ------------------------------------------------------------
# Features de lags y medias móviles (Tu lógica v3 Original)
# ------------------------------------------------------------
def add_lags_mas(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Versión COMPATIBLE con el entrenamiento original del planner v3:
    - Lags con NOMBRES GENÉRICOS: lag_24, lag_48, lag_72, lag_168
    - MAs con NOMBRES GENÉRICOS:  ma_24,  ma_72,  ma_168
    """
    d = df.copy()
    if target_col not in d.columns:
        d[target_col] = 0.0
    s = pd.to_numeric(d[target_col], errors="coerce")

    # LAGS (genéricos)
    for k in [24, 48, 72, 168]:
        d[f"lag_{k}"] = s.shift(k)

    # Medias sobre la serie desplazada (shift(1))
    s1 = s.shift(1)
    for w in [24, 72, 168]:
        d[f"ma_{w}"] = s1.rolling(w, min_periods=1).mean()

    for c in [f"lag_{k}" for k in [24,48,72,168]] + [f"ma_{w}" for w in [24,72,168]]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    return d

# ------------------------------------------------------------
# Dummies + reindex contra training columns (¡CORREGIDO!)
# ------------------------------------------------------------
def dummies_and_reindex(df: pd.DataFrame, training_cols: list) -> pd.DataFrame:
    """
    Crea dummies para dow, month, hour (si existen) y reindexa EXACTAMENTE
    a las columnas de entrenamiento (llenando faltantes con 0).
    """
    d = df.copy()

    cat_cols = []
    for c in ["dow", "month", "hour"]:
        if c in d.columns:
            cat_cols.append(c)

    if cat_cols:
        d = pd.get_dummies(d, columns=cat_cols, drop_first=False)

    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Reindex exacto a columnas del entrenamiento
    X = d.reindex(columns=training_cols, fill_value=0.0)

    # --- INICIO DE LA CORRECCIÓN ---
    # Reemplazar infinitos (generados por el bucle) ANTES de rellenar NaNs.
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    # -----------------------------
    
    # Relleno forward y luego rellenar cualquier NaN restante con 0
    X = X.ffill().fillna(0.0)
    return X

