# src/inferencia/features.py
import pandas as pd
import numpy as np

TIMEZONE = "America/Santiago"

# ---------------- TS helpers ----------------
def _coerce_ts_series(s: pd.Series) -> pd.Series:
    """
    Convierte una serie a datetime con zona horaria consistente.
    - Acepta strings tipo 'YYYY-mm-dd HH:MM:SS%z' o similares.
    - Fuerza a UTC y luego convierte a TIMEZONE.
    """
    ts = pd.to_datetime(s, errors='coerce', dayfirst=True, utc=True)
    # Si ya viene con tz "mixta", el utc=True igual lo normaliza.
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    return ts.dt.tz_convert(TIMEZONE)

def ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garantiza un índice datetime llamado 'ts'. Reglas:
    1) Si ya existe columna 'ts': la convierte y la usa como índice.
    2) Si existen columnas 'fecha' y 'hora' (insensibles a mayúsculas): combina ambas.
    3) Si el índice ya es datetime: lo usa.
    """
    d = df.copy()
    cols_lower = {c.lower(): c for c in d.columns}

    if 'ts' in d.columns:
        d['ts'] = _coerce_ts_series(d['ts'].astype(str))
        d = d.dropna(subset=['ts']).sort_values('ts').set_index('ts')
        return d

    # Buscar fecha/hora
    fecha_col = None
    hora_col = None
    for c in d.columns:
        cl = c.lower()
        if 'fecha' == cl and fecha_col is None:
            fecha_col = c
        if cl == 'hora' and hora_col is None:
            hora_col = c

    if fecha_col is not None and hora_col is not None:
        s = d[fecha_col].astype(str).str.strip() + " " + d[hora_col].astype(str).str.strip()
        ts = _coerce_ts_series(s)
        d['ts'] = ts
        d = d.dropna(subset=['ts']).sort_values('ts').set_index('ts')
        return d

    # Si el índice es datetime, lo normalizo
    if isinstance(d.index, pd.DatetimeIndex):
        idx = pd.to_datetime(d.index, errors='coerce', utc=True)
        idx = idx.tz_convert(TIMEZONE)
        d.index = idx
        d = d.sort_index()
        d.index.name = 'ts'
        return d

    # Último intento: hay una sola columna que parece datetime
    for c in d.columns:
        try_ts = pd.to_datetime(d[c], errors='coerce', utc=True)
        if try_ts.notna().sum() > len(d) * 0.8:
            d['ts'] = try_ts.dt.tz_convert(TIMEZONE)
            d = d.dropna(subset=['ts']).sort_values('ts').set_index('ts')
            return d

    raise ValueError("No se pudo construir 'ts'. Aporta 'ts' o 'fecha'+'hora'.")

# ---------------- Time parts / dummies ----------------
def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega columnas: dow, month, hour, day, sin/cos hora y dow, y flag es_dia_de_pago si no existe.
    """
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        raise ValueError("add_time_parts requiere un índice datetime o una columna 'ts' ya indexada.")

    d["dow"]   = d.index.weekday
    d["month"] = d.index.month
    d["hour"]  = d.index.hour
    d["day"]   = d.index.day
    d["sin_hour"] = np.sin(2*np.pi*d["hour"]/24.0)
    d["cos_hour"] = np.cos(2*np.pi*d["hour"]/24.0)
    d["sin_dow"]  = np.sin(2*np.pi*d["dow"]/7.0)
    d["cos_dow"]  = np.cos(2*np.pi*d["dow"]/7.0)
    if "es_dia_de_pago" not in d.columns:
        d["es_dia_de_pago"] = d["day"].isin([1,2,15,16,29,30,31]).astype(int)
    return d

def dummies_and_reindex(df_last_row: pd.DataFrame, training_cols: list) -> pd.DataFrame:
    """
    Crea dummies solo de ['dow','month','hour'] si alguna de esas aparece en training_cols,
    y devuelve un DataFrame UNA FILA con exactamente training_cols en el mismo orden.
    """
    d = df_last_row.copy()
    base_cols = ["dow", "month", "hour"]
    need_dum = any([(c.startswith("dow_") or c.startswith("month_") or c.startswith("hour_")) for c in training_cols])

    if need_dum:
        for c in base_cols:
            if c not in d.columns:
                # rellena con la última hora del índice
                tmp = add_time_parts(pd.DataFrame(index=d.index))
                d[c] = tmp[c]
        d = pd.get_dummies(d, columns=[c for c in base_cols if c in d.columns], drop_first=False)

    # numérico y saneo
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.ffill().fillna(0)

    # reindex exacto
    out = d.reindex(columns=training_cols, fill_value=0).tail(1)
    return out

# ---------------- (Compat) lags/MA genéricos ----------------
def add_lags_mas(s: pd.Series, lags: list[int] = None, mas: list[int] = None, prefix: str = "") -> pd.DataFrame:
    """
    Pequeña utilidad para generar lags y medias móviles de una serie.
    Devuelve DataFrame con columnas lag_{k} y ma_{w} (con prefijo opcional).
    """
    lags = lags or []
    mas = mas or []
    d = pd.DataFrame(index=s.index)
    for k in lags:
        d[f"{prefix}lag_{k}"] = s.shift(k)
    for w in mas:
        d[f"{prefix}ma_{w}"] = s.rolling(w, min_periods=1).mean()
    return d

