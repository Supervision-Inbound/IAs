# src/inferencia/features.py
import pandas as pd
import numpy as np

TIMEZONE = "America/Santiago"

def ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye columna temporal 'ts' a partir de:
      A) 'ts' si existe (se parsea y normaliza a TZ)
      B) 'fecha' + 'hora' (robusto, dayfirst=True y HORA estricta a HH:MM, formato fijo)
      C) 'datatime'/'datetime' como fallback
    - No mira ningún otro archivo (solo el CSV histórico).
    - Normaliza TZ a America/Santiago.
    - Devuelve df.set_index('ts').sort_index()
    """
    d = df.copy()
    cols = {c.lower().strip(): c for c in d.columns}

    def has(name): return name in cols
    def col(name): return cols[name]

    # A) 'ts' directo
    if has('ts'):
        ts = pd.to_datetime(d[col('ts')], errors='coerce', dayfirst=True, infer_datetime_format=True)
        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT')
        else:
            ts = ts.dt.tz_convert(TIMEZONE)
        d['ts'] = ts

    # B) 'fecha' + 'hora' (estilo original)
    elif any(h in cols for h in ['fecha','date']) and any(h in cols for h in ['hora','hour','hora numero','hora_número','h']):
        fecha_col = next(cols[k] for k in ['fecha','date'] if k in cols)
        hora_col  = next(cols[k] for k in ['hora','hour','hora numero','hora_número','h'] if k in cols)

        # Fecha D/M/Y → dayfirst=True
        fecha_dt = pd.to_datetime(d[fecha_col], errors='coerce', dayfirst=True)

        # Hora estricta HH:MM (corta basura tipo "8", "8.0", "8:0", etc.)
        hora_str = d[hora_col].astype(str).str.strip().str.replace('.', ':', regex=False)
        hora_str = hora_str.str.slice(0, 5)  # HH:MM

        # Formato fijo evita ambigüedades
        ts = pd.to_datetime(fecha_dt.astype(str) + " " + hora_str,
                            errors='coerce', format="%Y-%m-%d %H:%M")
        ts = ts.dt.tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT')
        d['ts'] = ts

    # C) 'datatime'/'datetime' fallback
    else:
        dt_col = None
        for k in ['datatime', 'datetime', 'fecha_hora', 'fecha y hora']:
            if k in cols:
                dt_col = cols[k]
                break
        if dt_col is None:
            raise ValueError("Se requiere 'ts' o ('fecha' + 'hora') o 'datatime' en el CSV.")
        ts = pd.to_datetime(d[dt_col], errors='coerce', dayfirst=True, infer_datetime_format=True)
        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT')
        else:
            ts = ts.dt.tz_convert(TIMEZONE)
        d['ts'] = ts

    d = d.dropna(subset=['ts']).sort_values('ts').set_index('ts')
    return d


def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    idx = d.index
    d['dow'] = idx.dayofweek
    d['month'] = idx.month
    d['hour'] = idx.hour
    d['day'] = idx.day
    d['sin_hour'] = np.sin(2 * np.pi * d['hour'] / 24)
    d['cos_hour'] = np.cos(2 * np.pi * d['hour'] / 24)
    d['sin_dow']  = np.sin(2 * np.pi * d['dow'] / 7)
    d['cos_dow']  = np.cos(2 * np.pi * d['dow'] / 7)
    return d


def add_lags_mas(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    d = df.copy()
    for lag in [24, 48, 72, 168]:
        d[f'lag_{lag}'] = d[target_col].shift(lag)
    for window in [24, 72, 168]:
        d[f'ma_{window}'] = d[target_col].rolling(window, min_periods=1).mean()
    return d


def dummies_and_reindex(df_row: pd.DataFrame, training_cols: list) -> pd.DataFrame:
    d = df_row.copy()
    d = pd.get_dummies(d, columns=['dow', 'month', 'hour'], drop_first=False)
    # ensure training columns
    for c in training_cols:
        if c not in d.columns:
            d[c] = 0
    return d.reindex(columns=training_cols, fill_value=0)

