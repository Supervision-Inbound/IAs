# src/inferencia/features.py
import pandas as pd
import numpy as np

TIMEZONE = "America/Santiago"

def _coerce_ts_series(s: pd.Series) -> pd.DatetimeIndex:
    """Parsea una serie a datetime con TZ America/Santiago, tolerante a formatos."""
    # Pandas >=2 ya ignora infer_datetime_format; evitamos ese arg.
    ts = pd.to_datetime(s, errors='coerce', dayfirst=True)
    # Si viene ya tz-aware, convertimos; si no, localizamos.
    if getattr(ts.dt, "tz", None) is not None:
        return ts.dt.tz_convert(TIMEZONE)
    return ts.dt.tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT')

def ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye/normaliza 'ts' y devuelve un DataFrame con:
      - Índice = 'ts' (tz-aware America/Santiago)
      - SIN columna 'ts' duplicada
      - Ordenado por 'ts'
    Reglas:
      1) Si ya hay índice 'ts' o columna 'ts', se respeta y se normaliza.
      2) Si no hay 'ts', se intenta 'fecha' + 'hora'.
      3) Fallback: 'datetime'/'datatime'.
    """
    d = df.copy()

    # Caso A: ya existe índice con nombre 'ts' -> convertir a columna temporal para normalizar
    if isinstance(d.index, pd.DatetimeIndex) and d.index.name == 'ts':
        d = d.reset_index()

    # Caso B: si existe columna 'ts', normalizarla
    cols = {c.strip().lower(): c for c in d.columns}
    ts_col = cols.get('ts')

    if ts_col is not None:
        ts = _coerce_ts_series(d[ts_col].astype(str))
        d['__ts__'] = ts
    else:
        # Caso C: 'fecha' + 'hora'
        fecha_key = next((k for k in ['fecha','date'] if k in cols), None)
        hora_key  = next((k for k in ['hora','hour','hora numero','hora_número','h'] if k in cols), None)

        if fecha_key and hora_key:
            fecha_col = cols[fecha_key]
            hora_col  = cols[hora_key]

            fecha_dt = pd.to_datetime(d[fecha_col], errors='coerce', dayfirst=True)

            # Normalizar hora a HH:MM
            hora_str = d[hora_col].astype(str).str.strip().str.replace('.', ':', regex=False)
            # Reparar formatos parciales tipo "8" -> "08:00", "8:0" -> "08:00"
            def _fix_hhmm(x: str) -> str:
                if ':' not in x:
                    # solo hora -> HH:MM
                    try:
                        h = int(float(x))
                        return f"{h:02d}:00"
                    except:
                        return x
                parts = x.split(':')
                try:
                    h = int(float(parts[0]))
                except:
                    return x
                m = 0
                if len(parts) > 1:
                    try:
                        m = int(float(parts[1]))
                    except:
                        m = 0
                return f"{h:02d}:{m:02d}"
            hora_str = hora_str.apply(_fix_hhmm)

            ts = pd.to_datetime(
                fecha_dt.astype(str) + " " + hora_str,
                errors='coerce', format="%Y-%m-%d %H:%M"
            )
            ts = ts.dt.tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT')
            d['__ts__'] = ts
        else:
            # Caso D: fallback a 'datetime'/'datatime'
            dt_key = next((k for k in ['datetime','datatime','fecha_hora'] if k in cols), None)
            if not dt_key:
                raise ValueError("No se pudieron inferir columnas temporales. Se esperaba 'ts' o 'fecha'+'hora'.")
            dt_col = cols[dt_key]
            d['__ts__'] = _coerce_ts_series(d[dt_col].astype(str))

    # Limpieza: columna/índice 'ts' duplicados
    # Si existía una columna 'ts', la descartamos a favor de '__ts__'
    if ts_col is not None and 'ts' in d.columns:
        d = d.drop(columns=['ts'])

    # Construir índice final
    d = d.dropna(subset=['__ts__']).rename(columns={'__ts__':'ts'})
    # Evitar ambigüedad: si por alguna razón quedó una columna 'ts' y también la usaremos de índice, elimínala
    if 'ts' in d.columns and 'ts' in getattr(d.index, 'names', []):
        d = d.drop(columns=['ts'])
    # Orden y set_index seguro
    d = d.sort_values('ts').set_index('ts')
    # Asegurar que NO haya una columna 'ts' residual
    if 'ts' in d.columns:
        d = d.drop(columns=['ts'])

    # Normalizar tipo para downstream
    d.index = pd.DatetimeIndex(d.index).tz_convert(TIMEZONE)
    return d

def add_time_parts(df_or_indexed: pd.DataFrame) -> pd.DataFrame:
    """Agrega dow, month, hour, day y senoidales. Soporta que 'ts' sea índice o columna."""
    d = df_or_indexed.copy()
    if 'ts' in d.columns:
        idx = _coerce_ts_series(d['ts'].astype(str))
    elif isinstance(d.index, pd.DatetimeIndex):
        idx = d.index
    else:
        raise ValueError("add_time_parts requiere un índice datetime o una columna 'ts'.")

    d['dow']   = idx.dayofweek
    d['month'] = idx.month
    d['hour']  = idx.hour
    d['day']   = idx.day

    d['sin_hour'] = np.sin(2*np.pi*d['hour']/24.0)
    d['cos_hour'] = np.cos(2*np.pi*d['hour']/24.0)
    d['sin_dow']  = np.sin(2*np.pi*d['dow']/7.0)
    d['cos_dow']  = np.cos(2*np.pi*d['dow']/7.0)
    return d

def add_lags_mas(df_in: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Lags (1..168) y medias móviles (24,72,168) sobre target_col."""
    d = df_in.copy()
    d[target_col] = pd.to_numeric(d[target_col], errors='coerce')
    for lag in [1,2,3,6,12,24,48,72,168]:
        d[f'lag_{target_col}_{lag}'] = d[target_col].shift(lag)
    for window in [24, 72, 168]:
        d[f'ma_{window}'] = d[target_col].rolling(window, min_periods=1).mean()
    return d

def dummies_and_reindex(df_row: pd.DataFrame, training_cols: list) -> pd.DataFrame:
    d = df_row.copy()
    d = pd.get_dummies(d, columns=['dow', 'month', 'hour'], drop_first=False)
    # asegurar columnas del entrenamiento
    for c in training_cols:
        if c not in d.columns:
            d[c] = 0
    return d.reindex(columns=training_cols, fill_value=0)


