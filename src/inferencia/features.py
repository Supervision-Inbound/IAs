# src/inferencia/features.py
import pandas as pd
import numpy as np

TIMEZONE = "America/Santiago"

def ensure_ts(df: pd.DataFrame, tz: str = TIMEZONE) -> pd.DataFrame:
    """
    Construye columna temporal 'ts' a partir de:
      A) 'ts' si existe (se parsea y normaliza a TZ)
      B) 'fecha' + 'hora' (robusto, dayfirst=True y HORA estricta a HH:MM, formato fijo)
      C) 'datatime'/'datetime' como fallback
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
    
    # B) 'fecha' + 'hora' (estilo original)
    elif any(h in cols for h in ['fecha','date']) and any(h in cols for h in ['hora','hour']):
        fecha_col = col('fecha') if has('fecha') else col('date')
        hora_col = col('hora') if has('hora') else col('hour')
        
        # Forzar hora a HH:MM:SS
        hora_s = pd.to_datetime(d[hora_col], format='%H:%M:%S', errors='coerce').dt.strftime('%H:%M:%S')
        # Robustez por si viene como HH:MM
        if hora_s.isna().any():
             hora_s = pd.to_datetime(d[hora_col], format='%H:%M', errors='coerce').dt.strftime('%H:%M:00')
        
        # Combinar
        ts_str = d[fecha_col].astype(str) + " " + hora_s.astype(str)
        ts = pd.to_datetime(ts_str, errors='coerce', dayfirst=True)
    
    # C) 'datatime'/'datetime'
    elif any(h in cols for h in ['datatime','datetime']):
        dt_col = col('datatime') if has('datatime') else col('datetime')
        ts = pd.to_datetime(d[dt_col], errors='coerce', dayfirst=True, infer_datetime_format=True)
    
    else:
        raise ValueError("No se encontraron columnas 'ts', ('fecha' y 'hora'), o 'datatime' en el CSV.")

    # Normalizar Timezone
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(tz, ambiguous='NaT', nonexistent='NaT')
    else:
        ts = ts.dt.tz_convert(tz)
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

def add_es_dia_de_pago(df: pd.DataFrame) -> pd.Series:
    """Devuelve una Serie booleana/int. Requiere un índice 'ts' o una columna 'day'."""
    if 'day' in df.columns:
        day_col = df['day']
    elif isinstance(df.index, pd.DatetimeIndex):
        day_col = df.index.day
    else:
        raise ValueError("El DataFrame debe tener columna 'day' o un DatetimeIndex.")
    
    return day_col.isin([1,2,15,16,29,30,31]).astype(int)


def add_lags_mas(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    <--- MODIFICADO: Replicamos las features del script de entrenamiento v8 ---
    Genera lags, MAs, EMAs, STDs y MAXs sobre la columna 'target_col'.
    IMPORTANTE: Esta 'target_col' será 'recibidos_nacional' tanto para
    el planner COMO para el nuevo modelo TMO.
    """
    d = df.copy()
    
    # 1. Lags
    # Usamos los lags del training script (1,2,3,6,12,24,48,72,168)
    lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]
    for lag in lags:
        d[f"lag_{target_col}_{lag}"] = d[target_col].shift(lag)

    # 2. Rolling (MA, STD, MAX)
    # Usamos el shift(1) para evitar data leakage en rolling features
    target_shift1 = d[target_col].shift(1)
    
    # MAs (ventanas 6, 12, 24, 72, 168)
    ma_windows = [6, 12, 24, 72, 168]
    for w in ma_windows:
        d[f"ma_{target_col}_{w}"] = target_shift1.rolling(w, min_periods=1).mean()

    # STD y MAX (ventanas 24, 72)
    vol_windows = [24, 72]
    for w in vol_windows:
        d[f"std_{target_col}_{w}"] = target_shift1.rolling(w, min_periods=2).std()
        d[f"max_{target_col}_{w}"] = target_shift1.rolling(w, min_periods=1).max()

    # 3. EMAs (spans 6, 12, 24)
    ema_spans = [6, 12, 24]
    for span in ema_spans:
        d[f"ema_{target_col}_{span}"] = target_shift1.ewm(span=span, adjust=False, min_periods=1).mean()
    
    # Rellenar NaNs generados por rolling/lags (importante para inferencia)
    # Rellenamos con ffill (para propagar el último valor conocido)
    # y luego bfill (para rellenar el inicio si es necesario)
    cols_features = [c for c in d.columns if c.startswith(('lag_', 'ma_', 'std_', 'max_', 'ema_'))]
    d[cols_features] = d[cols_features].ffill().bfill()
    
    return d


def dummies_and_reindex(df: pd.DataFrame, training_columns: list) -> (pd.DataFrame, pd.DataFrame):
    """
    Aplica get_dummies y reindexa para que coincida con el entrenamiento.
    Devuelve (df_procesado, df_faltantes)
    """
    d = df.copy()
    # OHE para features categóricas (dow, month, hour)
    cat_cols = [c for c in ['dow', 'month', 'hour'] if c in d.columns]
    if cat_cols:
        d = pd.get_dummies(d, columns=cat_cols, drop_first=False)
    
    # Reindexar
    missing_in_df = set(training_columns) - set(d.columns)
    for c in missing_in_df:
        d[c] = 0 # Columnas que estaban en train pero no en este df (ej. hour=3)
    
    extra_in_df = set(d.columns) - set(training_columns)
    df_extra = d[list(extra_in_df)].copy() # Columnas en este df pero no en train
    d = d[training_columns] # Ordenar y filtrar
    
    return d, df_extra
