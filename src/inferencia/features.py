# src/inferencia/features.py
import pandas as pd
import numpy as np

TIMEZONE = "America/Santiago"

def ensure_ts(df: pd.DataFrame, tz: str = TIMEZONE) -> pd.DataFrame:
    """
    Construye columna temporal 'ts' a partir de 'fecha' + 'hora' o 'datatime'.
    Normaliza TZ a America/Santiago.
    Devuelve df.set_index('ts').sort_index()
    """
    d = df.copy()
    cols = {c.lower().strip(): c for c in d.columns}

    def has(name): return name in cols
    def col(name): return cols[name]

    ts_col = None

    # A) 'ts' directo
    if has('ts'):
        ts_col = col('ts')
    # B) 'fecha' + 'hora'
    elif has('fecha') and has('hora'):
        # Limpiar hora (ej. '9:00' -> '09:00:00')
        hora_limpia = pd.to_datetime(d[col('hora')], format='%H:%M:%S', errors='coerce').dt.strftime('%H:%M:%S')
        # Si falla (ej. ya es un str '09:00'), usarlo directo
        if hora_limpia.isna().all():
             hora_limpia = d[col('hora')].astype(str)
             
        d['ts_temp'] = d[col('fecha')].astype(str) + " " + hora_limpia.astype(str)
        ts_col = 'ts_temp'
    # C) 'datatime' o 'datetime'
    elif has('datatime'):
        ts_col = col('datatime')
    elif has('datetime'):
        ts_col = col('datetime')
    
    if ts_col:
        # Quitamos infer_datetime_format (deprecated)
        ts = pd.to_datetime(d[ts_col], errors='coerce', dayfirst=True)
    else:
        raise ValueError("No se encontraron columnas 'ts', ('fecha' y 'hora'), o 'datatime' en el CSV.")

    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(tz, ambiguous='NaT', nonexistent='NaT')
    else:
        ts = ts.dt.tz_convert(tz)
    
    d['ts'] = ts
    d = d.dropna(subset=['ts']).sort_values('ts')
    
    # Eliminar duplicados de índice (común en históricos)
    d = d.loc[~d.index.duplicated(keep='first')]
    
    d = d.set_index('ts')
    
    # Re-muestrear a 1H para asegurar que no falten horas
    d = d.asfreq('h')
    
    return d


def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Añade features de tiempo cíclicas y categóricas."""
    d = df.copy()
    # Asegurar que el índice es datetime
    if not isinstance(d.index, pd.DatetimeIndex):
        try:
            d.index = pd.to_datetime(d.index, errors='coerce')
            d = d.dropna(subset=['index'])
        except Exception as e:
            raise ValueError(f"add_time_parts requiere un DatetimeIndex. Error: {e}")
            
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

# --- ¡¡AQUÍ ESTÁ LA CORRECCIÓN!! ---
def add_lags_mas(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Crea features autoregresivas (lags y MAs) para el target_col."""
    
    # No copiar el df, crear un df nuevo solo con las features AR
    d = pd.DataFrame(index=df.index)
    
    target = df[target_col] # Usar el target original
    target_shifted = target.shift(1) # Usar shift(1) para MAs/EMAs
    
    for lag in [24, 48, 72, 168]:
        d[f"lag_{lag}"] = target.shift(lag)
    
    for w in [6, 12, 24, 72, 168]:
        d[f"ma_{w}"] = target_shifted.rolling(w, min_periods=1).mean()
        
    for span in [6, 12, 24]:
        d[f"ema_{span}"] = target_shifted.ewm(span=span, adjust=False, min_periods=1).mean()
        
    for w in [24, 72]:
        d[f"std_{w}"] = target_shifted.rolling(w, min_periods=2).std()
        d[f"max_{w}"] = target_shifted.rolling(w, min_periods=1).max()
        
    # Rellenar NaNs (ej. std al inicio)
    # ffill() para rellenar NaNs (ej. std al inicio)
    # bfill() para el primer registro
    # fillna(0) para lo que quede
    return d.ffill().bfill().fillna(0)
# --- FIN DE LA CORRECCIÓN ---


def dummies_and_reindex(df: pd.DataFrame, model_columns: list) -> pd.DataFrame:
    """Aplica dummies y reindexa para alinear con el entrenamiento."""
    # Columnas categóricas esperadas por el modelo (basado en train)
    cat_cols = ['dow', 'month', 'hour']
    
    # Asegurarnos de que no haya duplicados *antes* de get_dummies
    df_no_duplicates = df.loc[:, ~df.columns.duplicated()]
    
    df_dummies = pd.get_dummies(df_no_duplicates, columns=cat_cols, drop_first=False)
    
    # Reindexar para asegurar que todas las columnas del modelo existan
    df_reindexed = df_dummies.reindex(columns=model_columns, fill_value=0)
    
    return df_reindexed

# --- FUNCIONES AÑADIDAS ---

def mark_holidays_index(idx: pd.DatetimeIndex, holidays_set: set) -> pd.Series:
    """Crea una Serie (0/1) de feriados a partir de un DatetimeIndex y un set de holidays."""
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, errors='coerce')
    
    # Manejar el caso de que el set de feriados esté vacío
    if not holidays_set:
        return pd.Series(0, index=idx.date)
        
    return pd.Series(idx.date).isin(holidays_set).astype(int)


def add_es_dia_de_pago(df: pd.DataFrame) -> pd.DataFrame:
    """Añade feature 'es_dia_de_pago' (usa 'day' si existe, si no, la crea desde el index)."""
    d = df.copy()
    if 'day' not in d.columns:
        # Asumir que el índice es datetime si 'day' no está
        if isinstance(d.index, pd.DatetimeIndex):
            d['day'] = d.index.day
        else:
            print("WARN: No se puede crear 'es_dia_de_pago' sin columna 'day' o DatetimeIndex.")
            d['es_dia_de_pago'] = 0
            return d
            
    d["es_dia_de_pago"] = d["day"].isin([1,2,15,16,29,30,31]).astype(int)
    return d
