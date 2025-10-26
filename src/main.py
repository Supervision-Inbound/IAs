# src/main.py
import argparse
import os
import numpy as np
import pandas as pd
import warnings

from src.inferencia.inferencia_core import forecast_120d
from src.inferencia.features import ensure_ts, add_es_dia_de_pago
# <--- MODIFICADO: Ya no importamos 'load_historico_tmo'
# from src.data.loader_tmo import load_historico_tmo 

DATA_FILE = "data/historical_data.csv"
HOLIDAYS_FILE = "data/Feriados_Chilev2.csv"
# <--- MODIFICADO: Eliminada la referencia al TMO_HIST_FILE
# TMO_HIST_FILE = "data/HISTORICO_TMO.csv" 

TARGET_CALLS_NEW = "contestados"
TARGET_TMO_NEW = "tmo (segundos)" # <--- MODIFICADO: Usamos el TMO del CSV principal
TZ = "America/Santiago"

def smart_read_historical(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    # Fallback a delimitador ;
    try:
        return pd.read_csv(path, delimiter=';', low_memory=False)
    except Exception as e:
        raise ValueError(f"No se pudo leer el CSV {path} con , ni con ;. Error: {e}")

def parse_tmo_to_seconds(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().replace(",", ".")
    if s.replace(".", "", 1).isdigit():
        try: return float(s)
        except: return np.nan
    
    # Check si es HH:MM:SS o MM:SS
    parts = s.split(":")
    try:
        if len(parts) == 3: return float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
        if len(parts) == 2: return float(parts[1])*60 + float(parts[0]) # Asumimos MM:SS
    except Exception:
        return np.nan
    return np.nan

def load_holidays(path: str) -> set:
    try:
        df_h = pd.read_csv(path, header=None, names=['date', 'desc'])
        df_h['date'] = pd.to_datetime(df_h['date'], dayfirst=True).dt.date
        return set(df_h['date'])
    except Exception as e:
        print(f"WARN: No se pudo cargar feriados desde {path}. {e}", file=sys.stderr)
        return set()

def mark_holidays_index(idx: pd.DatetimeIndex, holidays_set: set) -> pd.Series:
    return pd.Series(idx.date, index=idx).isin(holidays_set).astype(int)


def run_forecast_pipeline():
    """
    Función principal que ejecuta el pipeline de inferencia.
    """
    print(f"Iniciando pipeline de inferencia...")
    print(f"Usando data histórica: {DATA_FILE}")

    # 1) Cargar histórico (el CSV unificado)
    df_raw = smart_read_historical(DATA_FILE)
    
    # 2) <--- MODIFICADO: Procesamiento unificado
    # Ya no cargamos TMO por separado.
    # Aseguramos que 'recibidos_nacional' y 'tmo (segundos)' existan y sean numéricos
    
    dfh = ensure_ts(df_raw, tz=TZ)
    
    if TARGET_CALLS_NEW not in dfh.columns:
        # Intentar buscar columnas candidatas para LLAMADAS
        call_cands = ['recibidos', 'llamadas_recibidas', 'calls']
        found_call = False
        for c in call_cands:
            if c in dfh.columns:
                dfh = dfh.rename(columns={c: TARGET_CALLS_NEW})
                found_call = True
                break
        if not found_call:
             raise ValueError(f"No se encuentra la columna de llamadas '{TARGET_CALLS_NEW}' en {DATA_FILE}")
    
    if TARGET_TMO_NEW not in dfh.columns:
        # Intentar buscar columnas candidatas para TMO
        tmo_cands = ['tmo_general', 'tmo', 'tmo (s)', 'tmo seg']
        found_tmo = False
        for c in tmo_cands:
             if c in dfh.columns:
                dfh = dfh.rename(columns={c: TARGET_TMO_NEW})
                found_tmo = True
                break
        if not found_tmo:
            raise ValueError(f"No se encuentra la columna TMO '{TARGET_TMO_NEW}' en {DATA_FILE}")

    print(f"Usando '{TARGET_CALLS_NEW}' para llamadas y '{TARGET_TMO_NEW}' para TMO.")

    # Convertir a numérico
    dfh[TARGET_CALLS_NEW] = pd.to_numeric(dfh[TARGET_CALLS_NEW], errors='coerce')
    
    # Intentar parsear TMO si es string (HH:MM:SS)
    if not pd.api.types.is_numeric_dtype(dfh[TARGET_TMO_NEW]):
        print(f"Columna '{TARGET_TMO_NEW}' no es numérica, intentando parsear H:M:S...")
        dfh[TARGET_TMO_NEW] = dfh[TARGET_TMO_NEW].apply(parse_tmo_to_seconds)
        
    dfh[TARGET_TMO_NEW] = pd.to_numeric(dfh[TARGET_TMO_NEW], errors='coerce')

    # Rellenar nulos iniciales o faltantes
    dfh[TARGET_CALLS_NEW] = dfh[TARGET_CALLS_NEW].fillna(0)
    dfh[TARGET_TMO_NEW] = dfh[TARGET_TMO_NEW].fillna(method='ffill').fillna(200) # ffill y luego un valor default
    
    # 3) Derivar calendario para el histórico
    holidays_set = load_holidays(HOLIDAYS_FILE)
    if "feriados" not in dfh.columns:
        dfh["feriados"] = mark_holidays_index(dfh.index, holidays_set).values
    dfh["feriados"] = pd.to_numeric(dfh["feriados"], errors='coerce').fillna(0).astype(int)
    
    if "es_dia_de_pago" not in dfh.columns:
        dfh["es_dia_de_pago"] = add_es_dia_de_pago(dfh).values

    # 4) ffill de columnas clave para evitar NaN en el borde
    for c in [TARGET_TMO_NEW, "feriados", "es_dia_de_pago"]:
        if c in dfh.columns:
            dfh[c] = dfh[c].ffill()

    # 5) <--- MODIFICADO: Forecast
    # Ya no pasamos 'df_tmo_hist_only'. La función 'forecast_120d'
    # usará el 'dfh' unificado para las features de TMO.
    print("Iniciando forecast_120d...")
    df_hourly = forecast_120d(
        dfh.reset_index(),
        # df_tmo_hist_only.reset_index() if df_tmo_hist_only is not None else None, # <--- ELIMINADO
        holidays_set=holidays_set
    )

    print("Pipeline de inferencia completado.")
    return df_hourly


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    run_forecast_pipeline()
