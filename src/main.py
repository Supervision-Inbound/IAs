# src/main.py
import argparse
import os
import re, unicodedata  # <-- Importar
import numpy as np
import pandas as pd

from src.inferencia.inferencia_core import forecast_120d
from src.inferencia.features import ensure_ts, add_es_dia_de_pago
from src.inferencia.utils_io import load_holidays # Asumiendo que utils_io tiene load_holidays
# from src.data.loader_tmo import load_historico_tmo # <-- ELIMINADO

DATA_FILE = "data/historical_data.csv"
HOLIDAYS_FILE = "data/Feriados_Chilev2.csv"
# TMO_HIST_FILE = "data/HISTORICO_TMO.csv" # <-- ELIMINADO

TARGET_CALLS_NEW = "recibidos_nacional"
TARGET_TMO_NEW = "tmo_general" # Nombre estándar interno que usa inferencia_core
TZ = "America/Santiago"

# ==================================================================
# Funciones de utilidad (copiadas de tu script de entrenamiento)
# para encontrar la columna TMO correcta.
# ==================================================================
def _norm(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def find_tmo_col(df: pd.DataFrame) -> str:
    norm_map = {_norm(c): c for c in df.columns}
    # 1. Buscar 'tmo (s)' o 'tmo (segundos)' exacto (normalizado)
    for n, orig in norm_map.items():
        if re.match(r"^tmo\s*\(\s*(s|segundos)\s*\)$", n):
            return orig
    # 2. Buscar heurística
    for n, orig in norm_map.items():
        if "tmo" in n and ("(s)" in n or "segundo" in n):
            return orig
    # 3. Fallback al nombre estándar (si ya existe)
    if TARGET_TMO_NEW in df.columns:
         return TARGET_TMO_NEW
    raise ValueError(f"No se encontró columna TMO ('{TARGET_TMO_NEW}', 'tmo (s)' o 'tmo (segundos)') en el CSV.")

# ==================================================================

def smart_read_historical(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
        if df.shape[1] > 1 or (df.shape[1] == 1 and ';' not in str(df.iloc[0,0])):
             return df
    except Exception:
        pass
    print("Leyendo CSV con separador ';'")
    return pd.read_csv(path, delimiter=';', low_memory=False)

def main():
    parser = argparse.ArgumentParser(description="Ejecutar inferencia de demanda")
    parser.add_argument("--data", type=str, default=DATA_FILE, help="Ruta al CSV de datos históricos.")
    parser.add_argument("--holidays", type=str, default=HOLIDAYS_FILE, help="Ruta al CSV de feriados.")
    args = parser.parse_args()

    # 1) Cargar histórico (ahora contiene tanto llamadas como TMO)
    print(f"Cargando datos históricos desde: {args.data}")
    dfh = smart_read_historical(args.data)
    dfh = ensure_ts(dfh, TZ) # Crea 'ts' indexado y normalizado a TZ

    # 2) Validar y renombrar TMO (NUEVA LÓGICA)
    # El TMO ahora viene del MISMO archivo.
    
    # Encontrar la columna TMO real (p.ej. "tmo (segundos)")
    tmo_col_real = find_tmo_col(dfh)
    
    # Renombrarla al nombre estándar que espera inferencia_core (TARGET_TMO_NEW)
    if tmo_col_real != TARGET_TMO_NEW:
        print(f"Usando columna '{tmo_col_real}' como target TMO (renombrada a '{TARGET_TMO_NEW}')")
        dfh[TARGET_TMO_NEW] = pd.to_numeric(dfh[tmo_col_real], errors='coerce')
    else:
        print(f"Usando columna TMO existente: '{TARGET_TMO_NEW}'")
        dfh[TARGET_TMO_NEW] = pd.to_numeric(dfh[TARGET_TMO_NEW], errors='coerce')

    # *** ELIMINADO BLOQUE de 'load_historico_tmo' y merge ***
    print(f"TMO se leerá desde la columna '{tmo_col_real}' del archivo principal.")

    # 3) Derivar calendario para el histórico
    holidays_set = load_holidays(args.holidays)
    if "feriados" not in dfh.columns:
        dfh["feriados"] = mark_holidays_index(dfh.index, holidays_set).values
    dfh["feriados"] = pd.to_numeric(dfh["feriados"], errors="coerce").fillna(0).astype(int)
    
    if "es_dia_de_pago" not in dfh.columns:
        dfh["es_dia_de_pago"] = add_es_dia_de_pago(dfh).values

    # 4) ffill de columnas clave (incluyendo el nuevo TMO)
    fill_cols = [TARGET_TMO_NEW, "feriados", "es_dia_de_pago"]
    # Añadir otras columnas si existen (p.ej. proporciones)
    for c in ["proporcion_comercial", "proporcion_tecnica", "tmo_comercial", "tmo_tecnico"]:
        if c in dfh.columns:
            fill_cols.append(c)
            
    for c in fill_cols:
        if c in dfh.columns:
            dfh[c] = dfh[c].ffill()
        else:
            print(f"Advertencia: Columna '{c}' no encontrada para ffill.")

    # 5) Forecast (LLAMADA MODIFICADA)
    print("Iniciando forecast_120d...")
    df_hourly = forecast_120d(
        dfh.reset_index(), # Pasamos el DF histórico completo
        holidays_set=holidays_set
        # Ya no se pasa 'df_tmo_hist_only'
    )

    print("\nProceso de inferencia completado.")
    print(df_hourly.head())

if __name__ == "__main__":
    # (Asegúrate de que 'load_holidays' esté importado correctamente,
    # moví la importación arriba)
    
    # Placeholder simple si no existe en utils_io
    if 'load_holidays' not in globals():
        def load_holidays(path):
            print(f"Cargando feriados (simple) de {path}")
            try:
                df_h = pd.read_csv(path, delimiter=';') # Asumiendo ;
                if 'fecha' not in df_h.columns:
                     df_h = pd.read_csv(path) # Fallback a ,
                return set(pd.to_datetime(df_h['fecha'], dayfirst=True).dt.date)
            except Exception as e:
                print(f"WARN: No se pudieron cargar feriados: {e}. Usando set vacío.")
                return set()

    if 'mark_holidays_index' not in globals():
        def mark_holidays_index(idx, holidays_set):
             return pd.Series(idx.date).isin(holidays_set).astype(int)
             
    main()
