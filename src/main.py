# src/main.py
import argparse
import os
import re, unicodedata
import numpy as np
import pandas as pd

from src.inferencia.inferencia_core import forecast_120d
# --- IMPORTACIONES CORREGIDAS ---
from src.inferencia.features import ensure_ts, add_es_dia_de_pago, mark_holidays_index
from src.inferencia.utils_io import load_holidays
# --------------------------------

DATA_FILE = "data/historical_data.csv"
HOLIDAYS_FILE = "data/Feriados_Chilev2.csv"
# TMO_HIST_FILE = "data/HISTORICO_TMO.csv" # <-- ELIMINADO
TARGET_CALLS_NEW = "recibidos_nacional"
TARGET_TMO_NEW = "tmo_general" # Nombre estándar interno
TZ = "America/Santiago"

# ==================================================================
# Funciones de utilidad para encontrar TMO (de train_v7.py)
# ==================================================================
def _norm(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def find_tmo_col(df: pd.DataFrame) -> str:
    """Encuentra la columna TMO correcta, p.ej. 'tmo (segundos)'."""
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
    """Lee CSV detectando separador ',' o ';'."""
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
    parser.add_argument("--horizonte", type=int, default=120, help="Días a predecir.")
    args = parser.parse_args()

    # 1) Cargar histórico (ahora contiene tanto llamadas como TMO)
    print(f"Cargando datos históricos desde: {args.data}")
    dfh = smart_read_historical(args.data)
    dfh = ensure_ts(dfh, TZ) # Crea 'ts' indexado y normalizado a TZ

    # 2) Validar y renombrar TMO (NUEVA LÓGICA v7)
    tmo_col_real = find_tmo_col(dfh)
    
    if tmo_col_real != TARGET_TMO_NEW:
        print(f"Usando columna '{tmo_col_real}' como target TMO (renombrada a '{TARGET_TMO_NEW}')")
        dfh[TARGET_TMO_NEW] = pd.to_numeric(dfh[tmo_col_real], errors='coerce')
    else:
        print(f"Usando columna TMO existente: '{TARGET_TMO_NEW}'")
        dfh[TARGET_TMO_NEW] = pd.to_numeric(dfh[TARGET_TMO_NEW], errors='coerce')

    # 3) Derivar calendario para el histórico
    holidays_set = load_holidays(args.holidays)
    
    if "feriados" not in dfh.columns:
        print("Creando columna 'feriados' desde el índice...")
        dfh["feriados"] = mark_holidays_index(dfh.index, holidays_set).values
    dfh["feriados"] = pd.to_numeric(dfh["feriados"], errors='coerce').fillna(0).astype(int)
    
    if "es_dia_de_pago" not in dfh.columns:
        print("Creando columna 'es_dia_de_pago'...")
        dfh = add_es_dia_de_pago(dfh) # Esta función devuelve el DF
    
    # 4) ffill de columnas clave (incluyendo el nuevo TMO)
    fill_cols = [TARGET_TMO_NEW, "feriados", "es_dia_de_pago"]
    for c in fill_cols:
        if c in dfh.columns:
            dfh[c] = dfh[c].ffill()
        else:
            print(f"Advertencia: Columna '{c}' no encontrada para ffill.")

    # 5) Forecast
    print(f"Iniciando forecast (horizonte={args.horizonte} días)...")
    df_hourly = forecast_120d(
        dfh.reset_index(), # Pasamos el DF histórico completo
        holidays_set=holidays_set,
        horizon_days=args.horizonte # Pasamos el horizonte
    )

    print("\nProceso de inferencia completado.")
    print(df_hourly.head())

if __name__ == "__main__":
    main()
