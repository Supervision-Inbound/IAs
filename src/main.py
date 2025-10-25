# src/main.py (Original v1)
import argparse
import os
import numpy as np
import pandas as pd

from src.inferencia.inferencia_core import forecast_120d # <-- Llama a la función única
from src.inferencia.features import ensure_ts
from src.data.loader_tmo import load_historico_tmo

DATA_FILE = "data/historical_data.csv"
HOLIDAYS_FILE = "data/Feriados_Chilev2.csv"
TMO_HIST_FILE = "data/HISTORICO_TMO.csv"

TARGET_CALLS_NEW = "recibidos_nacional"
TARGET_TMO_NEW = "tmo_general"
TZ = "America/Santiago"

def smart_read_historical(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    # Añadir manejo explícito para delimitador ;
    try:
        return pd.read_csv(path, delimiter=';', low_memory=False)
    except Exception as e:
        print(f"ERROR: No se pudo leer {path} con ',' ni ';'. Error: {e}")
        raise

def parse_tmo_to_seconds(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().replace(",", ".")
    if ':' in s: # Manejar formato mm:ss o hh:mm:ss
        parts = s.split(":")
        try:
            if len(parts) == 3: return float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
            if len(parts) == 2: return float(parts[0])*60 + float(parts[1])
        except ValueError: return np.nan # Si las partes no son números
    try: # Intentar convertir directamente si no hay ':'
        return float(s)
    except ValueError: return np.nan

def load_holidays(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        print(f"WARN: Archivo de feriados no encontrado en {csv_path}")
        return set()
    try: # Intentar leer con ambos delimitadores
        try: fer = pd.read_csv(csv_path)
        except (pd.errors.ParserError, UnicodeDecodeError): fer = pd.read_csv(csv_path, delimiter=';')
    except Exception as e:
        print(f"WARN: No se pudo leer el archivo de feriados {csv_path}. Error: {e}")
        return set()

    cols_map = {c.lower().strip(): c for c in fer.columns}
    fecha_col = None
    for cand in ["fecha", "date", "dia", "día"]:
        if cand in cols_map:
            fecha_col = cols_map[cand]; break
    if not fecha_col:
        print(f"WARN: No se encontró columna de fecha en {csv_path}.")
        return set()
    try:
        # Usar dayfirst=True y errors='coerce'
        fechas = pd.to_datetime(fer[fecha_col].astype(str), dayfirst=True, errors="coerce").dropna().dt.date
    except Exception as e:
        print(f"WARN: Error parseando fechas de feriados: {e}")
        return set()
    print(f"INFO: Cargados {len(fechas)} feriados desde {csv_path}")
    return set(fechas)

def mark_holidays_index(dt_index, holidays_set: set) -> pd.Series:
     # Asegurar índice datetime
    if not isinstance(dt_index, pd.DatetimeIndex):
        try: dt_index = pd.to_datetime(dt_index)
        except Exception: return pd.Series(0, index=dt_index, dtype=int, name="feriados") # Fallback

    tz = getattr(dt_index, "tz", None)
    try:
        idx_dates = dt_index.tz_convert(TZ).date if tz is not None else dt_index.date
    except Exception: # Fallback si falla conversión TZ
        idx_dates = dt_index.date

    return pd.Series([d in holidays_set for d in idx_dates], index=dt_index, dtype=int, name="feriados")

def add_es_dia_de_pago(df_idx: pd.DataFrame | pd.Index) -> pd.Series:
     # Asegurar índice datetime
    idx = df_idx if isinstance(df_idx, pd.Index) else df_idx.index
    if not isinstance(idx, pd.DatetimeIndex):
        try: idx = pd.to_datetime(idx)
        except Exception: return pd.Series(0, index=idx, dtype=int, name="es_dia_de_pago") # Fallback

    dias = [1,2,15,16,29,30,31]
    return pd.Series(idx.day.isin(dias).astype(int), index=idx, name="es_dia_de_pago")

def main(horizonte_dias: int):
    os.makedirs("public", exist_ok=True)
    print(f"INFO: Directorio 'public' asegurado.")

    # 1) Leer histórico principal (llamadas)
    print(f"INFO: Cargando datos históricos desde {DATA_FILE}...")
    dfh = smart_read_historical(DATA_FILE)
    dfh.columns = dfh.columns.str.strip()
    print(f"INFO: {DATA_FILE} cargado. Columnas: {list(dfh.columns)}")

    target_calls_found = False
    if TARGET_CALLS_NEW in dfh.columns:
        target_calls_found = True
    else:
        for cand in ["recibidos_nacional", "recibidos", "total_llamadas", "llamadas"]:
            if cand in dfh.columns:
                print(f"INFO: Renombrando columna '{cand}' a '{TARGET_CALLS_NEW}'.")
                dfh = dfh.rename(columns={cand: TARGET_CALLS_NEW})
                target_calls_found = True
                break
    if not target_calls_found:
         raise ValueError(f"ERROR: No se encontró la columna target de llamadas ('{TARGET_CALLS_NEW}' o alternativas) en {DATA_FILE}")

    target_tmo_found = False
    if TARGET_TMO_NEW in dfh.columns:
         print(f"INFO: Columna TMO '{TARGET_TMO_NEW}' encontrada en {DATA_FILE}. Parseando a segundos...")
         dfh[TARGET_TMO_NEW] = dfh[TARGET_TMO_NEW].apply(parse_tmo_to_seconds)
         target_tmo_found = True
    else:
        tmo_source = None
        for cand in ["tmo (segundos)", "tmo_seg", "tmo", "tmo_general"]:
            if cand in dfh.columns:
                tmo_source = cand; break
        if tmo_source:
            print(f"INFO: Usando columna '{tmo_source}' como TMO desde {DATA_FILE} y renombrando a '{TARGET_TMO_NEW}'.")
            dfh[TARGET_TMO_NEW] = dfh[tmo_source].apply(parse_tmo_to_seconds)
            if tmo_source != TARGET_TMO_NEW:
                 # Intentar eliminar la columna original si existe
                 try: dfh = dfh.drop(columns=[tmo_source])
                 except KeyError: pass # Ignorar si ya no existe
            target_tmo_found = True

    try:
        dfh = ensure_ts(dfh)
        print(f"INFO: Timestamps procesados para {DATA_FILE}. Rango: {dfh.index.min()} a {dfh.index.max()}")
    except ValueError as e:
        print(f"ERROR: Falló ensure_ts para {DATA_FILE}: {e}")
        return # Detener si no se pueden procesar timestamps

    # 2) Fusionar HISTORICO_TMO.csv (alineado por ts)
    df_tmo_hist_only = None
    if os.path.exists(TMO_HIST_FILE):
        try:
            print(f"INFO: Cargando y uniendo {TMO_HIST_FILE}...")
            df_tmo = load_historico_tmo(TMO_HIST_FILE)
            df_tmo_hist_only = df_tmo.copy() # Guardar copia para pasar a forecast_120d (v1)

            # Asegurarse que TARGET_TMO_NEW exista antes de combine_first
            if TARGET_TMO_NEW not in dfh.columns:
                 dfh[TARGET_TMO_NEW] = np.nan

            dfh = dfh.join(df_tmo, how="left", lsuffix='_hist', rsuffix='_tmo') # Añadir sufijos para evitar colisiones

            # Priorizar TMO de TMO_HISTORICO si existe
            tmo_col_from_tmo_hist = TARGET_TMO_NEW # Nombre estándar esperado de load_historico_tmo
            if tmo_col_from_tmo_hist in df_tmo.columns: # Verificar si load_historico_tmo lo devolvió
                 # El join crea _tmo si ya existía en dfh
                 tmo_col_joined = tmo_col_from_tmo_hist + '_tmo' if tmo_col_from_tmo_hist+'_hist' in dfh.columns else tmo_col_from_tmo_hist
                 if tmo_col_joined in dfh.columns:
                     print(f"INFO: Priorizando TMO desde {TMO_HIST_FILE} (columna '{tmo_col_joined}').")
                     # La columna original (si existía) podría ser '_hist'
                     tmo_col_original = tmo_col_from_tmo_hist + '_hist' if tmo_col_joined != tmo_col_from_tmo_hist else tmo_col_from_tmo_hist
                     dfh[TARGET_TMO_NEW] = dfh[tmo_col_joined].combine_first(dfh[tmo_col_original])
                     target_tmo_found = True # Marcar como encontrado/priorizado
                     # Limpiar columnas auxiliares del join
                     cols_to_drop = [c for c in [tmo_col_joined, tmo_col_original] if c != TARGET_TMO_NEW and c in dfh.columns]
                     if cols_to_drop: dfh = dfh.drop(columns=cols_to_drop)
                 else:
                      print(f"WARN: Se esperaba la columna '{tmo_col_joined}' después del join con TMO_HIST, pero no se encontró.")
            elif target_tmo_found:
                 print(f"INFO: TMO ya presente desde {DATA_FILE}, no se sobrescribió.")
            else:
                 print(f"WARN: No se encontró TMO en {TMO_HIST_FILE} para priorizar.")

        except Exception as e:
            print(f"WARN: No se pudo cargar o unir {TMO_HIST_FILE}. Error: {e}")
    else:
         print(f"INFO: Archivo {TMO_HIST_FILE} no encontrado, se usarán datos de TMO (si existen) de {DATA_FILE}.")


    # Asegurar que la columna TMO exista después de todos los intentos, antes de ffill
    if not target_tmo_found:
        print(f"WARN: No se encontró columna TMO ('{TARGET_TMO_NEW}' o alternativas) en ningún archivo. Se usará NaN antes de inferencia.")
        dfh[TARGET_TMO_NEW] = np.nan # Dejar NaN para que la lógica v1 de inferencia lo maneje
    else:
         # Asegurar que TMO sea numérico (los NaNs se mantienen para lógica v1)
         dfh[TARGET_TMO_NEW] = pd.to_numeric(dfh[TARGET_TMO_NEW], errors='coerce')


    # 3) Derivar calendario para el histórico
    holidays_set = load_holidays(HOLIDAYS_FILE)
    if "feriados" not in dfh.columns:
        print("INFO: Generando columna 'feriados'...")
        dfh["feriados"] = mark_holidays_index(dfh.index, holidays_set).values
    dfh["feriados"] = pd.to_numeric(dfh["feriados"], errors="coerce").fillna(0).astype(int)
    if "es_dia_de_pago" not in dfh.columns:
        print("INFO: Generando columna 'es_dia_de_pago'...")
        dfh["es_dia_de_pago"] = add_es_dia_de_pago(dfh.index).values # Pasar índice directamente
    dfh["es_dia_de_pago"] = pd.to_numeric(dfh["es_dia_de_pago"], errors="coerce").fillna(0).astype(int)
    print(f"INFO: Columnas de calendario procesadas. Feriados encontrados: {dfh['feriados'].sum()}")


    # 4) ffill de columnas clave (LÓGICA v1 - EXCLUYE TMO)
    print("INFO: Aplicando ffill a columnas auxiliares (excluyendo TMO)...")
    ffill_cols_v1 = ["feriados", "es_dia_de_pago",
                     "proporcion_comercial", "proporcion_tecnica", "tmo_comercial", "tmo_tecnico"]
    for c in ffill_cols_v1:
        if c in dfh.columns:
            # Rellenar solo si la columna no es completamente NaN
            if not dfh[c].isnull().all():
                 print(f"  - Rellenando NaNs en '{c}'...")
                 dfh[c] = dfh[c].ffill()
            else:
                 print(f"  - Columna '{c}' es completamente NaN, no se rellena.")
        #else: # Descomentar si quieres verbosidad sobre columnas faltantes
        #    print(f"  - Columna '{c}' no encontrada para ffill.")


    # 5) Forecast (Llamada original v1)
    print("\nINFO: Iniciando forecast_120d...")
    df_hourly = forecast_120d(
        dfh.reset_index(), # Pasar dfh preparado con lógica v1
        df_tmo_hist_only.reset_index() if df_tmo_hist_only is not None else None, # Se pasa, aunque v31 lo ignore
        horizon_days=horizonte_dias,
        holidays_set=holidays_set
    )
    print("INFO: forecast_120d completado.")


    # 6) Alertas clima
    try:
        from src.inferencia.alertas_clima import generar_alertas
        print("INFO: Generando alertas de clima...")
        # Pasar copia de df_hourly por si generar_alertas lo modifica
        generar_alertas(df_hourly[["calls"]].copy())
        print("INFO: Alertas de clima generadas.")
    except ImportError:
        print("WARN: Módulo 'alertas_clima' no encontrado. Saltando generación de alertas.")
    except Exception as e:
         print(f"ERROR: Falló la generación de alertas de clima: {e}")

    print("--- Proceso Principal Finalizado ---")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizonte", type=int, default=120, help="Número de días a predecir.")
    args = ap.parse_args()
    print(f"INFO: Ejecutando main con horizonte = {args.horizonte} días.")
    main(args.horizonte)
