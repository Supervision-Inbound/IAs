# src/main.py
import argparse
import os
import numpy as np
import pandas as pd

# --- ¡Importar la nueva función orquestadora! ---
from src.inferencia.inferencia_core import forecast_separate_outputs
# --- Ya no se importan Erlang ni Utils_IO aquí ---

from src.inferencia.features import ensure_ts
from src.data.loader_tmo import load_historico_tmo

DATA_FILE = "data/historical_data.csv"
HOLIDAYS_FILE = "data/Feriados_Chilev2.csv"
TMO_HIST_FILE = "data/HISTORICO_TMO.csv"

TARGET_CALLS_NEW = "recibidos_nacional"
TARGET_TMO_NEW = "tmo_general"
TZ = "America/Santiago"
# PUBLIC_DIR se define ahora en inferencia_core.py

def smart_read_historical(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
        if df.shape[1] > 1: return df
    except Exception: pass
    try: # Añadir fallback para delimitador ;
        df = pd.read_csv(path, delimiter=';', low_memory=False)
        if df.shape[1] > 1: return df
    except Exception: pass
    # Si ambos fallan, intentar con ; y manejo de errores más explícito
    try:
        return pd.read_csv(path, delimiter=';', low_memory=False, on_bad_lines='warn')
    except Exception as e:
        print(f"ERROR: No se pudo leer {path} con ',' ni ';'. Error: {e}")
        raise # Re-lanzar la excepción para detener la ejecución


def parse_tmo_to_seconds(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().replace(",", ".")
    if ':' in s:
        parts = s.split(":")
        try:
            if len(parts) == 3: return float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
            if len(parts) == 2: return float(parts[0])*60 + float(parts[1])
        except ValueError: return np.nan # Manejar partes no numéricas
    try: # Intentar convertir directamente a float si no hay ':'
        return float(s)
    except ValueError: return np.nan


def load_holidays(csv_path: str) -> set:
    if not os.path.exists(csv_path): return set()
    try: # Intentar leer con ambos delimitadores
        try: fer = pd.read_csv(csv_path)
        except pd.errors.ParserError: fer = pd.read_csv(csv_path, delimiter=';')
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
    # Usar format explícito si es posible, dayfirst=True como fallback
    try:
        fechas = pd.to_datetime(fer[fecha_col].astype(str), dayfirst=True, errors="coerce").dropna().dt.date
    except Exception as e:
        print(f"WARN: Error parseando fechas de feriados: {e}")
        return set()
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
    # Ya no se crea PUBLIC_DIR aquí, lo hace inferencia_core

    # 1) Leer histórico principal (llamadas)
    dfh = smart_read_historical(DATA_FILE)
    dfh.columns = dfh.columns.str.strip()
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
         target_tmo_found = True
         # Convertir TMO existente si está en formato mm:ss o similar
         dfh[TARGET_TMO_NEW] = dfh[TARGET_TMO_NEW].apply(parse_tmo_to_seconds)
    else:
        tmo_source = None
        for cand in ["tmo (segundos)", "tmo_seg", "tmo", "tmo_general"]:
            if cand in dfh.columns:
                tmo_source = cand; break
        if tmo_source:
            print(f"INFO: Usando columna '{tmo_source}' como TMO y renombrando a '{TARGET_TMO_NEW}'.")
            dfh[TARGET_TMO_NEW] = dfh[tmo_source].apply(parse_tmo_to_seconds)
            # Eliminar la columna original para evitar confusión
            if tmo_source != TARGET_TMO_NEW:
                 dfh = dfh.drop(columns=[tmo_source])
            target_tmo_found = True

    dfh = ensure_ts(dfh)

    # 2) Fusionar HISTORICO_TMO.csv (alineado por ts)
    if os.path.exists(TMO_HIST_FILE):
        try:
            print(f"INFO: Cargando y uniendo {TMO_HIST_FILE}...")
            df_tmo = load_historico_tmo(TMO_HIST_FILE)
            # Asegurarse que TARGET_TMO_NEW exista antes de combine_first
            if TARGET_TMO_NEW not in dfh.columns:
                 dfh[TARGET_TMO_NEW] = np.nan

            dfh = dfh.join(df_tmo, how="left")
            # Priorizar TMO de TMO_HISTORICO si existe
            if "tmo_general" in df_tmo.columns: # Chequear en df_tmo antes de usarlo
                dfh[TARGET_TMO_NEW] = dfh["tmo_general"].combine_first(dfh[TARGET_TMO_NEW])
                target_tmo_found = True # Marcar como encontrado si vino de aquí
            elif TARGET_TMO_NEW in df_tmo.columns: # Si loader_tmo lo renombró
                 dfh[TARGET_TMO_NEW] = dfh[TARGET_TMO_NEW+"_right"].combine_first(dfh[TARGET_TMO_NEW+"_left"]) # Pandas añade sufijos en join si los nombres colisionan
                 target_tmo_found = True

        except Exception as e:
            print(f"WARN: No se pudo cargar o unir {TMO_HIST_FILE}. Error: {e}")

    # Asegurar que la columna TMO exista después de todos los intentos
    if not target_tmo_found:
        print(f"WARN: No se encontró columna TMO ('{TARGET_TMO_NEW}' o alternativas) en ningún archivo. Se usará 0.")
        dfh[TARGET_TMO_NEW] = 0.0
    # Asegurar que TMO sea numérico y rellenar NaNs ANTES de pasarlo a la inferencia de TMO v8
    dfh[TARGET_TMO_NEW] = pd.to_numeric(dfh[TARGET_TMO_NEW], errors='coerce').ffill().fillna(0.0)


    # 3) Derivar calendario para el histórico
    holidays_set = load_holidays(HOLIDAYS_FILE)
    if "feriados" not in dfh.columns:
        dfh["feriados"] = mark_holidays_index(dfh.index, holidays_set).values
    dfh["feriados"] = pd.to_numeric(dfh["feriados"], errors="coerce").fillna(0).astype(int)
    if "es_dia_de_pago" not in dfh.columns:
        dfh["es_dia_de_pago"] = add_es_dia_de_pago(dfh.index).values # Pasar índice directamente
    dfh["es_dia_de_pago"] = pd.to_numeric(dfh["es_dia_de_pago"], errors="coerce").fillna(0).astype(int)


    # 4) ffill de columnas clave (Lógica v1 - EXCLUYE TMO)
    #    (El TMO ya se rellenó específicamente arriba para v8)
    print("INFO: Aplicando ffill a columnas de calendario y auxiliares (excluyendo TMO)...")
    for c in ["feriados", "es_dia_de_pago",
              "proporcion_comercial", "proporcion_tecnica", "tmo_comercial", "tmo_tecnico"]:
        if c in dfh.columns:
            dfh[c] = dfh[c].ffill()

    # --- ¡NUEVA LLAMADA A LA FUNCIÓN ORQUESTADORA! ---
    # 5) Forecast (Llama a la función que genera salidas separadas)
    forecast_separate_outputs(
        dfh.reset_index(), # Se pasa el dfh preparado
        horizon_days=horizonte_dias,
        holidays_set=holidays_set
    )
    # --- Ya no se calcula Erlang ni se escriben JSONs aquí ---

    # --- Las alertas de clima ahora deben leer el JSON de llamadas ---
    # (Opcional: podrías modificar generar_alertas para aceptar el path del JSON)
    # from src.inferencia.alertas_clima import generar_alertas
    # calls_json_path = os.path.join(PUBLIC_DIR, "prediccion_horaria_llamadas.json")
    # if os.path.exists(calls_json_path):
    #     try:
    #         df_calls_output = pd.read_json(calls_json_path)
    #         df_calls_output['ts'] = pd.to_datetime(df_calls_output['ts'])
    #         df_calls_output = df_calls_output.set_index('ts')
    #         generar_alertas(df_calls_output[["llamadas_hora"]].rename(columns={"llamadas_hora":"calls"}))
    #     except Exception as e:
    #         print(f"WARN: No se pudieron generar alertas de clima. Error leyendo JSON: {e}")
    # else:
    #      print(f"WARN: Archivo {calls_json_path} no encontrado para generar alertas.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizonte", type=int, default=120)
    args = ap.parse_args()
    main(args.horizonte)
