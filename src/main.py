# src/main.py
import argparse
import os
import numpy as np
import pandas as pd

# --- ¡Nuevos Imports! ---
from src.inferencia.inferencia_core import forecast_calls_v1, forecast_tmo_v8
from src.inferencia.erlang import required_agents, schedule_agents
from src.inferencia.utils_io import write_daily_json, write_hourly_json
# -------------------------

from src.inferencia.features import ensure_ts
from src.data.loader_tmo import load_historico_tmo

DATA_FILE = "data/historical_data.csv"
HOLIDAYS_FILE = "data/Feriados_Chilev2.csv"
TMO_HIST_FILE = "data/HISTORICO_TMO.csv"

TARGET_CALLS_NEW = "recibidos_nacional"
TARGET_TMO_NEW = "tmo_general"
TZ = "America/Santiago"
PUBLIC_DIR = "public" # Asegurarse que esté definido

def smart_read_historical(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    return pd.read_csv(path, delimiter=';', low_memory=False)

def parse_tmo_to_seconds(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().replace(",", ".")
    if s.replace(".", "", 1).isdigit():
        try: return float(s)
        except: return np.nan
    parts = s.split(":")
    try:
        if len(parts) == 3: return float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
        if len(parts) == 2: return float(parts[0])*60 + float(parts[1])
        return float(s)
    except:
        return np.nan

def load_holidays(csv_path: str) -> set:
    if not os.path.exists(csv_path): return set()
    fer = pd.read_csv(csv_path)
    cols_map = {c.lower().strip(): c for c in fer.columns}
    fecha_col = None
    for cand in ["fecha", "date", "dia", "día"]:
        if cand in cols_map:
            fecha_col = cols_map[cand]; break
    if not fecha_col: return set()
    fechas = pd.to_datetime(fer[fecha_col].astype(str), dayfirst=True, errors="coerce").dropna().dt.date
    return set(fechas)

def mark_holidays_index(dt_index, holidays_set: set) -> pd.Series:
    tz = getattr(dt_index, "tz", None)
    idx_dates = dt_index.tz_convert(TZ).date if tz is not None else dt_index.date
    return pd.Series([d in holidays_set for d in idx_dates], index=dt_index, dtype=int, name="feriados")

def add_es_dia_de_pago(df_idx: pd.DataFrame) -> pd.Series:
    dias = [1,2,15,16,29,30,31]
    return pd.Series(df_idx.index.day.isin(dias).astype(int), index=df_idx.index, name="es_dia_de_pago")

def main(horizonte_dias: int):
    os.makedirs(PUBLIC_DIR, exist_ok=True)

    # 1) Leer histórico principal (llamadas)
    dfh = smart_read_historical(DATA_FILE)
    dfh.columns = dfh.columns.str.strip()
    if TARGET_CALLS_NEW not in dfh.columns:
        for cand in ["recibidos_nacional", "recibidos", "total_llamadas", "llamadas"]:
            if cand in dfh.columns:
                dfh = dfh.rename(columns={cand: TARGET_CALLS_NEW})
                break
    if TARGET_TMO_NEW not in dfh.columns:
        tmo_source = None
        for cand in ["tmo (segundos)", "tmo_seg", "tmo", "tmo_general"]:
            if cand in dfh.columns:
                tmo_source = cand; break
        if tmo_source:
            dfh[TARGET_TMO_NEW] = dfh[tmo_source].apply(parse_tmo_to_seconds)

    dfh = ensure_ts(dfh)

    # 2) Fusionar HISTORICO_TMO.csv (alineado por ts)
    if os.path.exists(TMO_HIST_FILE):
        try:
            df_tmo = load_historico_tmo(TMO_HIST_FILE)
            dfh = dfh.join(df_tmo, how="left")
            if "tmo_general" in dfh.columns:
                dfh[TARGET_TMO_NEW] = dfh["tmo_general"].combine_first(dfh[TARGET_TMO_NEW])
        except Exception as e:
            print(f"WARN: No se pudo cargar o unir {TMO_HIST_FILE}. Error: {e}")
            if TARGET_TMO_NEW not in dfh.columns:
                dfh[TARGET_TMO_NEW] = 0

    # 3) Derivar calendario para el histórico
    holidays_set = load_holidays(HOLIDAYS_FILE)
    if "feriados" not in dfh.columns:
        dfh["feriados"] = mark_holidays_index(dfh.index, holidays_set).values
    dfh["feriados"] = pd.to_numeric(dfh["feriados"], errors="coerce").fillna(0).astype(int)
    if "es_dia_de_pago" not in dfh.columns:
        dfh["es_dia_de_pago"] = add_es_dia_de_pago(dfh).values

    # 4) ffill de columnas clave (Lógica v1)
    # --- ¡LÍNEA CORREGIDA! ---
    # (Se excluye TARGET_TMO_NEW para replicar v1)
    for c in ["feriados", "es_dia_de_pago",
              "proporcion_comercial", "proporcion_tecnica", "tmo_comercial", "tmo_tecnico"]:
        if c in dfh.columns:
            dfh[c] = dfh[c].ffill()
    
    # --- ¡NUEVA LÓGICA DE ORQUESTACIÓN! ---

    # 5) Forecast Llamadas (Lógica v1 Perfecta)
    print("--- Iniciando Inferencia v1 (Llamadas) ---")
    df_hourly_calls = forecast_calls_v1(
        dfh.reset_index(),
        horizon_days=horizonte_dias,
        holidays_set=holidays_set
    )
    print("--- Inferencia v1 (Llamadas) Completada ---")

    # 6) Forecast TMO (Lógica v8 Dinámica)
    print("--- Iniciando Inferencia v8 (TMO) ---")
    future_ts = df_hourly_calls.index 
    
    # Para el TMO v8, SÍ rellenamos el TMO histórico
    if TARGET_TMO_NEW in dfh.columns:
        dfh[TARGET_TMO_NEW] = dfh[TARGET_TMO_NEW].ffill()

    pred_tmo = forecast_tmo_v8(
        dfh.reset_index(), 
        future_ts,         
        holidays_set
    )
    print("--- Inferencia v8 (TMO) Completada ---")

    # 7) Fusión de resultados
    df_hourly = df_hourly_calls.copy()
    df_hourly["tmo_s"] = np.round(pred_tmo).astype(int)

    # 8) Lógica de Erlang
    print("Calculando agentes requeridos (Erlang C)...")
    df_hourly["agents_prod"] = 0
    for ts in df_hourly.index:
        a, _ = required_agents(float(df_hourly.at[ts, "calls"]), float(df_hourly.at[ts, "tmo_s"]))
        df_hourly.at[ts, "agents_prod"] = int(a)
    df_hourly["agents_sched"] = df_hourly["agents_prod"].apply(schedule_agents)

    # 9) Salidas JSON
    print("Generando archivos JSON de salida...")
    write_hourly_json(f"{PUBLIC_DIR}/prediccion_horaria.json",
                      df_hourly, "calls", "tmo_s", "agents_sched")
    write_daily_json(f"{PUBLIC_DIR}/prediccion_diaria.json",
                     df_hourly, "calls", "tmo_s")

    # 10) Alertas clima
    from src.inferencia.alertas_clima import generar_alertas
    generar_alertas(df_hourly[["calls"]])
    
    print("--- Proceso de Inferencia Finalizado ---")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizonte", type=int, default=120)
    args = ap.parse_args()
    main(args.horizonte)
