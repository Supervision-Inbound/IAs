# src/main.py
import argparse
import os
import numpy as np
import pandas as pd

from src.inferencia.inferencia_core import forecast_120d
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
    os.makedirs("public", exist_ok=True)

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
    df_tmo_hist_only = None  # <-- NUEVA LÍNEA: Variable para TMO puro
    if os.path.exists(TMO_HIST_FILE):
        df_tmo = load_historico_tmo(TMO_HIST_FILE)  # index ts
        df_tmo_hist_only = df_tmo.copy()  # <-- NUEVA LÍNEA: Guardamos la data pura
        
        # left-join sobre dfh (mantiene las horas que usa el planner)
        dfh = dfh.join(df_tmo, how="left")
        # si tmo_general llegó desde TMO_HIST, úsalo como autoridad
        if "tmo_general" in dfh.columns:
            dfh[TARGET_TMO_NEW] = dfh["tmo_general"].combine_first(dfh[TARGET_TMO_NEW])

    # 3) Derivar calendario para el histórico
    holidays_set = load_holidays(HOLIDAYS_FILE)
    if "feriados" not in dfh.columns:
        dfh["feriados"] = mark_holidays_index(dfh.index, holidays_set).values
    dfh["feriados"] = pd.to_numeric(dfh["feriados"], errors="coerce").fillna(0).astype(int)
    if "es_dia_de_pago" not in dfh.columns:
        dfh["es_dia_de_pago"] = add_es_dia_de_pago(dfh).values

    # 4) ffill de columnas clave para evitar NaN en el borde
    for c in [TARGET_TMO_NEW, "feriados", "es_dia_de_pago",
              "proporcion_comercial", "proporcion_tecnica", "tmo_comercial", "tmo_tecnico"]:
        if c in dfh.columns:
            dfh[c] = dfh[c].ffill()

    # 5) Forecast (le pasamos feriados y el DF de TMO puro)
    df_hourly = forecast_120d(
        dfh.reset_index(),
        # <-- INICIO BLOQUE MODIFICADO -->
        df_tmo_hist_only.reset_index() if df_tmo_hist_only is not None else None,
        # <-- FIN BLOQUE MODIFICADO -->
        horizon_days=horizonte_dias,
        holidays_set=holidays_set
    )

    # 6) Alertas clima (usa la curva del planner)
    from src.inferencia.alertas_clima import generar_alertas
    generar_alertas(df_hourly[["calls"]])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizonte", type=int, default=120)
    args = ap.parse_args()
    main(args.horizonte)
