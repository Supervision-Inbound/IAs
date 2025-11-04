# src/main.py
import argparse
import os
import numpy as np
import pandas as pd
import sys

# Asegurar que 'src' esté en el path (importante para GitHub Actions)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inferencia.inferencia_core import forecast_120d
from inferencia.features import ensure_ts

DATA_FILE = "data/historical_data.csv"
HOLIDAYS_FILE = "data/Feriados_Chilev2.csv"

# ======= Claves de negocio =======
TARGET_CALLS_NEW = "recibidos_nacional"
TARGET_TMO_NEW = "tmo_general"
FERIADOS_COL = "feriados" 
TZ = "America/Santiago"

def smart_read_historical(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
        if df.shape[1] > 1: return df
    except Exception: pass
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
        if cand in cols_map: fecha_col = cols_map[cand]; break
    if not fecha_col: return set()
    fechas = pd.to_datetime(fer[fecha_col].astype(str), dayfirst=True, errors="coerce").dropna().dt.date
    return set(fechas)

def mark_holidays_index(dt_index, holidays_set: set) -> pd.Series:
    tz = getattr(dt_index, "tz", None)
    idx_dates = dt_index.tz_convert(TZ).date if tz is not None else dt_index.date
    return pd.Series([d in holidays_set for d in idx_dates], index=dt_index, dtype=int, name=FERIADOS_COL)


def main(horizonte_dias: int):
    os.makedirs("public", exist_ok=True)

    # 1) Leer histórico
    dfh = smart_read_historical(DATA_FILE)
    dfh.columns = dfh.columns.str.strip()

    # 2) Normalizar volumen
    if TARGET_CALLS_NEW not in dfh.columns:
        for cand in ["recibidos_nacional", "recibidos", "contestados", "total_llamadas", "llamadas", "target"]:
            if cand in dfh.columns:
                dfh = dfh.rename(columns={cand: TARGET_CALLS_NEW})
                break
    
    # 🚨 RENOMBRADO VITAL: La V7 espera la columna "target"
    if TARGET_CALLS_NEW in dfh.columns:
        dfh = dfh.rename(columns={TARGET_CALLS_NEW: "target"})
    else:
        # Si no existe, la creamos (aunque la V7-LSTM la necesita)
        if "target" not in dfh.columns:
            dfh["target"] = 0 

    # 3) Normalizar TMO
    if TARGET_TMO_NEW not in dfh.columns:
        tmo_source = None
        for cand in ["tmo_general", "tmo (segundos)", "tmo (s)", "tmo_seg", "tmo", "aht"]:
            if cand in dfh.columns: tmo_source = cand; break
        if tmo_source:
            dfh[TARGET_TMO_NEW] = dfh[tmo_source].apply(parse_tmo_to_seconds)

    # 4) Asegurar índice temporal (features.py lo limpia y maneja DST)
    dfh = ensure_ts(dfh)

    # 5) Derivar calendario (feriados) - Necesario para V7
    holidays_set = load_holidays(HOLIDAYS_FILE)
    if FERIADOS_COL not in dfh.columns:
        dfh[FERIADOS_COL] = mark_holidays_index(dfh.index, holidays_set).values
    dfh[FERIADOS_COL] = pd.to_numeric(dfh[FERIADOS_COL], errors="coerce").fillna(0).astype(int)
    
    dfh = dfh.sort_index()
    dfh['es_post_feriado'] = ((dfh[FERIADOS_COL].shift(1).fillna(0) == 1) & (dfh[FERIADOS_COL] == 0)).astype(int)
    dfh['es_pre_feriado'] = ((dfh[FERIADOS_COL].shift(-1).fillna(0) == 1) & (dfh[FERIADOS_COL] == 0)).astype(int)

    # 6) ffill columnas clave
    for c in [TARGET_TMO_NEW, FERIADOS_COL, 'es_pre_feriado', 'es_post_feriado']:
        if c in dfh.columns:
            dfh[c] = dfh[c].ffill()

    # 7) Forecast
    # 🚨 CORRECCIÓN DEL TypeError: 
    # La V7 (inferencia_core) no acepta 'horizon_days' ni 'holidays_set' directamente.
    # 'horizon_days' lo ignora (usa el de config) y 'holidays_set' lo espera en dfh.
    df_hourly = forecast_120d(
        dfh
        # 'artifacts_path' usará el default de config.
    )

    # 8) Alertas clima (Asumiendo que alertas_clima.py existe)
    # from inferencia.alertas_clima import generar_alertas
    # generar_alertas(df_hourly[["calls"]]) # V7 devuelve 'target_pred', no 'calls'
    
    # (Opcional: Renombrar 'target_pred' a 'calls' si el resto del pipeline lo espera)
    df_hourly = df_hourly.rename(columns={"target_pred": "calls"})
    print("Inferencia completada.")
    # print(df_hourly.head())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizonte", type=int, default=120)
    args = ap.parse_args()
    
    # 🚨 NOTA: El argumento 'horizonte' no se está usando en la V7.
    # Si necesitas que se use, la V7 (inferencia_core) debe ser modificada 
    # para aceptar 'horizon_days' (como en mi penúltima respuesta).
    
    main(args.horizonte)
