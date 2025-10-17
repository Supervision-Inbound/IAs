# src/main.py
import argparse
import os
import numpy as np
import pandas as pd

from src.inferencia.inferencia_core import forecast_120d
from src.inferencia.features import ensure_ts

DATA_FILE = "data/historical_data.csv"
HOLIDAYS_FILE = "data/Feriados_Chilev2.csv"

TARGET_CALLS_NEW = "recibidos_nacional"
TARGET_TMO_NEW = "tmo_general"
TZ = "America/Santiago"

def smart_read_historical(path: str) -> pd.DataFrame:
    """Lee CSV con autodetección de separador (coma/;), como en la inferencia original."""
    try:
        df = pd.read_csv(path)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    # fallback con ';'
    df = pd.read_csv(path, delimiter=';')
    return df

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
            fecha_col = cols_map[cand]
            break
    if not fecha_col:
        return set()
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

    # 1) Leer histórico y normalizar columnas
    dfh = smart_read_historical(DATA_FILE)
    dfh.columns = dfh.columns.str.strip()

    # Mapear nombres a los esperados por los modelos
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
        dfh[TARGET_TMO_NEW] = dfh[tmo_source].apply(parse_tmo_to_seconds) if tmo_source else np.nan

    print("Cols historical_data.csv:", list(dfh.columns))
    dfh = ensure_ts(dfh)

    # 2) Derivar calendario (feriados, es_dia_de_pago)
    holidays_set = load_holidays(HOLIDAYS_FILE)
    if "feriados" not in dfh.columns:
        dfh["feriados"] = mark_holidays_index(dfh.index, holidays_set).values
    dfh["feriados"] = pd.to_numeric(dfh["feriados"], errors="coerce").fillna(0).astype(int)
    if "es_dia_de_pago" not in dfh.columns:
        dfh["es_dia_de_pago"] = add_es_dia_de_pago(dfh).values

    # 3) CAP de fechas futuras + filtro de llamadas válidas (CLAVE)
    now_tz = pd.Timestamp.now(tz=TZ)
    max_ts_bruto = dfh.index.max()
    dfh = dfh.loc[dfh.index <= now_tz]              # nunca mirar más allá de "ahora"
    max_ts_hoy = dfh.index.max()

    dfh[TARGET_CALLS_NEW] = pd.to_numeric(dfh[TARGET_CALLS_NEW], errors="coerce")
    mask_valid = dfh[TARGET_CALLS_NEW].notna() & (dfh[TARGET_CALLS_NEW] > 0)
    if not mask_valid.any():
        mask_valid = dfh[TARGET_CALLS_NEW].notna()
    dfh = dfh.loc[mask_valid]

    # Forward-fill de auxiliares
    for c in [TARGET_TMO_NEW, "feriados", "es_dia_de_pago"]:
        if c in dfh.columns:
            dfh[c] = dfh[c].ffill()

    last_used = dfh.index.max()
    print("DEBUG fechas")
    print("  max_ts_bruto del CSV:", max_ts_bruto)
    print("  max_ts_<=hoy        :", max_ts_hoy)
    print("  last_ts_valido      :", last_used)
    print("  tail llamadas:")
    try:
        print(dfh[[TARGET_CALLS_NEW]].tail(8))
    except Exception:
        print(dfh.tail(8))

    # 4) Forecast (planner + tmo + erlang) → JSON horario y diario
    df_hourly = forecast_120d(dfh.reset_index(), horizon_days=horizonte_dias)

    # 5) Alertas clima (usa la curva del planner)
    from src.inferencia.alertas_clima import generar_alertas
    generar_alertas(df_hourly[["calls"]])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizonte", type=int, default=120)
    args = ap.parse_args()
    main(args.horizonte)
