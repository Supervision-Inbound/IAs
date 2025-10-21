# src/main.py
import argparse
import os
import numpy as np
import pandas as pd

from src.inferencia.inferencia_core import forecast_120d
from src.inferencia.features import ensure_ts

from src.data.loader_tmo import load_historico_tmo
from src.inferencia.tmo_from_historico import forecast_tmo_from_historico
from src.inferencia.utils_io import write_hourly_json, write_daily_json

DATA_FILE = "data/historical_data.csv"
HOLIDAYS_FILE = "data/Feriados_Chilev2.csv"
TMO_HIST_FILE = "data/HISTORICO_TMO.csv"

TARGET_CALLS_NEW = "recibidos_nacional"
TARGET_TMO_NEW   = "tmo_general"
TZ = "America/Santiago"

def smart_read_historical(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
        if df.shape[1] > 1: return df
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
        if len(parts) == 2: return float(parts[0])*60   + float(parts[1])
        return float(s)
    except:
        return np.nan

def load_holidays(csv_path: str) -> set:
    if not os.path.exists(csv_path): return set()
    fer = pd.read_csv(csv_path)
    cols_map = {c.lower().strip(): c for c in fer.columns}
    fecha_col = next((cols_map[k] for k in ("fecha","date","dia","día") if k in cols_map), None)
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

    # 1) Histórico principal
    dfh = smart_read_historical(DATA_FILE)
    dfh.columns = dfh.columns.str.strip()

    if TARGET_CALLS_NEW not in dfh.columns:
        for cand in ["recibidos_nacional","recibidos","total_llamadas","llamadas"]:
            if cand in dfh.columns:
                dfh = dfh.rename(columns={cand: TARGET_CALLS_NEW})
                break

    if TARGET_TMO_NEW not in dfh.columns:
        tmo_source = None
        for cand in ["tmo (segundos)","tmo_seg","tmo","tmo_general"]:
            if cand in dfh.columns:
                tmo_source = cand; break
        if tmo_source:
            dfh[TARGET_TMO_NEW] = dfh[tmo_source].apply(parse_tmo_to_seconds)

    dfh = ensure_ts(dfh)

    # 2) Merge del HISTORICO_TMO (si existe)
    if os.path.exists(TMO_HIST_FILE):
        df_tmo = load_historico_tmo(TMO_HIST_FILE)  # index ts y tz listo
        dfh = dfh.join(df_tmo, how="left")
        if "tmo_general" in dfh.columns:
            dfh[TARGET_TMO_NEW] = dfh["tmo_general"].combine_first(dfh[TARGET_TMO_NEW])

    # 3) Calendario
    holidays_set = load_holidays(HOLIDAYS_FILE)
    if "feriados" not in dfh.columns:
        dfh["feriados"] = mark_holidays_index(dfh.index, holidays_set).values
    dfh["feriados"] = pd.to_numeric(dfh["feriados"], errors="coerce").fillna(0).astype(int)
    if "es_dia_de_pago" not in dfh.columns:
        dfh["es_dia_de_pago"] = add_es_dia_de_pago(dfh).values

    # 4) Forzar numérico + ffill para evitar NaN al borde
    for c in [TARGET_TMO_NEW, "feriados", "es_dia_de_pago",
              "proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico",
              "q_llamadas_general","q_llamadas_comercial","q_llamadas_tecnico"]:
        if c in dfh.columns:
            dfh[c] = pd.to_numeric(dfh[c], errors="coerce").ffill()

    # 5) Forecast (planner + TMO interno que luego se sobreescribe)
    df_hourly = forecast_120d(
        dfh.reset_index(),   # forecast_120d llama ensure_ts internamente
        horizon_days=horizonte_dias,
        holidays_set=holidays_set
    )

    # 6) OVERRIDE TMO con HISTORICO_TMO (independiente del modelo)
    future_idx = df_hourly.index
    tmo_future = forecast_tmo_from_historico(
        historico_path=TMO_HIST_FILE,
        future_idx=future_idx,
        lookback_days=90
    )
    df_hourly["tmo_s"] = tmo_future.values.astype(float)

    # 7) Agentes y JSONs finales
    try:
        from src.inferencia.erlang import required_agents
        df_hourly["agentes_requeridos"] = required_agents(
            traffic_calls=df_hourly["calls"].astype(float).values,
            aht_seconds=df_hourly["tmo_s"].astype(float).values
        ).astype(int)
    except Exception:
        df_hourly["agentes_requeridos"] = (df_hourly["calls"] / 20).round().astype(int)

    write_hourly_json("public/prediccion_horaria.json", df_hourly,
                      calls_col="calls", tmo_col="tmo_s", agentes_col="agentes_requeridos")
    write_daily_json("public/prediccion_diaria.json", df_hourly,
                     calls_col="calls", tmo_col="tmo_s")

    # (opcional) alertas clima
    try:
        from src.alertas_clima import generar_alertas
        generar_alertas(df_hourly[["calls"]])
    except Exception as e:
        print("⚠️ Alertas clima no generadas:", e)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizonte", type=int, default=120)
    args = ap.parse_args()
    main(args.horizonte)
