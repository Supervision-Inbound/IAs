# src/main.py
import argparse, os, json
import numpy as np
import pandas as pd

from src.inferencia.features import ensure_ts
from src.inferencia.utils_io import write_json
from src.inferencia_llamadas.forecast import forecast_llamadas
from src.inferencia_tmo.forecast import forecast_tmo
from src.data.loader_tmo import load_historico_tmo  # tu loader real de TMO

DATA_FILE      = "data/historical_data.csv"
HOLIDAYS_FILE  = "data/Feriados_Chilev2.csv"
TMO_HIST_FILE  = "data/TMO_HISTORICO.csv"
TZ = "America/Santiago"

def smart_read_historical(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
        if df.shape > (0,1):
            return df
    except Exception:
        pass
    return pd.read_csv(path, delimiter=';', low_memory=False)

def load_holidays(csv_path: str) -> set:
    if not os.path.exists(csv_path): return set()
    fer = pd.read_csv(csv_path)
    cols_map = {c.lower().strip(): c for c in fer.columns}
    fecha_col = None
    for cand in ["fecha","date","dia","día"]:
        if cand in cols_map:
            fecha_col = cols_map[cand]; break
    if not fecha_col: return set()
    fechas = pd.to_datetime(fer[fecha_col].astype(str), dayfirst=True, errors="coerce").dropna().dt.date
    return set(fechas)

def main(horizonte_dias: int):
    os.makedirs("public", exist_ok=True)

    # 1) Llamadas: leer y pronosticar SOLO con el planner
    dfh = smart_read_historical(DATA_FILE)
    dfh.columns = dfh.columns.str.strip()
    dfh = ensure_ts(dfh)  # aquí se detecta fecha/hora o ts

    df_calls_fut = forecast_llamadas(dfh, horizon_days=horizonte_dias)  # -> public/llamadas_*.json

    # 2) TMO: leer histórico TMO y pronosticar SOLO TMO (usa llamadas futuras como insumo)
    df_tmo_hist = load_historico_tmo(TMO_HIST_FILE) if os.path.exists(TMO_HIST_FILE) else pd.DataFrame()
    df_tmo_fut = forecast_tmo(df_calls_fut, df_tmo_hist)

    # 3) (Opcional) empaquetar una salida combinada horario/diario para compatibilidad
    combined_hourly = df_calls_fut.join(df_tmo_fut, how="left").reset_index() \
                        .rename(columns={"index":"ts"})
    combined_hourly["ts"] = combined_hourly["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    write_json("public/prediccion_horaria.json", combined_hourly.to_dict(orient="records"))

    # diario combinado
    daily = (combined_hourly.assign(date=pd.to_datetime(combined_hourly["ts"]).dt.date)
             .groupby("date", as_index=False)
             .agg(total_llamadas=("calls","sum"),
                  tmo_general=("tmo_s","mean")))
    # tmo diario ponderado por llamadas
    try:
        tmp = combined_hourly.copy()
        tmp["date"] = pd.to_datetime(tmp["ts"]).dt.date
        g = tmp.groupby("date").apply(
            lambda gdf: int(round(np.nansum(gdf["tmo_s"]*gdf["calls"])/max(1,np.nansum(gdf["calls"])) ))
        ).reset_index(name="tmo_ponderado")
        daily = daily.merge(g, on="date", how="left")
        daily["tmo_general"] = daily["tmo_ponderado"].fillna(daily["tmo_general"]).astype(float)
        daily = daily.drop(columns=["tmo_ponderado"])
    except Exception:
        pass

    write_json("public/prediccion_diaria.json",
               daily.rename(columns={"date":"fecha"}).to_dict(orient="records"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizonte", type=int, default=120)
    args = ap.parse_args()
    main(args.horizonte)
