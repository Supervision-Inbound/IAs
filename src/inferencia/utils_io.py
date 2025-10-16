# src/inferencia/utils_io.py
import json, os
import pandas as pd

def write_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def write_hourly_json(path: str, df_hourly: pd.DataFrame, calls_col: str, tmo_col: str, agentes_col: str):
    out = (df_hourly.reset_index()
                   .rename(columns={"index":"ts", calls_col:"llamadas_hora", tmo_col:"tmo_hora", agentes_col:"agentes_requeridos"}))
    out["ts"] = pd.to_datetime(out["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    write_json(path, out.to_dict(orient="records"))

def write_daily_json(path: str, df_hourly: pd.DataFrame, calls_col: str, tmo_col: str):
    tmp = (df_hourly.reset_index()
                     .rename(columns={"index":"ts"})
                     .assign(fecha=lambda d: pd.to_datetime(d["ts"]).dt.date)
                     .groupby("fecha", as_index=False)
                     .agg(llamadas_diarias=(calls_col,"sum"),
                          tmo_diario=(tmo_col,"mean")))
    write_json(path, tmp.to_dict(orient="records"))

