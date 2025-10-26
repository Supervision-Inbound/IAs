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
    # Asegurar que ts es string serializable
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
    out = out.dropna(subset=["ts"])
    out["ts"] = out["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Tipos serializables
    out["llamadas_hora"] = out["llamadas_hora"].astype(int)
    out["tmo_hora"] = out["tmo_hora"].astype(float)
    out["agentes_requeridos"] = out["agentes_requeridos"].astype(int)

    write_json(path, out.to_dict(orient="records"))

def write_daily_json(path: str, df_hourly: pd.DataFrame, calls_col: str, tmo_col: str):
    tmp = (df_hourly.reset_index()
                     .rename(columns={"index": "ts"}))

    # Asegurar que 'ts' es datetime (puede venir como string del index)
    tmp["ts"] = pd.to_datetime(tmp["ts"], errors="coerce")
    tmp = tmp.dropna(subset=["ts"])

    # Construir 'fecha' y convertir a string para JSON
    tmp["fecha"] = tmp["ts"].dt.date.astype(str)

    # Agregar por día: suma de llamadas, promedio de TMO
    daily = (tmp.groupby("fecha", as_index=False)
                .agg(llamadas_diarias=(calls_col, "sum"),
                     tmo_diario=(tmo_col, "mean")))

    # Asegurar tipos serializables
    daily["llamadas_diarias"] = daily["llamadas_diarias"].astype(int)
    daily["tmo_diario"] = daily["tmo_diario"].astype(float)

    write_json(path, daily.to_dict(orient="records"))
