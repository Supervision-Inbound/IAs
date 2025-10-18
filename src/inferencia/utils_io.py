# src/inferencia/utils_io.py
import json
import pandas as pd
import numpy as np

TZ = "America/Santiago"

def _to_date_index(idx):
    """Devuelve serie de fecha (date) respetando tz si existe."""
    try:
        return idx.tz_convert(TZ).date
    except Exception:
        return idx.date

def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def write_hourly_json(path, df_hourly, calls_col, tmo_col, staff_col=None):
    out = df_hourly.copy()
    out = out.reset_index().rename(columns={"index": "ts"})
    out["ts"] = out["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    cols = ["ts", calls_col, tmo_col]
    if staff_col and staff_col in out.columns:
        cols.append(staff_col)
    write_json(path, out[cols].to_dict(orient="records"))

def write_daily_json(path, df_hourly, calls_col, tmo_col):
    """
    Agrega por día con TMO ponderado (call-center real):
      - llamadas_diarias = SUM(calls)
      - tmo_diario_s     = SUM(calls * tmo_s) / SUM(calls)
    """
    # Garantizar tipos numéricos
    df = df_hourly[[calls_col, tmo_col]].copy()
    df[calls_col] = pd.to_numeric(df[calls_col], errors="coerce").fillna(0.0)
    df[tmo_col]   = pd.to_numeric(df[tmo_col],   errors="coerce").fillna(0.0)

    # Clave diaria
    df["date"] = _to_date_index(df.index)

    # Sumas por día
    g_calls = df.groupby("date", as_index=False)[calls_col].sum().rename(
        columns={calls_col: "llamadas_diarias"}
    )

    # Ponderación por volumen: SUM(calls * tmo) / SUM(calls)
    df["wx"] = df[calls_col] * df[tmo_col]
    g_w = df.groupby("date", as_index=False).agg(wsum=("wx", "sum"), csum=(calls_col, "sum"))
    daily = g_calls.merge(g_w, on="date", how="left")

    # Evitar división por cero
    denom = daily["csum"].replace(0, np.nan)
    daily["tmo_diario_s"] = (daily["wsum"] / denom).fillna(0).round().astype(int)

    # Formato final
    daily["llamadas_diarias"] = daily["llamadas_diarias"].round().astype(int)
    daily = daily[["date", "llamadas_diarias", "tmo_diario_s"]].sort_values("date")
    daily["date"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")

    write_json(path, daily.to_dict(orient="records"))
