# src/inferencia/utils_io.py
import json
import pandas as pd
import numpy as np

def _to_date_index(idx):
    """Devuelve serie de fecha (date) respetando tz si existe."""
    try:
        return idx.tz_convert("America/Santiago").date
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
    Agrupa por día:
      - llamadas_diarias = SUM(calls)
      - tmo_diario_s     = SUM(calls * tmo_s) / MAX(SUM(calls), 1)
    IMPORTANTE: TMO ponderado por volumen, no promedio simple.
    """
    df = df_hourly[[calls_col, tmo_col]].copy()
    df["date"] = _to_date_index(df.index)
    # Totales por día
    g = df.groupby("date", as_index=False).agg(
        llamadas_diarias=(calls_col, "sum"),
        _wsum=("dummy", "size")  # placeholder para que agg no quede vacío
    )
    g = g.drop(columns=["_wsum"])

    # Ponderación por volumen
    df["wx"] = df[calls_col].astype(float) * df[tmo_col].astype(float)
    w = df.groupby("date", as_index=False).agg(
        wsum=("wx", "sum"),
        csum=(calls_col, "sum")
    )
    daily = g.merge(w, on="date", how="left")
    daily["tmo_diario_s"] = (daily["wsum"] / daily["csum"].replace(0, np.nan)).fillna(0).round().astype(int)

    # Redondeos/formatos finales
    daily["llamadas_diarias"] = daily["llamadas_diarias"].round().astype(int)
    daily = daily[["date", "llamadas_diarias", "tmo_diario_s"]].sort_values("date")
    daily["date"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")

    write_json(path, daily.to_dict(orient="records"))
