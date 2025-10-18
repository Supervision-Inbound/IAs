# src/inferencia/utils_io.py
import json
import pandas as pd
import numpy as np

TZ = "America/Santiago"

def _to_date_index(idx):
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
    # incluimos mezcla por categoría si existe (útil para debugs/QA)
    for extra in ["proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico"]:
        if extra in out.columns:
            cols.append(extra)
    write_json(path, out[cols].to_dict(orient="records"))

def write_daily_json(path, df_hourly, calls_col, tmo_col):
    """
    Agregación diaria:
      - Llamadas diarias = SUM(calls)
      - TMO diario:
          * Si existen columnas de mezcla por categoría:
            calls_com = calls * prop_com ; calls_tec = calls * prop_tec
            tmo_diario = SUM(calls_com * tmo_com + calls_tec * tmo_tec) / SUM(calls)
          * Si no existen, usar ponderación simple: SUM(calls * tmo) / SUM(calls)
    """
    df = df_hourly.copy()
    df[calls_col] = pd.to_numeric(df[calls_col], errors="coerce").fillna(0.0)
    df[tmo_col]   = pd.to_numeric(df[tmo_col],   errors="coerce").fillna(0.0)

    df["date"] = _to_date_index(df.index)

    # llamadas diarias
    g_calls = df.groupby("date", as_index=False)[calls_col].sum().rename(
        columns={calls_col: "llamadas_diarias"}
    )

    # ¿tenemos mezcla por categoría?
    has_mix = all(c in df.columns for c in [
        "proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico"
    ])

    if has_mix:
        # asegurar tipos
        for c in ["proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        calls_com = df[calls_col] * df["proporcion_comercial"].clip(0,1)
        calls_tec = df[calls_col] * df["proporcion_tecnica"].clip(0,1)
        wx = (calls_com * df["tmo_comercial"]) + (calls_tec * df["tmo_tecnico"])
    else:
        wx = df[calls_col] * df[tmo_col]

    w = pd.DataFrame({
        "date": df["date"],
        "wsum": wx,
        "csum": df[calls_col]
    }).groupby("date", as_index=False).sum()

    daily = g_calls.merge(w, on="date", how="left")
    denom = daily["csum"].replace(0, np.nan)
    daily["tmo_diario_s"] = (daily["wsum"] / denom).fillna(0).round().astype(int)

    daily["llamadas_diarias"] = daily["llamadas_diarias"].round().astype(int)
    daily = daily[["date", "llamadas_diarias", "tmo_diario_s"]].sort_values("date")
    daily["date"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")

    write_json(path, daily.to_dict(orient="records"))

