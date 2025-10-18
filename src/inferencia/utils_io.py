# src/inferencia/utils_io.py
import json
import numpy as np
import pandas as pd

TIMEZONE = "America/Santiago"

# ---------------------------
# IO helpers
# ---------------------------

def write_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ---------------------------
# utilidades internas
# ---------------------------

def _dates_from_index(idx: pd.DatetimeIndex) -> np.ndarray:
    """
    Devuelve array de fechas (date) desde un DatetimeIndex,
    respetando la TZ si existe.
    """
    try:
        if idx.tz is not None:
            return idx.tz_convert(TIMEZONE).date
    except Exception:
        pass
    return idx.date

def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    """
    Promedio ponderado básico con manejo de casos edge.
    """
    if values.size == 0:
        return None
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    wsum = np.nansum(w)
    if not np.isfinite(wsum) or wsum <= 0:
        # fallback: simple mean si no hay peso válido
        vm = np.nanmean(v) if v.size > 0 else np.nan
        return float(vm) if np.isfinite(vm) else None
    val = np.nansum(v * w) / wsum
    return float(val) if np.isfinite(val) else None

# ---------------------------
# salida diaria (llamadas y TMO)
# ---------------------------

def write_daily_json(
    path: str,
    df_hourly: pd.DataFrame,
    col_calls: str = "calls",
    col_tmo: str = "tmo_s",
):
    """
    Genera JSON diario con:
      - total_llamadas: suma de llamadas del día (col_calls)
      - tmo_general:    promedio ponderado por llamadas del día

    Si existen columnas de mezcla comercial/técnica
    (proporcion_comercial, proporcion_tecnica, tmo_comercial, tmo_tecnico),
    se usa primero TMO por mezcla hora a hora:
        tmo_mix_h = prop_com*tmo_com + prop_tec*tmo_tec
    y luego:
        TMO_diario = sum(calls_h * tmo_mix_h) / sum(calls_h)

    En caso contrario, se usa col_tmo (por defecto "tmo_s") como valor base por hora.
    """
    if not isinstance(df_hourly.index, pd.DatetimeIndex):
        raise ValueError("write_daily_json requiere un DataFrame con DatetimeIndex en el índice.")

    d = df_hourly.copy()

    # --- Comprobar columnas de mezcla por tipo ---
    has_mix = all(c in d.columns for c in [
        "proporcion_comercial", "proporcion_tecnica", "tmo_comercial", "tmo_tecnico"
    ])

    if has_mix:
        # resguardo básico: proporciones dentro de [0,1] y suma ~1
        pc = d["proporcion_comercial"].astype(float).clip(0, 1)
        pt = d["proporcion_tecnica"].astype(float).clip(0, 1)
        s = (pc + pt).replace(0, np.nan)
        # normalizar suave si no suma 1
        pc = np.where(np.isfinite(s), pc / s, 0.5)
        pt = np.where(np.isfinite(s), pt / s, 0.5)

        tmo_com = d["tmo_comercial"].astype(float)
        tmo_tec = d["tmo_tecnico"].astype(float)

        tmo_mix = pc * tmo_com + pt * tmo_tec
        d["_tmo_base"] = tmo_mix
    else:
        d["_tmo_base"] = d[col_tmo].astype(float)

    # llamadas como pesos
    d["_calls_w"] = d[col_calls].astype(float).clip(lower=0)

    # fecha diaria
    d["_date"] = _dates_from_index(d.index)

    # agregación diaria
    out_rows = []
    for fecha, sub in d.groupby("_date", sort=True):
        total_llamadas = float(np.nansum(sub["_calls_w"].values))
        tmo_general = _weighted_mean(sub["_tmo_base"].values, sub["_calls_w"].values)
        # limpieza y formato
        row = {
            "fecha": str(fecha),
            "total_llamadas": int(round(total_llamadas)) if np.isfinite(total_llamadas) else 0,
            "tmo_general": int(round(tmo_general)) if (tmo_general is not None and np.isfinite(tmo_general)) else None,
        }
        out_rows.append(row)

    write_json(path, out_rows)


