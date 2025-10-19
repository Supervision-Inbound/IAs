# src/inferencia/features.py
from __future__ import annotations
import os, unicodedata, json
import numpy as np
import pandas as pd
from typing import List

TZ = "America/Santiago"
DEBUG_PATH = "public/ensure_ts_debug.json"

# -----------------------------
# utilidades
# -----------------------------
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def _norm(c: str) -> str:
        c = _strip_accents(str(c)).lower().strip()
        c = c.replace("/", " ").replace("-", " ").replace(".", " ")
        c = "_".join(c.split())
        return c
    d = df.copy()
    d.columns = [_norm(c) for c in d.columns]
    return d

def _write_debug(payload: dict):
    try:
        os.makedirs("public", exist_ok=True)
        with open(DEBUG_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass

# -----------------------------
# ensure_ts robusto + diagnóstico
# -----------------------------
def ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reglas:
      1) Si hay índice DatetimeIndex, se usa (ajustando TZ).
      2) Intento A: columna única tipo timestamp en {'ts','timestamp','datatime','datetime','fechahora','fecha_hora'}.
         Si parsea < 10 filas válidas, se considera fallo y se intenta B.
      3) Intento B: pareja fecha + hora. 'fecha' = cualquier col que contenga 'fecha' o 'date'.
         'hora' = cualquier col que contenga 'hora' o 'hour'.
         Normaliza horas numéricas y cadenas a 'HH:MM:SS'.
    Devuelve DF indexado por ts (tz=America/Santiago).
    """
    if df is None or df.empty:
        raise ValueError("DataFrame vacío en ensure_ts.")

    raw_cols = list(df.columns)
    d = normalize_columns(df)
    ncols = list(d.columns)

    debug = {
        "raw_columns": raw_cols,
        "normalized_columns": ncols,
        "path_chosen": None,
        "single_col_candidate": None,
        "single_valid_rows": None,
        "pair_date_candidates": [],
        "pair_hour_candidates": [],
        "pair_chosen": None,
        "pair_valid_rows": None,
    }

    # 0) Si ya viene con índice datetime
    if isinstance(d.index, pd.DatetimeIndex):
        idx = d.index
        try:
            if idx.tz is None:
                d.index = idx.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
            else:
                d.index = idx.tz_convert(TZ)
        except Exception:
            d = d.reset_index()  # volvemos a columnas para reprocesar
        else:
            debug["path_chosen"] = "index_already_datetime"
            _write_debug(debug)
            return d

    # 1) Intento A: columna única timestamp
    single_ts_candidates = ["ts", "timestamp", "datatime", "datetime", "fechahora", "fecha_hora"]
    col_single = next((c for c in single_ts_candidates if c in d.columns), None)
    debug["single_col_candidate"] = col_single

    if col_single is not None:
        tsA = pd.to_datetime(d[col_single].astype(str), dayfirst=True, errors="coerce")
        dA = d.assign(ts=tsA).dropna(subset=["ts"]).copy()
        validA = int(dA.shape[0])
        debug["single_valid_rows"] = validA
        if validA >= 10:  # umbral mínimo razonable para aceptar esta ruta
            dA = dA.sort_values("ts")
            try:
                if getattr(dA["ts"].dt, "tz", None) is None:
                    dA["ts"] = dA["ts"].dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
                else:
                    dA["ts"] = dA["ts"].dt.tz_convert(TZ)
            except Exception:
                dA = dA.dropna(subset=["ts"])
            dA = dA.set_index("ts")
            debug["path_chosen"] = "single_column"
            _write_debug(debug)
            return dA
        # si no cumple filas válidas, seguimos al intento B

    # 2) Intento B: pareja fecha + hora
    # candidatos amplios (cualquier col que contenga 'fecha' o 'date', y 'hora' o 'hour')
    date_cands = [c for c in d.columns if ("fecha" in c) or (c.startswith("date")) or (c == "date")]
    hour_cands = [c for c in d.columns if ("hora" in c) or (c == "hour") or (c.startswith("hour"))]
    debug["pair_date_candidates"] = date_cands
    debug["pair_hour_candidates"] = hour_cands

    if date_cands and hour_cands:
        # heurística: preferir 'fecha' exacta y 'hora_numero' si existen
        c_fecha = "fecha" if "fecha" in date_cands else date_cands[0]
        if "hora_numero" in hour_cands:
            c_hora = "hora_numero"
        elif "hora" in hour_cands:
            c_hora = "hora"
        else:
            c_hora = hour_cands[0]

        debug["pair_chosen"] = {"fecha": c_fecha, "hora": c_hora}

        h = d[c_hora]
        # normalizar horas a "HH:MM:SS"
        if np.issubdtype(h.dtype, np.number):
            hh = h.astype("Int64").astype(object).where(h.notna(), None)
            h_str = hh.apply(lambda x: f"{int(x):02d}:00:00" if x is not None and pd.notna(x) else None)
        else:
            h_str = h.astype(str).str.strip()
            def _fmt_hour(s: str) -> str:
                s0 = s.replace(",", ".")
                if s0.isdigit():
                    return f"{int(float(s0)):02d}:00:00"
                parts = s0.split(":")
                try:
                    if len(parts) == 1:
                        return f"{int(float(parts[0])):02d}:00:00"
                    if len(parts) == 2:
                        return f"{int(parts[0]):02d}:{int(parts[1]):02d}:00"
                    if len(parts) >= 3:
                        return f"{int(parts[0]):02d}:{int(parts[1]):02d}:{int(float(parts[2])):02d}"
                except Exception:
                    pass
                return s
            h_str = h_str.apply(_fmt_hour)

        tsB = pd.to_datetime(
            d[c_fecha].astype(str).str.strip() + " " + h_str.astype(str),
            dayfirst=True, errors="coerce"
        )
        dB = d.assign(ts=tsB).dropna(subset=["ts"]).copy()
        validB = int(dB.shape[0])
        debug["pair_valid_rows"] = validB

        if validB == 0 and col_single is not None:
            # fallback: si la columna única tenía algo, úsala aunque sea corta
            tsA = pd.to_datetime(d[col_single].astype(str), dayfirst=True, errors="coerce")
            dA = d.assign(ts=tsA).dropna(subset=["ts"]).copy()
            validA = int(dA.shape[0])
            if validA > 0:
                dA = dA.sort_values("ts")
                try:
                    if getattr(dA["ts"].dt, "tz", None) is None:
                        dA["ts"] = dA["ts"].dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
                    else:
                        dA["ts"] = dA["ts"].dt.tz_convert(TZ)
                except Exception:
                    dA = dA.dropna(subset=["ts"])
                dA = dA.set_index("ts")
                debug["path_chosen"] = "single_column_fallback"
                _write_debug(debug)
                return dA

        if validB > 0:
            dB = dB.sort_values("ts")
            try:
                if getattr(dB["ts"].dt, "tz", None) is None:
                    dB["ts"] = dB["ts"].dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
                else:
                    dB["ts"] = dB["ts"].dt.tz_convert(TZ)
            except Exception:
                dB = dB.dropna(subset=["ts"])
            dB = dB.set_index("ts")
            debug["path_chosen"] = "pair_fecha_hora"
            _write_debug(debug)
            return dB

    debug["path_chosen"] = "error"
    _write_debug(debug)
    raise ValueError("Se requiere 'ts' o ('fecha' + 'hora'). No se encontró combinación válida.")

# -----------------------------
# Partes de tiempo
# -----------------------------
def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    idx = d.index if isinstance(d.index, pd.DatetimeIndex) else pd.to_datetime(d["ts"], errors="coerce")
    d["dow"]   = idx.dayofweek
    d["month"] = idx.month
    d["hour"]  = idx.hour
    d["day"]   = idx.day

    d["sin_hour"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["cos_hour"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["sin_dow"]  = np.sin(2 * np.pi * d["dow"]  / 7)
    d["cos_dow"]  = np.cos(2 * np.pi * d["dow"]  / 7)
    return d

# -----------------------------
# Lags y medias móviles
# -----------------------------
def add_lags_mas(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    d = df.copy()
    for lag in [24, 48, 72, 168]:
        d[f"lag_{lag}"] = d[target_col].shift(lag)
    for win in [24, 72, 168]:
        d[f"ma_{win}"] = d[target_col].shift(1).rolling(win, min_periods=1).mean()
    return d

# -----------------------------
# Reindex a columnas de entrenamiento
# -----------------------------
def dummies_and_reindex_with_scaler_means(df: pd.DataFrame, training_cols: List[str], scaler) -> pd.DataFrame:
    d = df.copy()
    cat_cols = [c for c in ["dow", "month", "hour"] if c in d.columns]
    if cat_cols:
        d = pd.get_dummies(d, columns=cat_cols, drop_first=False, dtype=float)

    X = d.reindex(columns=training_cols, fill_value=np.nan)

    try:
        means = None
        if hasattr(scaler, "mean_"):
            means = scaler.mean_
        elif hasattr(scaler, "mean"):
            means = scaler.mean
        if means is not None and len(means) == X.shape[1]:
            for i, col in enumerate(X.columns):
                X[col] = X[col].fillna(float(means[i]))
        else:
            X = X.fillna(0.0)
    except Exception:
        X = X.fillna(0.0)

    return X.astype(float)

