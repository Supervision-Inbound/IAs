# src/inferencia/features.py
from __future__ import annotations
import unicodedata
import numpy as np
import pandas as pd
from typing import List

TZ = "America/Santiago"

# -----------------------------
# Normalización de columnas
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
    d.columns = [ _norm(c) for c in d.columns ]
    return d

# -----------------------------
# ensure_ts robusto
# -----------------------------
def ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Acepta:
      - Una columna de fecha-hora: ts | timestamp | datatime | datetime
      - Dos columnas: fecha + hora (hora|hora_numero|hora_num|h|time)
        * 'hora' puede venir como 0..23, 'HH', 'HH:MM', 'HH:MM:SS'
    Devuelve DataFrame indexado por 'ts' (DatetimeIndex tz-aware).
    """
    if df is None or df.empty:
        raise ValueError("DataFrame vacío en ensure_ts.")

    d = normalize_columns(df)

    # 1) Si ya viene con índice DatetimeIndex, solo asegurar tz
    if isinstance(d.index, pd.DatetimeIndex):
        idx = d.index
        try:
            if idx.tz is None:
                d.index = idx.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
            else:
                d.index = idx.tz_convert(TZ)
        except Exception:
            # si hay NaT por DST, caer al parse en columna
            d = d.reset_index()

    # 2) Candidatos de columna única
    single_ts_candidates = ["ts", "timestamp", "datatime", "datetime", "fecha_hora", "fechahora"]
    col_single = next((c for c in single_ts_candidates if c in d.columns), None)

    if col_single is not None:
        ts = pd.to_datetime(d[col_single].astype(str), dayfirst=True, errors="coerce")
        d = d.assign(ts=ts).dropna(subset=["ts"])
    else:
        # 3) Candidatos por parejas fecha + hora
        fecha_candidates = ["fecha", "date", "dia", "día"]
        hora_candidates  = ["hora", "hora_numero", "hora_num", "h", "time"]

        c_fecha = next((c for c in fecha_candidates if c in d.columns), None)
        c_hora  = next((c for c in hora_candidates  if c in d.columns), None)

        if c_fecha is None or c_hora is None:
            # último intento: algunos históricos traen 'datatime' con otro nombre
            # o 'datetime' separado raro; si no hay forma, error claro:
            raise ValueError("Se requiere 'ts' o ('fecha' + 'hora'). No se encontró combinación válida.")

        # normalizar 'hora' a string HH:MM:SS
        h = d[c_hora]
        if np.issubdtype(h.dtype, np.number):
            # 0..23 -> "HH:00:00"
            hh = h.astype("Int64").astype(object).where(h.notna(), None)
            h_str = hh.apply(lambda x: f"{int(x):02d}:00:00" if x is not None and pd.notna(x) else None)
        else:
            h_str = h.astype(str).str.strip()
            # "7" -> "07:00:00", "7:3" -> "07:03:00", "7:30"->"07:30:00"
            def _fmt_hour(s: str) -> str:
                s0 = s.replace(",", ".")
                if s0.isdigit():
                    return f"{int(s0):02d}:00:00"
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
                # fallback: dejar como viene y que to_datetime resuelva
                return s
            h_str = h_str.apply(_fmt_hour)

        ts = pd.to_datetime(
            d[c_fecha].astype(str).str.strip() + " " + h_str.astype(str),
            dayfirst=True, errors="coerce"
        )
        d = d.assign(ts=ts).dropna(subset=["ts"])

    # ordenar por ts y fijar índice con TZ
    d = d.sort_values("ts")
    try:
        if getattr(d["ts"].dt, "tz", None) is None:
            d["ts"] = d["ts"].dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
        else:
            d["ts"] = d["ts"].dt.tz_convert(TZ)
    except Exception:
        # si alguna fila cae en NaT por DST, eliminarla
        d = d.dropna(subset=["ts"])

    d = d.set_index("ts")
    return d

# -----------------------------
# Partes de tiempo
# -----------------------------
def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Usar índice si existe; si no, buscar 'ts'
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
    # lags típicos del entrenamiento (ajusta si tu train usó otros):
    for lag in [24, 48, 72, 168]:
        d[f"lag_{lag}"] = d[target_col].shift(lag)
    for win in [24, 72, 168]:
        d[f"ma_{win}"] = d[target_col].shift(1).rolling(win, min_periods=1).mean()
    return d

# -----------------------------
# Reindex a columnas de entrenamiento
# -----------------------------
def dummies_and_reindex_with_scaler_means(df: pd.DataFrame, training_cols: List[str], scaler) -> pd.DataFrame:
    """
    - One-hot de dow, month, hour si existen
    - Reindex a training_cols; las faltantes se rellenan con la media aprendida por el scaler (si se puede),
      y si no, con 0.0 como fallback seguro.
    """
    d = df.copy()
    cat_cols = [c for c in ["dow", "month", "hour"] if c in d.columns]
    if cat_cols:
        d = pd.get_dummies(d, columns=cat_cols, drop_first=False, dtype=float)

    X = d.reindex(columns=training_cols, fill_value=np.nan)

    # Rellenar NaN con medias del scaler si están disponibles
    try:
        means = None
        if hasattr(scaler, "mean_"):
            means = scaler.mean_
        elif hasattr(scaler, "mean"):
            means = scaler.mean
        if means is not None and len(means) == X.shape[1]:
            # usar medias columna a columna
            for i, col in enumerate(X.columns):
                X[col] = X[col].fillna(float(means[i]))
        else:
            X = X.fillna(0.0)
    except Exception:
        X = X.fillna(0.0)

    return X.astype(float)
