# src/inferencia/features.py
import numpy as np
import pandas as pd

TIMEZONE = "America/Santiago"


def ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea 'ts' y devuelve df.set_index('ts').sort_index() normalizado a America/Santiago.
    Acepta:
      - 'ts' directo (cualquier capitalización)
      - 'fecha' + 'hora' (cualquier variante razonable de nombre/caso)
      - una sola columna tipo datetime: 'datetime','datatime','timestamp','fechahora','fecha_hora','fecha y hora'
      - autodetección: si alguna columna se parsea mayoritariamente como datetime
    """
    if df is None or df.empty:
        raise ValueError("DataFrame vacío.")

    # mapa normalizado -> original
    norm = {c.lower().strip().replace("  ", " ").replace(" ", "_"): c for c in df.columns}
    def has(name): return name in norm
    def col(name): return df[norm[name]]

    # 1) 'ts' directo
    if has("ts"):
        out = df.copy()
        out["ts"] = pd.to_datetime(col("ts"), errors="coerce", dayfirst=True)
        out = out.dropna(subset=["ts"]).sort_values("ts")
        ts = out["ts"]
        out["ts"] = ts.dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT") if ts.dt.tz is None else ts.dt.tz_convert(TIMEZONE)
        return out.set_index("ts")

    # 2) 'fecha' + 'hora'
    fecha_like = next((k for k in norm.keys() if k.startswith("fecha")), None)
    hora_like  = next((k for k in norm.keys() if k.startswith("hora")), None)
    if fecha_like and hora_like:
        out = df.copy()
        out["ts"] = pd.to_datetime(
            df[norm[fecha_like]].astype(str) + " " + df[norm[hora_like]].astype(str),
            errors="coerce", dayfirst=True
        )
        out = out.dropna(subset=["ts"]).sort_values("ts")
        ts = out["ts"]
        out["ts"] = ts.dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT") if ts.dt.tz is None else ts.dt.tz_convert(TIMEZONE)
        return out.set_index("ts")

    # 3) candidatos single-datetime
    singles = ["datetime","datatime","timestamp","fecha_hora","fechahora","fecha_y_hora","date_time","datehour","date_hour"]
    for s in singles:
        if has(s):
            out = df.copy()
            out["ts"] = pd.to_datetime(col(s), errors="coerce", dayfirst=True)
            out = out.dropna(subset=["ts"]).sort_values("ts")
            ts = out["ts"]
            out["ts"] = ts.dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT") if ts.dt.tz is None else ts.dt.tz_convert(TIMEZONE)
            return out.set_index("ts")

    # 4) autodetección: columna con >80% parseable
    rates = []
    for c in df.columns:
        parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
        rate = 1.0 - parsed.isna().mean()
        rates.append((c, rate))
    rates.sort(key=lambda x: x[1], reverse=True)
    if rates and rates[0][1] >= 0.8:
        c = rates[0][0]
        out = df.copy()
        out["ts"] = pd.to_datetime(out[c], errors="coerce", dayfirst=True)
        out = out.dropna(subset=["ts"]).sort_values("ts")
        ts = out["ts"]
        out["ts"] = ts.dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT") if ts.dt.tz is None else ts.dt.tz_convert(TIMEZONE)
        return out.set_index("ts")

    raise ValueError(
        "No se encontró columna temporal. Aporta 'ts', o ('fecha'+'hora'), "
        "o una columna tipo 'datetime'/'timestamp'."
    )


def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega partes de tiempo (usa columna 'ts' si existe, si no intenta 'index')."""
    work = df.copy()
    if "ts" in work.columns:
        ts = pd.to_datetime(work["ts"], errors="coerce")
    else:
        ts = pd.to_datetime(work.index, errors="coerce")
        work = work.reset_index().rename(columns={"index": "ts"})
    ts = ts.dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT") if ts.dt.tz is None else ts.dt.tz_convert(TIMEZONE)
    work["ts"] = ts
    work["dow"] = work["ts"].dt.dayofweek
    work["month"] = work["ts"].dt.month
    work["hour"] = work["ts"].dt.hour
    work["day"] = work["ts"].dt.day
    work["sin_hour"] = np.sin(2 * np.pi * work["hour"] / 24.0)
    work["cos_hour"] = np.cos(2 * np.pi * work["hour"] / 24.0)
    work["sin_dow"] = np.sin(2 * np.pi * work["dow"] / 7.0)
    work["cos_dow"] = np.cos(2 * np.pi * work["dow"] / 7.0)
    return work


def add_lags_mas(df: pd.DataFrame, target_col: str,
                 lags=(24, 48, 72, 168), mas=(24, 72, 168)) -> pd.DataFrame:
    work = df.copy()
    for lag in lags:
        work[f"lag_{lag}"] = work[target_col].shift(lag)
    for w in mas:
        work[f"ma_{w}"] = work[target_col].rolling(w, min_periods=1).mean()
    return work


def dummies_and_reindex(df_row: pd.DataFrame, cols_expected,
                        dummies_cols=('dow','month','hour')) -> pd.DataFrame:
    """
    One-hot de columnas categóricas y reindex a columnas esperadas por el modelo.
    """
    X = pd.get_dummies(df_row.copy(), columns=list(dummies_cols), drop_first=False)
    # Por si el modelo esperaba columnas que no están en esta fila
    missing = [c for c in cols_expected if c not in X.columns]
    for c in missing:
        X[c] = 0
    # Y por si esta fila trae columnas que el modelo no tenía
    X = X.reindex(columns=cols_expected, fill_value=0)
    return X

