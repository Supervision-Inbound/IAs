# src/inferencia/features.py
import numpy as np
import pandas as pd
from typing import Iterable, List

TZ = "America/Santiago"

# ---------------------------
# FECHAS / TIMESTAMPS ROBUSTO
# ---------------------------

def _pick_col(cols: Iterable[str], candidates: Iterable[str]) -> str | None:
    cmap = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        key = cand.lower().strip()
        if key in cmap:
            return cmap[key]
    return None

def _normalize_hour_series(s: pd.Series) -> pd.Series:
    # Recorta a HH:MM y limpia
    return s.astype(str).str.strip().str.slice(0, 5)

def ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un DataFrame indexado por 'ts' (DatetimeIndex con TZ=America/Santiago).
    Casos soportados (idempotente):
      1) Si el índice YA es DatetimeIndex -> no crea 'ts'; normaliza TZ y elimina col 'ts' si duplica.
      2) Si hay columna 'ts' -> la usa como índice (sin duplicar), normaliza TZ.
      3) Si hay ('fecha' + 'hora') -> construye 'ts' y la usa como índice.
    Nunca deja 'ts' como columna si también es índice.
    """
    d = df.copy()

    # Caso 1: índice ya es DatetimeIndex
    if isinstance(d.index, pd.DatetimeIndex):
        # Si además existe una columna 'ts', elimínala para evitar ambigüedad
        if "ts" in d.columns:
            d = d.drop(columns=["ts"])
        # Normalización de TZ
        try:
            if d.index.tz is None:
                d.index = d.index.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
                d = d[~d.index.isna()]
            else:
                # Convertimos a TZ objetivo
                d.index = d.index.tz_convert(TZ)
        except Exception:
            # Si algo falla (DST raro), dejamos el índice como está para no romper
            pass
        # Asegura nombre del índice
        d.index.name = "ts"
        return d.sort_index()

    # Caso 2: hay columna 'ts'
    if "ts" in d.columns:
        ts = pd.to_datetime(d["ts"], errors="coerce", dayfirst=True)
        d = d.drop(columns=["ts"])
        d = d.assign(ts=ts).dropna(subset=["ts"]).sort_values("ts")
        try:
            if d["ts"].dt.tz is None:
                d["ts"] = d["ts"].dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
                d = d.dropna(subset=["ts"])
            else:
                d["ts"] = d["ts"].dt.tz_convert(TZ)
        except Exception:
            pass
        d = d.set_index("ts")
        d.index.name = "ts"
        return d

    # Caso 3: construir desde ('fecha' + 'hora')
    c_fecha = _pick_col(d.columns, ["fecha", "date", "dia", "día"])
    c_hora  = _pick_col(d.columns, ["hora", "hour", "hh"])

    if c_fecha is None or c_hora is None:
        # Último intento: si hubiese columnas con nombres muy raros, fallback a error claro
        raise ValueError("Se requiere 'ts' o ('fecha' + 'hora'). No se encontró combinación válida.")

    h = _normalize_hour_series(d[c_hora])
    ts = pd.to_datetime(d[c_fecha].astype(str) + " " + h, dayfirst=True, errors="coerce")
    d = d.assign(ts=ts).dropna(subset=["ts"]).sort_values("ts")
    try:
        if d["ts"].dt.tz is None:
            d["ts"] = d["ts"].dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
            d = d.dropna(subset=["ts"])
        else:
            d["ts"] = d["ts"].dt.tz_convert(TZ)
    except Exception:
        pass
    d = d.set_index("ts")
    d.index.name = "ts"
    return d

# ---------------------------
# FEATURES DE TIEMPO
# ---------------------------

def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega columnas de partes de tiempo. Admite:
      - index DatetimeIndex, o columna 'ts' si está presente (se usa sin set_index).
    """
    d = df.copy()
    if isinstance(d.index, pd.DatetimeIndex):
        idx = d.index
    elif "ts" in d.columns:
        idx = pd.to_datetime(d["ts"], errors="coerce")
    else:
        raise ValueError("add_time_parts requiere DatetimeIndex o columna 'ts'.")

    d["dow"] = idx.dayofweek
    d["month"] = idx.month
    d["hour"] = idx.hour
    d["day"] = idx.day
    d["sin_hour"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["cos_hour"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["sin_dow"] = np.sin(2 * np.pi * d["dow"] / 7)
    d["cos_dow"] = np.cos(2 * np.pi * d["dow"] / 7)
    return d

def add_lags_mas(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Agrega lags y medias móviles para el 'target_col'.
    Requiere índice temporal ordenado.
    """
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        # si viene con 'ts' columna, establecemos index temporal temporalmente
        if "ts" in d.columns:
            d = d.set_index(pd.to_datetime(d["ts"], errors="coerce"))
        else:
            raise ValueError("add_lags_mas requiere DatetimeIndex o columna 'ts'.")

    d = d.sort_index()
    # Lags típicos
    for lag in [24, 48, 72, 168]:
        d[f"lag_{lag}"] = d[target_col].shift(lag)
    # Medias móviles
    for window in [24, 72, 168]:
        d[f"ma_{window}"] = d[target_col].rolling(window, min_periods=1).mean()

    return d

# ---------------------------
# DUMMIES + REINDEX
# ---------------------------

def dummies_and_reindex(df: pd.DataFrame, training_columns: List[str]) -> pd.DataFrame:
    """
    One-hot de 'dow','month','hour' y reindex a las columnas del entrenamiento.
    Si faltan columnas, las rellena con 0.
    """
    d = df.copy()
    # Aseguramos partes de tiempo
    if not all(c in d.columns for c in ["dow", "month", "hour"]):
        d = add_time_parts(d)

    base_cols = [c for c in d.columns if c.startswith(("lag_", "ma_", "sin_", "cos_"))]
    cat = pd.get_dummies(d[["dow", "month", "hour"]], drop_first=False, dtype=int)
    X = pd.concat([d[base_cols], cat], axis=1)

    # Reindex a orden de entrenamiento
    for c in training_columns:
        if c not in X.columns:
            X[c] = 0
    X = X[training_columns].fillna(0)
    return X

def dummies_and_reindex_with_scaler_means(df: pd.DataFrame, training_columns: List[str], scaler) -> pd.DataFrame:
    """
    Igual que dummies_and_reindex pero, si faltan columnas numéricas
    que el scaler espera, las crea con el valor medio usado al entrenar (si está disponible).
    """
    d = df.copy()
    if not all(c in d.columns for c in ["dow", "month", "hour"]):
        d = add_time_parts(d)

    # intentamos recuperar medias del scaler (StandardScaler)
    scaler_means = None
    try:
        scaler_means = getattr(scaler, "mean_", None)
        scaler_features = getattr(scaler, "feature_names_in_", None)
        if scaler_means is not None and scaler_features is not None:
            scaler_means = dict(zip(list(scaler_features), list(scaler_means)))
    except Exception:
        scaler_means = None

    base_cols = [c for c in d.columns if c.startswith(("lag_", "ma_", "sin_", "cos_")) or c in [
        "feriados", "es_dia_de_pago", "calls",
        "proporcion_comercial","proporcion_tecnica",
        "tmo_comercial","tmo_tecnico",
    ]]

    cat = pd.get_dummies(d[["dow", "month", "hour"]], drop_first=False, dtype=int)
    X = pd.concat([d[base_cols], cat], axis=1)

    for c in training_columns:
        if c not in X.columns:
            if scaler_means and c in scaler_means:
                X[c] = scaler_means[c]
            else:
                X[c] = 0

    X = X[training_columns].infer_objects(copy=False).fillna(0)
    return X
