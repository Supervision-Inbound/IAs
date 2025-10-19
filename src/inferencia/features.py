# src/inferencia/features.py
import numpy as np
import pandas as pd

TIMEZONE = "America/Santiago"

def ensure_ts(df):
    d = df.copy()
    d.columns = [c.strip() for c in d.columns]
    # si viene ts, úsalo
    if "ts" in d.columns:
        ts = pd.to_datetime(d["ts"], errors="coerce")
    else:
        # busca 'fecha' y 'hora'
        c_fecha = next((c for c in d.columns if c.lower().strip() in ("fecha","date","dia","día")), None)
        c_hora  = next((c for c in d.columns if "hora" in c.lower()), None)
        if not c_fecha or not c_hora:
            raise ValueError("Se requiere 'ts' o ('fecha' + 'hora').")
        h = d[c_hora].astype(str).str.slice(0,5)
        ts = pd.to_datetime(d[c_fecha].astype(str) + " " + h, dayfirst=True, errors="coerce")
    d = d.assign(ts=ts).dropna(subset=["ts"]).sort_values("ts")
    d["ts"] = d["ts"].dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
    d = d.dropna(subset=["ts"])
    return d.set_index("ts")

def add_time_parts(df):
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        raise ValueError("add_time_parts requiere DatetimeIndex")
    d["dow"] = d.index.dayofweek
    d["month"] = d.index.month
    d["hour"] = d.index.hour
    d["sin_hour"] = np.sin(2*np.pi*d["hour"]/24)
    d["cos_hour"] = np.cos(2*np.pi*d["hour"]/24)
    d["sin_dow"]  = np.sin(2*np.pi*d["dow"]/7)
    d["cos_dow"]  = np.cos(2*np.pi*d["dow"]/7)
    return d

def add_lags_mas(df, target_col):
    d = df.copy()
    d[f"lag_24"] = d[target_col].shift(24)
    d[f"ma_24"]  = d[target_col].shift(1).rolling(24, min_periods=1).mean()
    d[f"ma_168"] = d[target_col].shift(1).rolling(168, min_periods=1).mean()
    return d

def dummies_and_reindex(df_row, training_cols):
    d = df_row.copy()
    d = pd.get_dummies(d, columns=['dow','month','hour'], drop_first=False)
    for c in training_cols:
        if c not in d.columns:
            d[c] = 0.0
    d = d.reindex(columns=training_cols, fill_value=0.0)
    return d.fillna(0.0)

# ===== Solo para TMO: usa medias del scaler cuando falten columnas =====
def dummies_and_reindex_with_scaler_means(df_row, training_cols, scaler):
    d = df_row.copy()
    d = pd.get_dummies(d, columns=['dow','month','hour'], drop_first=False)

    scaler_means = {}
    if hasattr(scaler, "mean_"):
        for name, mu in zip(training_cols, scaler.mean_):
            scaler_means[name] = float(mu)

    for c in training_cols:
        if c not in d.columns:
            d[c] = scaler_means.get(c, 0.0)

    d = d.reindex(columns=training_cols, fill_value=0.0)
    return d.fillna(0.0)
