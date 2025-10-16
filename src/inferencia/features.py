# src/inferencia/features.py
import numpy as np
import pandas as pd

def ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "ts" not in d.columns:
        if "fecha" in d.columns and "hora" in d.columns:
            d["ts"] = pd.to_datetime(d["fecha"].astype(str) + " " + d["hora"].astype(str), errors="coerce")
        else:
            raise ValueError("Se requiere 'ts' o ('fecha' + 'hora').")
    d = d.dropna(subset=["ts"]).sort_values("ts")
    if d["ts"].dt.tz is None:
        d["ts"] = d["ts"].dt.tz_localize("America/Santiago", nonexistent="NaT", ambiguous="NaT")
        d = d.dropna(subset=["ts"])
    return d.set_index("ts")

def add_time_parts(df_idx: pd.DataFrame) -> pd.DataFrame:
    d = df_idx.copy()
    d["dow"] = d.index.dayofweek
    d["month"] = d.index.month
    d["hour"] = d.index.hour
    d["sin_hour"] = np.sin(2*np.pi*d["hour"]/24)
    d["cos_hour"] = np.cos(2*np.pi*d["hour"]/24)
    d["sin_dow"] = np.sin(2*np.pi*d["dow"]/7)
    d["cos_dow"] = np.cos(2*np.pi*d["dow"]/7)
    return d

def add_lags_mas(d: pd.DataFrame, target: str) -> pd.DataFrame:
    x = d.copy()
    for lag in [24,48,72,168]:
        x[f"lag_{lag}"] = x[target].shift(lag)
    for win in [24,72,168]:
        x[f"ma_{win}"] = x[target].rolling(win, min_periods=1).mean()
    return x

def dummies_and_reindex(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    dd = pd.get_dummies(df, columns=["dow","month","hour"], drop_first=False)
    return dd.reindex(columns=cols, fill_value=0).fillna(0)

