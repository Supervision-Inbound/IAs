# src/inferencia/tmo_ar_inference.py
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Mapping, Sequence, Tuple

def _mad_cap(series: pd.Series, k: float = 3.5) -> pd.Series:
    med = series.median()
    mad = (series - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return series.clip(lower=series.quantile(0.01), upper=series.quantile(0.99))
    lower = med - k * 1.4826 * mad
    upper = med + k * 1.4826 * mad
    return series.clip(lower=lower, upper=upper)

def _clamp(series: pd.Series, low: float, high: float) -> pd.Series:
    return series.clip(lower=low, upper=high)

def _time_parts(ts: pd.Timestamp):
    dow   = ts.dayofweek
    month = ts.month
    hour  = ts.hour
    sin_hour = float(np.sin(2 * np.pi * hour / 24))
    cos_hour = float(np.cos(2 * np.pi * hour / 24))
    sin_dow  = float(np.sin(2 * np.pi * dow / 7))
    cos_dow  = float(np.cos(2 * np.pi * dow / 7))
    day = ts.day
    es_dia_de_pago = int(day in [1, 2, 15, 16, 29, 30, 31])
    return {
        "dow": dow, "month": month, "hour": hour, "day": day,
        "sin_hour": sin_hour, "cos_hour": cos_hour,
        "sin_dow": sin_dow, "cos_dow": cos_dow,
        "es_dia_de_pago": es_dia_de_pago,
    }

def _get_from_series(series: Mapping[pd.Timestamp, float] | pd.Series, key: pd.Timestamp):
    if hasattr(series, "get"):
        return series.get(key, np.nan)  # type: ignore
    try:
        return series.loc[key]  # type: ignore
    except Exception:
        return np.nan

def _row_features_tmo(
    ts: pd.Timestamp,
    tmo_hist: Mapping[pd.Timestamp, float],
    calls_at_ts,
    feriado_at_ts,
    prop_comercial,
    prop_tecnica,
):
    def lag(h: int):
        key = ts - pd.Timedelta(hours=h)
        val = _get_from_series(tmo_hist, key)
        return float(val) if pd.notna(val) else np.nan

    def rolling_mean(win_h: int):
        end = ts - pd.Timedelta(hours=1)
        vals = []
        for i in range(win_h):
            key = end - pd.Timedelta(hours=i)
            v = _get_from_series(tmo_hist, key)
            if pd.notna(v):
                vals.append(float(v))
        return float(np.mean(vals)) if vals else np.nan

    l24, l48, l72, l168 = lag(24), lag(48), lag(72), lag(168)
    ma24, ma72, ma168 = rolling_mean(24), rolling_mean(72), rolling_mean(168)

    tp = _time_parts(ts)

    return {
        "lag_tmo_24": l24, "lag_tmo_48": l48, "lag_tmo_72": l72, "lag_tmo_168": l168,
        "ma_tmo_24": ma24, "ma_tmo_72": ma72, "ma_tmo_168": ma168,
        "sin_hour": tp["sin_hour"], "cos_hour": tp["cos_hour"],
        "sin_dow": tp["sin_dow"], "cos_dow": tp["cos_dow"],
        "feriados": int(feriado_at_ts or 0),
        "es_dia_de_pago": tp["es_dia_de_pago"],
        "dow": tp["dow"], "month": tp["month"], "hour": tp["hour"],
        "proporcion_comercial": prop_comercial if prop_comercial is not None else np.nan,
        "proporcion_tecnica":  prop_tecnica  if prop_tecnica  is not None else np.nan,
        "recibidos_nacional": float(calls_at_ts) if calls_at_ts is not None and pd.notna(calls_at_ts) else np.nan,
    }

def predict_tmo_autoregresivo(
    idx_horizonte: pd.DatetimeIndex,
    serie_tmo_hist: pd.Series | Mapping[pd.Timestamp, float],
    calls_series: pd.Series | Mapping[pd.Timestamp, float],
    feriados_series: pd.Series | Mapping[pd.Timestamp, int],
    scaler_tmo,                        # StandardScaler
    training_cols_tmo: Sequence[str],  # columnas exactas (json)
    tmo_model,                         # tf.keras.Model
    fill_prop: tuple[float, float] = (0.55, 0.45),
    cap_k: float = 3.5,
    clamp_bounds: tuple[float, float] = (80.0, 900.0),
    strict_columns_check: bool = False,
) -> pd.Series:
    if hasattr(serie_tmo_hist, "to_dict"):
        tmo_hist = serie_tmo_hist.to_dict()  # type: ignore
    else:
        tmo_hist = dict(serie_tmo_hist)

    prop_com, prop_tec = fill_prop
    preds = []

    for ts in idx_horizonte:
        calls_t = _get_from_series(calls_series, ts)
        fer_t   = _get_from_series(feriados_series, ts)

        row = _row_features_tmo(ts, tmo_hist, calls_t, fer_t, prop_com, prop_tec)

        X = pd.DataFrame([row])
        X = pd.get_dummies(X, columns=["dow", "month", "hour"], drop_first=False)
        X = X.reindex(columns=list(training_cols_tmo), fill_value=0)
        if X.isna().any().any():
            X = X.fillna(method="ffill", axis=1).fillna(0)

        if strict_columns_check:
            assert list(X.columns) == list(training_cols_tmo), "Columnas no coinciden con el training."

        Xs = scaler_tmo.transform(X.values)
        y_hat = float(tmo_model.predict(Xs, verbose=0).ravel()[0])

        preds.append((ts, y_hat))
        tmo_hist[ts] = y_hat

    s_pred = pd.Series(dict(preds)).sort_index()
    s_pred = _mad_cap(s_pred, k=cap_k)
    s_pred = _clamp(s_pred, low=float(clamp_bounds[0]), high=float(clamp_bounds[1]))
    return s_pred

def predict_tmo_from_models_dir(
    idx_horizonte: pd.DatetimeIndex,
    serie_tmo_hist: pd.Series,
    calls_series: pd.Series,
    feriados_series: pd.Series,
    models_dir: str | Path = "models",
    fill_prop: tuple[float, float] = (0.55, 0.45),
    cap_k: float = 3.5,
    clamp_bounds: tuple[float, float] = (80.0, 900.0),
    strict_columns_check: bool = False,
) -> pd.Series:
    from joblib import load as joblib_load
    import tensorflow as tf

    models_dir = Path(models_dir)
    scaler_tmo = joblib_load(models_dir / "scaler_tmo.pkl")
    with open(models_dir / "training_columns_tmo.json", "r", encoding="utf-8") as f:
        training_cols_tmo = json.load(f)
    tmo_model = tf.keras.models.load_model(models_dir / "modelo_tmo.keras")

    return predict_tmo_autoregresivo(
        idx_horizonte=idx_horizonte,
        serie_tmo_hist=serie_tmo_hist,
        calls_series=calls_series,
        feriados_series=feriados_series,
        scaler_tmo=scaler_tmo,
        training_cols_tmo=training_cols_tmo,
        tmo_model=tmo_model,
        fill_prop=fill_prop,
        cap_k=cap_k,
        clamp_bounds=clamp_bounds,
        strict_columns_check=strict_columns_check,
    )
