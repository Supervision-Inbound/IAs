# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# -------- Imports resilientes (paquete / módulo plano) --------
# features
try:
    from .features import add_time_parts, ensure_ts
except Exception:
    try:
        from src.inferencia.features import add_time_parts, ensure_ts
    except Exception:
        from inferencia.features import add_time_parts, ensure_ts  # type: ignore

# erlang
try:
    from ..erlang import required_agents
except Exception:
    try:
        from src.erlang import required_agents
    except Exception:
        from erlang import required_agents  # type: ignore

# utils_io
try:
    from ..utils_io import write_json_pretty, write_daily_agg
except Exception:
    try:
        from src.utils_io import write_json_pretty, write_daily_agg
    except Exception:
        from utils_io import write_json_pretty, write_daily_agg  # type: ignore


# =======================
# Constantes / nombres
# =======================
TIMEZONE = "America/Santiago"

# Nombres estándar de columnas
TARGET_CALLS = "recibidos_nacional"
TARGET_TMO = "tmo_general"  # mapeado desde "TMO (segundos)" en main

# Artefactos de modelos (por defecto dentro de ./models)
CALLS_MODEL_NAME = "modelo_planner.keras"
CALLS_SCALER_NAME = "scaler_planner.pkl"
CALLS_COLS_NAME = "training_columns_planner.json"

TMO_MODEL_NAME = "modelo_tmo.keras"
TMO_SCALER_NAME = "scaler_tmo.pkl"
TMO_COLS_NAME = "training_columns_tmo.json"
TMO_BASELINE_NAME = "tmo_baseline_dow_hour.csv"       # baseline por dow/hour (opcional)
TMO_META_NAME = "tmo_residual_meta.json"              # metadatos (opcional)


# =======================
# Utilidades de TS
# =======================
def _as_ts_index(df: Optional[pd.DataFrame], tz: str = TIMEZONE, sort: bool = True) -> Optional[pd.DataFrame]:
    """
    Garantiza que 'ts' sea SOLO índice y DatetimeIndex en tz dada.
    Evita "'ts' is both an index level and a column label".
    """
    if df is None:
        return None
    d = df.copy()

    # Si 'ts' existe como columna y además el índice se llama 'ts', la quitamos
    if "ts" in d.columns and (d.index.name == "ts" or "ts" in (d.index.names or [])):
        d = d.drop(columns=["ts"])

    # Si aún no es índice 'ts' pero existe la columna, la usamos
    if d.index.name != "ts" and "ts" in d.columns:
        d = d.set_index("ts")

    # Asegurar dtype datetime
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, errors="coerce")

    # Localizar / convertir zona horaria, manejando DST
    try:
        if d.index.tz is None:
            d.index = d.index.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
        else:
            d.index = d.index.tz_convert(tz)
    except TypeError:
        # Compat para pandas antiguos
        try:
            if d.index.tz is None:
                d.index = d.index.tz_localize(tz)
            else:
                d.index = d.index.tz_convert(tz)
        except Exception:
            pass

    # Eliminar duplicados de índice (nos quedamos con el último)
    if d.index.has_duplicates:
        d = d[~d.index.duplicated(keep="last")]

    if sort:
        d = d.sort_index()

    return d


def _future_index(start: pd.Timestamp, hours: int, tz: str = TIMEZONE) -> pd.DatetimeIndex:
    start = pd.Timestamp(start).tz_convert(tz)
    start = (start + pd.Timedelta(hours=1)).floor("h")
    return pd.date_range(start=start, periods=hours, freq="h", tz=tz)


# =======================
# Carga de artefactos
# =======================
def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_scaler(path: Path) -> StandardScaler:
    import joblib
    return joblib.load(path)

def _load_calls(models_dir: Path):
    model = tf.keras.models.load_model(models_dir / CALLS_MODEL_NAME)
    scaler = _load_scaler(models_dir / CALLS_SCALER_NAME)
    cols = _load_json(models_dir / CALLS_COLS_NAME)
    return model, scaler, cols

def _load_tmo(models_dir: Path):
    model = tf.keras.models.load_model(models_dir / TMO_MODEL_NAME)
    scaler = _load_scaler(models_dir / TMO_SCALER_NAME)
    cols = _load_json(models_dir / TMO_COLS_NAME)
    # baseline residual, si existe
    baseline = None
    meta = {}
    if (models_dir / TMO_BASELINE_NAME).exists():
        baseline = pd.read_csv(models_dir / TMO_BASELINE_NAME)
        baseline.columns = [c.strip().lower() for c in baseline.columns]
        if "dow" not in baseline.columns or "hour" not in baseline.columns:
            if "day_of_week" in baseline.columns:
                baseline.rename(columns={"day_of_week": "dow"}, inplace=True)
        if "tmo_baseline" not in baseline.columns:
            for c in baseline.columns:
                if "baseline" in c:
                    baseline.rename(columns={c: "tmo_baseline"}, inplace=True)
                    break
        baseline = baseline[["dow", "hour", "tmo_baseline"]].copy()

    if (models_dir / TMO_META_NAME).exists():
        meta = _load_json(models_dir / TMO_META_NAME)

    return model, scaler, cols, baseline, meta


# =======================
# Ingeniería de features
# =======================
def _mark_paydays(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["day"] = d.index.day
    d["es_dia_de_pago"] = d["day"].isin([1, 2, 15, 16, 29, 30, 31]).astype(int)
    return d

def _one_hot_time(d: pd.DataFrame, keep_numeric: Iterable[str]) -> pd.DataFrame:
    d = pd.get_dummies(d, columns=["dow", "month", "hour"], drop_first=False)
    for c in keep_numeric:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

def _align_columns(X: pd.DataFrame, training_cols: Iterable[str]) -> pd.DataFrame:
    X = X.reindex(columns=list(training_cols), fill_value=0)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.ffill().bfill().fillna(0)
    return X

def _build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d = add_time_parts(d)
    d = _mark_paydays(d)
    d["sin_hour"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["cos_hour"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["sin_dow"] = np.sin(2 * np.pi * d["dow"] / 7)
    d["cos_dow"] = np.cos(2 * np.pi * d["dow"] / 7)
    return d


# =======================
# Predicción de llamadas (usa tu modelo existente)
# =======================
def _predict_calls_hourly(
    df_hist: pd.DataFrame,
    models_dir: Path,
    horizon_hours: int,
    holidays_set: Optional[set]
) -> pd.DataFrame:
    m, sc, cols = _load_calls(models_dir)

    d_hist = df_hist[[TARGET_CALLS, "feriados"]].copy()
    d_hist["feriados"] = d_hist["feriados"].fillna(0).astype(int)

    feats_hist = _build_time_features(d_hist)

    idx_fut = _future_index(d_hist.index[-1], horizon_hours)
    d_fut = pd.DataFrame(index=idx_fut, data={"feriados": 0})
    if holidays_set:
        d_fut["fecha"] = d_fut.index.date
        d_fut["feriados"] = d_fut["fecha"].isin(holidays_set).astype(int)
        d_fut.drop(columns=["fecha"], inplace=True)

    d_all = pd.concat([d_hist, d_fut], axis=0)
    d_all = _build_time_features(d_all)

    def _compute_lags_and_ma(series: pd.Series) -> pd.DataFrame:
        out = pd.DataFrame(index=series.index)
        for lag in [24, 48, 72, 168]:
            out[f"lag_{lag}"] = series.shift(lag)
        for win in [24, 72, 168]:
            out[f"ma_{win}"] = series.rolling(win, min_periods=1).mean()
        return out

    lags_ma_hist = _compute_lags_and_ma(d_all[TARGET_CALLS])

    base = pd.concat(
        [
            lags_ma_hist,
            d_all[["sin_hour", "cos_hour", "sin_dow", "cos_dow", "feriados", "es_dia_de_pago", "dow", "month", "hour"]],
        ],
        axis=1,
    )
    X_all = _one_hot_time(base, keep_numeric=["sin_hour", "cos_hour", "sin_dow", "cos_dow", "feriados", "es_dia_de_pago"])
    X_all = _align_columns(X_all, cols)
    X_s = sc.transform(X_all.fillna(0))

    y_all = d_all[TARGET_CALLS].copy()
    for ts in idx_fut:
        window_idx = y_all.index.union(pd.DatetimeIndex([ts])).sort_values()
        y_tmp = y_all.reindex(window_idx)
        ltmp = _compute_lags_and_ma(y_tmp).loc[[ts]]
        for c in ltmp.columns:
            X_all.loc[ts, c] = ltmp.loc[ts, c]
        x_row = X_all.loc[[ts]].fillna(0)
        x_row = _align_columns(x_row, cols)
        x_row_s = sc.transform(x_row)
        y_pred = float(m.predict(x_row_s, verbose=0).ravel()[0])
        y_all.loc[ts] = max(0.0, y_pred)

    out = pd.DataFrame(index=idx_fut, data={TARGET_CALLS: y_all.loc[idx_fut].values})
    return out


# =======================
# Predicción de TMO (nuevo)
# =======================
def _tmo_baseline_lookup(baseline_df: Optional[pd.DataFrame], idx: pd.DatetimeIndex) -> pd.Series:
    if baseline_df is None or baseline_df.empty:
        return pd.Series(index=idx, dtype=float)

    tmp = pd.DataFrame({"dow": idx.dayofweek, "hour": idx.hour}, index=idx)
    base = baseline_df.set_index(["dow", "hour"])
    vals = []
    for i, r in tmp.iterrows():
        key = (int(r["dow"]), int(r["hour"]))
        vals.append(float(base.loc[key, "tmo_baseline"]) if key in base.index else np.nan)
    return pd.Series(vals, index=idx, dtype=float)

def _predict_tmo_hourly(
    df_hist_joined: pd.DataFrame,
    models_dir: Path,
    horizon_hours: int,
    holidays_set: Optional[set]
) -> pd.DataFrame:
    m, sc, cols, baseline, meta = _load_tmo(models_dir)

    d_hist = df_hist_joined[[TARGET_CALLS, "feriados"]].copy()
    d_hist["feriados"] = d_hist["feriados"].fillna(0).astype(int)

    idx_fut = _future_index(d_hist.index[-1], horizon_hours)
    d_fut = pd.DataFrame(index=idx_fut, data={"feriados": 0})
    if holidays_set:
        d_fut["fecha"] = d_fut.index.date
        d_fut["feriados"] = d_fut["fecha"].isin(holidays_set).astype(int)
        d_fut.drop(columns=["fecha"], inplace=True)

    d_all = pd.concat([d_hist, d_fut], axis=0)
    d_all = _build_time_features(d_all)

    base_series = _tmo_baseline_lookup(baseline, d_all.index)
    d_all["tmo_baseline"] = base_series.values

    base_feats = ["sin_hour", "cos_hour", "sin_dow", "cos_dow", "feriados", "es_dia_de_pago", "dow", "month", "hour"]
    if "tmo_baseline" in d_all.columns:
        base_feats.append("tmo_baseline")
    X = d_all[base_feats].copy()
    X = _one_hot_time(X, keep_numeric=["sin_hour", "cos_hour", "sin_dow", "cos_dow", "feriados", "es_dia_de_pago", "tmo_baseline"])
    X = _align_columns(X, cols)
    X_s = sc.transform(X)

    y_hat = m.predict(X_s, verbose=0).ravel().astype(float)
    if "residual" in meta.get("type", "").lower() and "tmo_baseline" in d_all.columns:
        y_hat = y_hat + d_all["tmo_baseline"].fillna(method="ffill").fillna(method="bfill").fillna(0).values

    y_hat = np.clip(y_hat, 60.0, 1200.0)  # 1 a 20 minutos

    return pd.DataFrame(index=idx_fut, data={"tmo_s": y_hat[-len(idx_fut):]})


# =======================
# Consolidación / salida
# =======================
def _daily_weighted_aht(df_hourly: pd.DataFrame) -> pd.DataFrame:
    dd = df_hourly.copy()
    dd["calls_x_tmo"] = dd[TARGET_CALLS] * dd["tmo_s"]
    g = dd.groupby(dd.index.date)
    out = pd.DataFrame({
        "calls": g[TARGET_CALLS].sum(),
        "aht_s": (g["calls_x_tmo"].sum() / g[TARGET_CALLS].sum()).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    })
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    return out


# =======================
# API principal
# =======================
def forecast_120d(
    df_hist_joined: pd.DataFrame,
    df_hist_tmo_only: Optional[pd.DataFrame] = None,  # compat
    horizon_days: int = 120,
    holidays_set: Optional[set] = None,
    models_dir: Optional[Path] = None,
) -> pd.DataFrame:
    if models_dir is None:
        models_dir = Path("models")
    models_dir = Path(models_dir)

    df_hist_joined = _as_ts_index(df_hist_joined)
    if df_hist_joined is None or df_hist_joined.empty:
        raise ValueError("df_hist_joined vacío o None.")

    for col in [TARGET_CALLS, "feriados"]:
        if col not in df_hist_joined.columns:
            raise ValueError(f"Columna requerida no encontrada: '{col}'")

    horizon_hours = int(horizon_days) * 24

    calls_fut = _predict_calls_hourly(
        df_hist=df_hist_joined,
        models_dir=models_dir,
        horizon_hours=horizon_hours,
        holidays_set=holidays_set,
    )

    tmo_fut = _predict_tmo_hourly(
        df_hist_joined=df_hist_joined,
        models_dir=models_dir,
        horizon_hours=horizon_hours,
        holidays_set=holidays_set,
    )

    df_hourly = calls_fut.join(tmo_fut, how="left")

    agents = []
    for _, row in df_hourly.iterrows():
        a = required_agents(float(row[TARGET_CALLS]), float(row["tmo_s"]))
        agents.append(int(np.ceil(a)))
    df_hourly["agents"] = agents

    out_hourly = df_hourly.copy()
    out_hourly.index.name = "ts"
    out_hourly_reset = out_hourly.reset_index()
    out_hourly_reset["ts"] = out_hourly_reset["ts"].dt.tz_convert(TIMEZONE).dt.strftime("%Y-%m-%d %H:%M:%S%z")
    write_json_pretty("public/prediccion_horaria.json", out_hourly_reset)

    df_daily = _daily_weighted_aht(df_hourly)
    write_daily_agg("public/prediccion_diaria.json", df_daily)

    return df_hourly
