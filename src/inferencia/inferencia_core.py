# src/inferencia/inferencia_core.py
import os, json, glob, pathlib, re
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from src.inferencia.features import ensure_ts, add_time_parts, dummies_and_reindex
from src.erlang import required_agents, schedule_agents
from src.utils_io import write_daily_json, write_hourly_json

TIMEZONE = "America/Santiago"
PUBLIC_DIR = "public"

# === Artefactos planner (sin cambios)
PLANNER_MODEL = "models/modelo_planner.keras"
PLANNER_SCALER = "models/scaler_planner.pkl"
PLANNER_COLS   = "models/training_columns_planner.json"

# === Artefactos TMO (sin cambios)
TMO_MODEL    = "models/modelo_tmo.keras"
TMO_SCALER   = "models/scaler_tmo.pkl"
TMO_COLS     = "models/training_columns_tmo.json"
TMO_BASELINE = "models/tmo_baseline_dow_hour.csv"
TMO_META     = "models/tmo_residual_meta.json"

# === >>> ÚNICO CAMBIO SOLICITADO: usar RECIBIDOS como base del planner <<<
FEATURE_BASE_CALLS = "recibidos"  # (antes: contestados)
TARGET_TMO         = "tmo_general"

HIST_WINDOW_DAYS = 90

def _load_cols(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_tmo_residual_artifacts_or_fallback(df_hist: pd.DataFrame):
    has_meta = os.path.exists(TMO_META)
    has_base = os.path.exists(TMO_BASELINE)
    if has_meta and has_base:
        with open(TMO_META, "r", encoding="utf-8") as f:
            meta = json.load(f)
        base = pd.read_csv(TMO_BASELINE)[["dow","hour","tmo_baseline"]].copy()
        base["dow"]  = base["dow"].astype(int)
        base["hour"] = base["hour"].astype(int)
        resid_mean = float(meta.get("resid_mean", 0.0))
        resid_std  = float(meta.get("resid_std",  1.0)) or 1.0
        return base, resid_mean, resid_std
    # Fallback simple igual al que tenías
    d = add_time_parts(df_hist.copy())
    if TARGET_TMO in d.columns:
        tmo = pd.to_numeric(d[TARGET_TMO], errors="coerce")
        base = (d.assign(tmo_s=tmo).groupby(["dow","hour"])["tmo_s"]
                  .median().rename("tmo_baseline").reset_index())
        merged = d.merge(base, on=["dow","hour"], how="left")
        resid = pd.to_numeric(merged[TARGET_TMO], errors="coerce") - merged["tmo_baseline"]
        resid_mean = float(np.nanmean(resid))
        resid_std  = float(np.nanstd(resid)) or 1.0
    else:
        base = pd.DataFrame({"dow": np.repeat(np.arange(7), 24),
                             "hour": list(np.tile(np.arange(24), 7)),
                             "tmo_baseline": 180.0})
        resid_mean = 0.0
        resid_std  = 1.0
    return base, resid_mean, resid_std

# --- Helper para features del planner (igual que te funcionó)
_LAG_RE   = re.compile(r"^lag_([A-Za-z0-9_]+)_(\d+)$")
_MA_NAME1 = re.compile(r"^ma_(\d+)$")
_MA_NAME2 = re.compile(r"^ma_([A-Za-z0-9_]+)_(\d+)$")

def _ensure_base_series(dfp: pd.DataFrame, base_name: str, fallback_col: str = FEATURE_BASE_CALLS):
    if base_name not in dfp.columns:
        dfp[base_name] = pd.to_numeric(dfp[fallback_col], errors="coerce").copy()
    return dfp

def _build_planner_features(dfp: pd.DataFrame, cols_pl: list, tz=TIMEZONE) -> pd.DataFrame:
    d = dfp.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, errors="coerce", utc=True).tz_convert(tz)
    elif d.index.tz is None:
        d.index = d.index.tz_localize(tz)
    else:
        d.index = d.index.tz_convert(tz)

    if "feriados" not in d.columns: d["feriados"] = 0
    if "es_dia_de_pago" not in d.columns: d["es_dia_de_pago"] = 0
    if FEATURE_BASE_CALLS not in d.columns: d[FEATURE_BASE_CALLS] = 0.0
    d[FEATURE_BASE_CALLS] = pd.to_numeric(d[FEATURE_BASE_CALLS], errors="coerce")

    temp = add_time_parts(d[[c for c in d.columns if c in ("feriados","es_dia_de_pago", FEATURE_BASE_CALLS)]].copy())
    feat = pd.DataFrame(index=d.tail(1).index)

    for col in cols_pl:
        m = _LAG_RE.match(col)
        if m:
            base, k = m.group(1), int(m.group(2))
            _ensure_base_series(d, base, fallback_col=FEATURE_BASE_CALLS)
            d[base] = pd.to_numeric(d[base], errors="coerce")
            lag_col = f"__lag__{base}__{k}"
            if lag_col not in d.columns:
                d[lag_col] = d[base].shift(k)
            feat[col] = d[lag_col].tail(1).values
            continue

        m1 = _MA_NAME1.match(col)
        m2 = _MA_NAME2.match(col)
        if m1:
            w = int(m1.group(1))
            ma_col = f"__ma__{FEATURE_BASE_CALLS}__{w}"
            if ma_col not in d.columns:
                d[ma_col] = d[FEATURE_BASE_CALLS].rolling(w, min_periods=1).mean()
            feat[col] = d[ma_col].tail(1).values
            continue
        if m2:
            base, w = m2.group(1), int(m2.group(2))
            _ensure_base_series(d, base, fallback_col=FEATURE_BASE_CALLS)
            d[base] = pd.to_numeric(d[base], errors="coerce")
            ma_col = f"__ma__{base}__{w}"
            if ma_col not in d.columns:
                d[ma_col] = d[base].rolling(w, min_periods=1).mean()
            feat[col] = d[ma_col].tail(1).values
            continue

        if col.startswith("dow_") or col.startswith("month_") or col.startswith("hour_"):
            temp_dum = pd.get_dummies(temp[["dow","month","hour"]], columns=["dow","month","hour"], drop_first=False)
            feat[col] = temp_dum[col].tail(1).values if col in temp_dum.columns else 0
            continue

        if col in d.columns:
            feat[col] = d[col].tail(1).values
        elif col in temp.columns:
            feat[col] = temp[col].tail(1).values
        else:
            feat[col] = 0

    feat = feat.reindex(columns=cols_pl, fill_value=0)
    for c in feat.columns:
        feat[c] = pd.to_numeric(feat[c], errors="coerce")
    feat = feat.ffill().fillna(0)
    return feat

def forecast_120d(df_hist_joined: pd.DataFrame, horizon_days: int = 120, holidays_set: set | None = None):
    # Artefactos
    m_pl = tf.keras.models.load_model(PLANNER_MODEL, compile=False)
    sc_pl = joblib.load(PLANNER_SCALER)
    cols_pl = _load_cols(PLANNER_COLS)

    m_tmo = tf.keras.models.load_model(TMO_MODEL, compile=False)
    sc_tmo = joblib.load(TMO_SCALER)
    cols_tmo = _load_cols(TMO_COLS)

    # Histórico
    df = ensure_ts(df_hist_joined)
    if "feriados" not in df.columns: df["feriados"] = 0
    if "es_dia_de_pago" not in df.columns: df["es_dia_de_pago"] = 0
    if FEATURE_BASE_CALLS not in df.columns:
        raise ValueError(f"Falta columna {FEATURE_BASE_CALLS} en histórico.")

    # Artefactos TMO (original)
    tmo_base_table, resid_mean, resid_std = _load_tmo_residual_artifacts_or_fallback(df)

    # Prep residuo TMO (original)
    keep_cols = [FEATURE_BASE_CALLS, "feriados", "es_dia_de_pago"]
    if TARGET_TMO in df.columns: keep_cols.append(TARGET_TMO)
    df_tmp = add_time_parts(df[keep_cols].copy())
    df_tmp["ts"] = df.index
    df_tmp = df_tmp.merge(tmo_base_table, on=["dow","hour"], how="left").sort_values("ts").set_index("ts")

    if TARGET_TMO in df_tmp.columns:
        df_tmp["tmo_resid"] = pd.to_numeric(df_tmp[TARGET_TMO], errors="coerce") - df_tmp["tmo_baseline"]
    else:
        df_tmp["tmo_resid"] = 0.0

    last_ts = pd.to_datetime(df_tmp.index.max())
    recent_mask = df_tmp.index >= (last_ts - pd.Timedelta(days=HIST_WINDOW_DAYS))
    dfp = df_tmp.loc[recent_mask].copy() if recent_mask.any() else df_tmp.copy()

    dfp[FEATURE_BASE_CALLS] = pd.to_numeric(dfp[FEATURE_BASE_CALLS], errors="coerce").ffill().fillna(0.0)
    dfp["tmo_resid"] = pd.to_numeric(dfp["tmo_resid"], errors="coerce").fillna(0.0)

    # Cuadrícula futura
    future_ts = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=horizon_days*24, freq="h", tz=TIMEZONE)

    # Bucle iterativo (original con base recibidos)
    for ts in future_ts:
        # Planner (recibidos)
        X_pl = _build_planner_features(dfp, cols_pl, tz=TIMEZONE)
        yhat_rec = float(m_pl.predict(sc_pl.transform(X_pl), verbose=0).flatten()[0])
        yhat_rec = max(0.0, yhat_rec if np.isfinite(yhat_rec) else 0.0)

        # TMO: baseline + residuo
        dow = int(ts.weekday()); hour = int(ts.hour)
        base_row = tmo_base_table[(tmo_base_table["dow"]==dow)&(tmo_base_table["hour"]==hour)]
        tmo_base = float(base_row["tmo_baseline"].iloc[0]) if not base_row.empty else float(np.nanmedian(tmo_base_table["tmo_baseline"]))
        tmp_tmo = dfp[["tmo_resid","feriados","es_dia_de_pago"]].copy()
        tmp_tmo.loc[ts, ["tmo_resid","feriados","es_dia_de_pago"]] = [tmp_tmo["tmo_resid"].iloc[-1], 0, 0]
        # features TMO
        # (igual a tu original)
        dft = tmp_tmo.copy()
        for lag in [1,2,3,6,12,24,48,72,168]:
            dft[f"lag_resid_{lag}"] = dft["tmo_resid"].shift(lag)
        r1 = dft["tmo_resid"].shift(1)
        for w in [6,12,24,72,168]:
            dft[f"ma_resid_{w}"] = r1.rolling(w, min_periods=1).mean()
        for span in [6,12,24]:
            dft[f"ema_resid_{span}"] = r1.ewm(span=span, adjust=False, min_periods=1).mean()
        for w in [24,72]:
            dft[f"std_resid_{w}"] = r1.rolling(w, min_periods=2).std()
            dft[f"max_resid_{w}"] = r1.rolling(w, min_periods=1).max()
        dft = add_time_parts(dft)
        X_tmo = dummies_and_reindex(dft.tail(1), cols_tmo)
        yhat_z = float(m_tmo.predict(joblib.load(TMO_SCALER).transform(X_tmo), verbose=0).flatten()[0])  # scaler ya cargado arriba; mantener compat
        yhat_resid = yhat_z * resid_std + resid_mean
        yhat_tmo = max(0.0, tmo_base + (yhat_resid if np.isfinite(yhat_resid) else 0.0))

        # Actualizo iterativamente
        dfp.loc[ts, FEATURE_BASE_CALLS] = yhat_rec
        dfp.loc[ts, "tmo_baseline"] = tmo_base
        dfp.loc[ts, "tmo_resid"] = yhat_tmo - tmo_base
        dfp.loc[ts, "feriados"] = 0
        dfp.loc[ts, "es_dia_de_pago"] = 0

    # Salida (igual a original, pero calls = recibidos)
    df_hourly = pd.DataFrame(index=future_ts)
    df_hourly["calls"] = np.round(dfp.loc[future_ts, FEATURE_BASE_CALLS]).astype(int)
    df_hourly["tmo_s"] = np.round(dfp.loc[future_ts, "tmo_baseline"] + dfp.loc[future_ts, "tmo_resid"]).astype(int)

    # Erlang
    df_hourly["agents_prod"] = 0
    for ts in df_hourly.index:
        a, _ = required_agents(float(df_hourly.at[ts, "calls"]), float(df_hourly.at[ts, "tmo_s"]))
        df_hourly.at[ts, "agents_prod"] = int(a)
    df_hourly["agents_sched"] = df_hourly["agents_prod"].apply(schedule_agents)

    write_hourly_json(f"{PUBLIC_DIR}/prediccion_horaria.json", df_hourly, "calls", "tmo_s", "agents_sched")
    write_daily_json(f"{PUBLIC_DIR}/prediccion_diaria.json", df_hourly, "calls", "tmo_s", weights_col="calls")
    return df_hourly

