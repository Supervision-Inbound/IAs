# src/inferencia/inferencia_core.py
import os
import glob
import pathlib
import json
import re
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from .features import ensure_ts, add_time_parts, dummies_and_reindex
from .erlang import required_agents, schedule_agents
from .utils_io import write_daily_json, write_hourly_json

TIMEZONE = "America/Santiago"
PUBLIC_DIR = "public"

# ---------- Localización flexible de artefactos TMO ----------
def _candidate_dirs():
    here = pathlib.Path(".").resolve()
    bases = [
        here, here / "models", here / "modelos", here / "release", here / "releases",
        here / "artifacts", here / "outputs", here / "output", here / "dist", here / "build",
        pathlib.Path(os.environ.get("GITHUB_WORKSPACE", ".")),
        pathlib.Path("/kaggle/working/models"), pathlib.Path("/kaggle/working"),
    ]
    uniq = []
    for p in bases:
        try:
            p = p.resolve()
            if p.exists() and p not in uniq:
                uniq.append(p)
        except Exception:
            pass
    return uniq

def _find_one(patterns, search_dirs=None):
    if isinstance(patterns, str):
        patterns = [patterns]
    search_dirs = search_dirs or _candidate_dirs()
    for d in search_dirs:
        for pat in patterns:
            for match in glob.glob(str(d / "**" / pat), recursive=True):
                p = pathlib.Path(match)
                if p.is_file():
                    return str(p)
    return None

def _resolve_tmo_artifacts():
    return {
        "keras": _find_one(["modelo_tmo.keras", "tmo*.keras", "*_tmo*.keras"]),
        "scaler": _find_one(["scaler_tmo.pkl", "*tmo*scaler*.pkl", "*scaler*_tmo*.pkl"]),
        "cols": _find_one(["training_columns_tmo.json", "*tmo*columns*.json", "*training*columns*to*.json"]),
        "baseline": _find_one(["tmo_baseline_dow_hour.csv", "*tmo*baseline*.csv"]),
        "meta": _find_one(["tmo_residual_meta.json", "*tmo*residual*meta*.json"]),
    }

_paths = _resolve_tmo_artifacts()

# ---------- Modelos / columnas ----------
PLANNER_MODEL = "models/modelo_planner.keras"
PLANNER_SCALER = "models/scaler_planner.pkl"
PLANNER_COLS   = "models/training_columns_planner.json"

TMO_MODEL    = _paths.get("keras")    or "models/modelo_tmo.keras"
TMO_SCALER   = _paths.get("scaler")   or "models/scaler_tmo.pkl"
TMO_COLS     = _paths.get("cols")     or "models/training_columns_tmo.json"
TMO_BASELINE = _paths.get("baseline") or "models/tmo_baseline_dow_hour.csv"
TMO_META     = _paths.get("meta")     or "models/tmo_residual_meta.json"

# ---------- Negocio ----------
# TODO pedido: usar SOLO 'recibidos' (planner y TMO); NO usar 'contestados'
FEATURE_BASE_CALLS = "recibidos"
TARGET_TMO         = "tmo_general"  # nombre estándar TMO en segundos (ajusta si tu histórico usa otro)

HIST_WINDOW_DAYS = 90
ENABLE_OUTLIER_CAP = True
K_WEEKDAY = 6.0
K_WEEKEND = 7.0

# ---------- Utils ----------
def _load_cols(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _is_holiday(ts, holidays_set: set) -> int:
    if not holidays_set:
        return 0
    try:
        d = ts.tz_convert(TIMEZONE).date()
    except Exception:
        d = ts.date()
    return 1 if d in holidays_set else 0

# ---------- TMO residual: artefactos + fallback (no depende de contestados) ----------
def _try_build_tmo_artifacts_from_history(df_hist: pd.DataFrame):
    print("WARN: Artefactos TMO no encontrados. Construyendo baseline y meta desde histórico...")
    d = ensure_ts(df_hist.copy())
    d = add_time_parts(d)
    if TARGET_TMO in d.columns:
        tmo = pd.to_numeric(d[TARGET_TMO], errors="coerce")
        base = (d.assign(tmo_s=tmo).groupby(["dow","hour"])["tmo_s"].median().rename("tmo_baseline").reset_index())
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
    base["dow"]  = base["dow"].astype(int)
    base["hour"] = base["hour"].astype(int)
    print(f"INFO: baseline TMO generado (rows={len(base)}), resid_mean={resid_mean:.3f}, resid_std={resid_std:.3f}")
    return base, resid_mean, resid_std

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
    return _try_build_tmo_artifacts_from_history(df_hist)

def _add_tmo_resid_features(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in.copy()
    # lags/MA del residuo (independiente de recibidos/contestados)
    for lag in [1,2,3,6,12,24,48,72,168]:
        d[f"lag_resid_{lag}"] = d["tmo_resid"].shift(lag)
    r1 = d["tmo_resid"].shift(1)
    for w in [6,12,24,72,168]:
        d[f"ma_resid_{w}"] = r1.rolling(w, min_periods=1).mean()
    for span in [6,12,24]:
        d[f"ema_resid_{span}"] = r1.ewm(span=span, adjust=False, min_periods=1).mean()
    for w in [24,72]:
        d[f"std_resid_{w}"] = r1.rolling(w, min_periods=2).std()
        d[f"max_resid_{w}"] = r1.rolling(w, min_periods=1).max()
    d = add_time_parts(d)
    return d

# ---------- Parser de columnas del planner (lags/MA exactos) ----------
_LAG_RE   = re.compile(r"^lag_([A-Za-z0-9_]+)_(\d+)$")
_MA_NAME1 = re.compile(r"^ma_(\d+)$")                 # ma_24
_MA_NAME2 = re.compile(r"^ma_([A-Za-z0-9_]+)_(\d+)$") # ma_recibidos_72

def _extract_bases(cols_pl: list) -> set:
    bases = set([FEATURE_BASE_CALLS])
    for c in cols_pl:
        m = _LAG_RE.match(c)
        if m:
            bases.add(m.group(1))
        else:
            m2 = _MA_NAME2.match(c)
            if m2:
                bases.add(m2.group(1))
    return bases

def _ensure_base_series(dfp: pd.DataFrame, base_name: str, fallback_col: str = FEATURE_BASE_CALLS):
    if base_name not in dfp.columns:
        dfp[base_name] = pd.to_numeric(dfp[fallback_col], errors="coerce").copy()
    return dfp

def _build_planner_features(dfp: pd.DataFrame, cols_pl: list, tz=TIMEZONE) -> pd.DataFrame:
    """
    Construye EXACTAMENTE las columnas que pide training_columns_planner.json (una fila),
    usando como base 'recibidos' y cualquier otra base requerida (espejadas si no existen).
    """
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
            aux = f"__lag__{base}__{k}"
            if aux not in d.columns:
                d[aux] = d[base].shift(k)
            feat[col] = d[aux].tail(1).values
            continue

        m1 = _MA_NAME1.match(col)
        m2 = _MA_NAME2.match(col)
        if m1:
            w = int(m1.group(1))
            aux = f"__ma__{FEATURE_BASE_CALLS}__{w}"
            if aux not in d.columns:
                d[aux] = d[FEATURE_BASE_CALLS].rolling(w, min_periods=1).mean()
            feat[col] = d[aux].tail(1).values
            continue
        if m2:
            base, w = m2.group(1), int(m2.group(2))
            _ensure_base_series(d, base, fallback_col=FEATURE_BASE_CALLS)
            d[base] = pd.to_numeric(d[base], errors="coerce")
            aux = f"__ma__{base}__{w}"
            if aux not in d.columns:
                d[aux] = d[base].rolling(w, min_periods=1).mean()
            feat[col] = d[aux].tail(1).values
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

    # Saneo final
    feat = feat.reindex(columns=cols_pl, fill_value=0)
    for c in feat.columns:
        feat[c] = pd.to_numeric(feat[c], errors="coerce")
    feat = feat.ffill().fillna(0)
    return feat

# ---------- Núcleo ----------
def forecast_120d(df_hist_joined: pd.DataFrame, horizon_days: int = 120, holidays_set: set | None = None):
    # Artefactos
    m_pl = tf.keras.models.load_model(PLANNER_MODEL, compile=False)
    sc_pl = joblib.load(PLANNER_SCALER)
    cols_pl = _load_cols(PLANNER_COLS)

    m_tmo = tf.keras.models.load_model(TMO_MODEL, compile=False)
    sc_tmo = joblib.load(TMO_SCALER)
    cols_tmo = _load_cols(TMO_COLS)

    # Base histórica
    df = ensure_ts(df_hist_joined)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce", utc=True).tz_convert(TIMEZONE)
    elif df.index.tz is None:
        df.index = df.index.tz_localize(TIMEZONE)
    else:
        df.index = df.index.tz_convert(TIMEZONE)

    # Validaciones de columnas clave (solo recibidos + TMO + flags)
    if FEATURE_BASE_CALLS not in df.columns:
        raise ValueError(f"Falta columna {FEATURE_BASE_CALLS} en historical_data.csv")
    if "feriados" not in df.columns: df["feriados"] = 0
    if "es_dia_de_pago" not in df.columns: df["es_dia_de_pago"] = 0

    # --- Artefactos TMO ---
    tmo_base_table, resid_mean, resid_std = _load_tmo_residual_artifacts_or_fallback(df)

    # --- Preparación histórica para residuo TMO ---
    keep_cols = [FEATURE_BASE_CALLS, "feriados", "es_dia_de_pago"]
    if TARGET_TMO in df.columns: keep_cols.append(TARGET_TMO)

    df_tmp = add_time_parts(df[keep_cols].copy())
    df_tmp["ts"] = df.index
    df_tmp = df_tmp.merge(tmo_base_table, on=["dow","hour"], how="left").sort_values("ts").set_index("ts")

    if TARGET_TMO in df_tmp.columns:
        df_tmp["tmo_resid"] = pd.to_numeric(df_tmp[TARGET_TMO], errors="coerce") - df_tmp["tmo_baseline"]
    else:
        df_tmp["tmo_resid"] = np.nan

    idx = df_tmp.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, errors="coerce", utc=True).tz_convert(TIMEZONE)
        df_tmp = df_tmp.set_index(idx)

    last_ts = pd.to_datetime(df_tmp.index.max())
    mask_recent = df_tmp.index >= (last_ts - pd.Timedelta(days=HIST_WINDOW_DAYS))
    dfp = df_tmp.loc[mask_recent].copy() if mask_recent.any() else df_tmp.copy()

    # saneo
    dfp[FEATURE_BASE_CALLS] = pd.to_numeric(dfp[FEATURE_BASE_CALLS], errors="coerce").ffill().fillna(0.0)
    dfp["tmo_resid"] = pd.to_numeric(dfp["tmo_resid"], errors="coerce")
    if dfp["tmo_resid"].isna().all(): dfp["tmo_resid"] = 0.0

    # Determinar bases a espejar (si el planner entrenó con otras, las espejamos a recibidos)
    bases_required = set([FEATURE_BASE_CALLS])
    for c in cols_pl:
        m = _LAG_RE.match(c)
        if m:
            bases_required.add(m.group(1))
        else:
            m2 = _MA_NAME2.match(c)
            if m2:
                bases_required.add(m2.group(1))
    for b in bases_required:
        _ensure_base_series(dfp, b, fallback_col=FEATURE_BASE_CALLS)

    # Horizonte futuro
    future_ts = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=horizon_days * 24, freq="h", tz=TIMEZONE)

    # ===== Bucle iterativo =====
    print("Iniciando predicción iterativa (RECIBIDOS + TMO residual, sin contestados)...")
    for ts in future_ts:
        # ----- Planner sobre recibidos -----
        X_pl = _build_planner_features(dfp, cols_pl, tz=TIMEZONE)
        yhat_rec = float(m_pl.predict(sc_pl.transform(X_pl), verbose=0).flatten()[0])
        yhat_rec = 0.0 if not np.isfinite(yhat_rec) else max(0.0, yhat_rec)

        # ----- TMO: baseline + residuo (independiente de recibidos) -----
        try:
            dow = int(ts.tz_convert(TIMEZONE).weekday()); hour = int(ts.tz_convert(TIMEZONE).hour)
        except Exception:
            dow = int(ts.weekday()); hour = int(ts.hour)
        base_row = tmo_base_table[(tmo_base_table["dow"]==dow)&(tmo_base_table["hour"]==hour)]
        tmo_base = float(base_row["tmo_baseline"].iloc[0]) if not base_row.empty else \
                   float(np.nanmedian(tmo_base_table["tmo_baseline"])) if "tmo_baseline" in tmo_base_table.columns else 180.0

        tmp_tmo = dfp[["tmo_resid","feriados","es_dia_de_pago"]].copy()
        tmp_tmo.loc[ts, ["tmo_resid","feriados","es_dia_de_pago"]] = [tmp_tmo["tmo_resid"].iloc[-1], 0, 0]
        # features TMO
        dft = _add_tmo_resid_features(tmp_tmo)
        X_tmo = dummies_and_reindex(dft.tail(1), cols_tmo)
        yhat_z = float(m_tmo.predict(sc_tmo.transform(X_tmo), verbose=0).flatten()[0])
        yhat_resid = yhat_z * resid_std + resid_mean
        yhat_tmo = 0.0 if not np.isfinite(yhat_resid) else max(0.0, tmo_base + yhat_resid)

        # ----- Actualización iterativa -----
        # 1) Actualizar recibidos en ts y espejar otras bases del planner
        dfp.loc[ts, FEATURE_BASE_CALLS] = yhat_rec
        for b in bases_required:
            dfp.loc[ts, b] = yhat_rec

        # 2) TMO residual
        dfp.loc[ts, "tmo_baseline"] = tmo_base
        dfp.loc[ts, "tmo_resid"] = yhat_tmo - tmo_base
        dfp.loc[ts, "feriados"] = 0 if holidays_set is None else _is_holiday(ts, holidays_set)
        dfp.loc[ts, "es_dia_de_pago"] = 0

    print("Predicción iterativa completada.")

    # Salida horaria: usamos 'recibidos' como calls para front y Erlang
    df_hourly = pd.DataFrame(index=future_ts)
    df_hourly["calls"] = np.round(dfp.loc[future_ts, FEATURE_BASE_CALLS]).astype(int)
    df_hourly["tmo_s"] = np.round(dfp.loc[future_ts, "tmo_baseline"] + dfp.loc[future_ts, "tmo_resid"]).astype(int)

    # CAP de outliers sobre calls (recibidos)
    if ENABLE_OUTLIER_CAP:
        d_hist = add_time_parts(df[[FEATURE_BASE_CALLS]].copy())
        g = d_hist.groupby(["dow","hour"])[FEATURE_BASE_CALLS]
        base = g.median().rename("med").to_frame()
        mad  = g.apply(lambda x: np.median(np.abs(x - np.median(x)))).rename("mad")
        base = base.join(mad)
        if base["mad"].isna().all():
            base["mad"] = 0
        base["mad"] = base["mad"].replace(0, base["mad"].median() if not np.isnan(base["mad"].median()) else 1.0)

        d = add_time_parts(df_hourly.copy())
        capped = d.merge(base.reset_index(), on=["dow","hour"], how="left")
        capped["mad"] = capped["mad"].fillna(capped["mad"].median() if not np.isnan(capped["mad"].median()) else 1.0)
        capped["med"] = capped["med"].fillna(capped["med"].median() if not np.isnan(capped["med"].median()) else 0.0)

        is_weekend = capped["dow"].isin([5,6]).values
        K = np.where(is_weekend, K_WEEKEND, K_WEEKDAY).astype(float)
        upper = capped["med"].values + K * capped["mad"].values
        mask = capped["calls"].astype(float).values > upper
        capped.loc[mask, "calls"] = np.round(upper[mask]).astype(int)
        df_hourly["calls"] = capped["calls"].astype(int)

    # Erlang (usa calls=recibidos, como el original que pediste)
    df_hourly["agents_prod"] = 0
    for ts in df_hourly.index:
        a, _ = required_agents(float(df_hourly.at[ts, "calls"]), float(df_hourly.at[ts, "tmo_s"]))
        df_hourly.at[ts, "agents_prod"] = int(a)
    df_hourly["agents_sched"] = df_hourly["agents_prod"].apply(schedule_agents)

    # Salidas JSON (ponderación diaria por 'calls' = recibidos)
    write_hourly_json(f"{PUBLIC_DIR}/prediccion_horaria.json", df_hourly, "calls", "tmo_s", "agents_sched")
    write_daily_json(f"{PUBLIC_DIR}/prediccion_diaria.json", df_hourly, "calls", "tmo_s", weights_col="calls")
    return df_hourly
