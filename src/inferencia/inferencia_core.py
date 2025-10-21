# src/inferencia/inferencia_core.py
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from .features import ensure_ts, add_time_parts, add_lags_mas, dummies_and_reindex
from .erlang import required_agents, schedule_agents
from .utils_io import write_daily_json, write_hourly_json

TIMEZONE = "America/Santiago"
PUBLIC_DIR = "public"

PLANNER_MODEL = "models/modelo_planner.keras"
PLANNER_SCALER = "models/scaler_planner.pkl"
PLANNER_COLS = "models/training_columns_planner.json"

TMO_MODEL = "models/modelo_tmo.keras"
TMO_SCALER = "models/scaler_tmo.pkl"
TMO_COLS = "models/training_columns_tmo.json"

TARGET_CALLS = "recibidos_nacional"
TARGET_TMO = "tmo_general"

HIST_WINDOW_DAYS = 90

ENABLE_OUTLIER_CAP = True
K_WEEKDAY = 6.0
K_WEEKEND = 7.0

def _load_cols(path: str):
    with open(path, "r") as f:
        return json.load(f)

# ---------------- Feriados helpers ----------------
def _safe_ratio(num, den, fallback=1.0):
    num = float(num) if num is not None and not np.isnan(num) else np.nan
    den = float(den) if den is not None and not np.isnan(den) and den != 0 else np.nan
    if np.isnan(num) or np.isnan(den) or den == 0:
        return fallback
    return num/den

def _series_is_holiday(idx, holidays_set):
    tz = getattr(idx, "tz", None)
    idx_dates = idx.tz_convert(TIMEZONE).date if tz is not None else idx.date
    return pd.Series([d in holidays_set for d in idx_dates], index=idx, dtype="boolean")

def _is_holiday(ts, holidays_set):
    try:
        d = ts.tz_convert(TIMEZONE).date()
    except Exception:
        d = ts.date()
    return d in holidays_set

def compute_holiday_factors(df_hist, holidays_set,
                            col_calls=TARGET_CALLS, col_tmo=TARGET_TMO):
    cols = [col_calls] + ([col_tmo] if col_tmo in df_hist.columns else [])
    dfh = add_time_parts(df_hist[cols].copy())
    dfh["is_holiday"] = _series_is_holiday(dfh.index, holidays_set)

    med_hol_calls = dfh[dfh["is_holiday"] == True].groupby("hour")[col_calls].median()
    med_nor_calls = dfh[dfh["is_holiday"] != True].groupby("hour")[col_calls].median()

    if col_tmo in dfh.columns:
        med_hol_tmo = dfh[dfh["is_holiday"] == True].groupby("hour")[col_tmo].median()
        med_nor_tmo = dfh[dfh["is_holiday"] != True].groupby("hour")[col_tmo].median()
        g_hol_tmo = dfh[dfh["is_holiday"] == True][col_tmo].median()
        g_nor_tmo = dfh[dfh["is_holiday"] != True][col_tmo].median()
        global_tmo_factor = _safe_ratio(g_hol_tmo, g_nor_tmo, fallback=1.00)
    else:
        med_hol_tmo = med_nor_tmo = None
        global_tmo_factor = 1.00

    g_hol_calls = dfh[dfh["is_holiday"] == True][col_calls].median()
    g_nor_calls = dfh[dfh["is_holiday"] != True][col_calls].median()
    global_calls_factor = _safe_ratio(g_hol_calls, g_nor_calls, fallback=0.75)

    factors_calls_by_hour = {int(h): _safe_ratio(med_hol_calls.get(h, np.nan),
                                                 med_nor_calls.get(h, np.nan),
                                                 fallback=global_calls_factor)
                             for h in range(24)}
    if med_hol_tmo is not None:
        factors_tmo_by_hour = {int(h): _safe_ratio(med_hol_tmo.get(h, np.nan),
                                                   med_nor_tmo.get(h, np.nan),
                                                   fallback=global_tmo_factor)
                               for h in range(24)}
    else:
        factors_tmo_by_hour = {int(h): 1.0 for h in range(24)}

    # límites
    factors_calls_by_hour = {h: float(np.clip(v, 0.10, 1.60)) for h, v in factors_calls_by_hour.items()}
    factors_tmo_by_hour   = {h: float(np.clip(v, 0.70, 1.50)) for h, v in factors_tmo_by_hour.items()}

    # post-feriado
    dfh = dfh.copy()
    dfh["is_post_hol"] = (~(dfh["is_holiday"] == True)) & (dfh["is_holiday"].shift(1, fill_value=False) == True)
    med_nor_calls = dfh[dfh["is_holiday"] != True].groupby("hour")[col_calls].median()
    med_post_calls = dfh[dfh["is_post_hol"] == True].groupby("hour")[col_calls].median()
    post_calls_by_hour = {int(h): _safe_ratio(med_post_calls.get(h, np.nan),
                                              med_nor_calls.get(h, np.nan),
                                              fallback=1.05)
                          for h in range(24)}
    post_calls_by_hour = {h: float(np.clip(v, 0.90, 1.80)) for h, v in post_calls_by_hour.items()}

    return (factors_calls_by_hour, factors_tmo_by_hour,
            global_calls_factor, global_tmo_factor, post_calls_by_hour)

def apply_holiday_adjustment(df_future, holidays_set,
                             factors_calls_by_hour, factors_tmo_by_hour,
                             col_calls_future="calls", col_tmo_future="tmo_s"):
    if df_future.empty:
        return df_future
    is_hol = _series_is_holiday(df_future.index, holidays_set)
    d = add_time_parts(df_future.copy())
    hours = d["hour"].astype(int).values
    call_f = np.array([factors_calls_by_hour.get(int(h), 1.0) for h in hours])
    tmo_f  = np.array([factors_tmo_by_hour.get(int(h), 1.0) for h in hours])

    out = df_future.copy()
    mask = (is_hol == True).values

    out[col_calls_future] = out[col_calls_future].astype(float)
    out[col_tmo_future]   = out[col_tmo_future].astype(float)

    out.loc[mask, col_calls_future] = np.round(out.loc[mask, col_calls_future].values * call_f[mask]).astype(float)
    out.loc[mask, col_tmo_future]   = np.round(out.loc[mask, col_tmo_future].values   * tmo_f[mask]).astype(float)
    return out

def apply_post_holiday_adjustment(df_future, holidays_set, post_calls_by_hour,
                                  col_calls_future="calls"):
    if df_future.empty:
        return df_future
    idx = df_future.index
    prev_idx = (idx - pd.Timedelta(days=1))
    try:
        prev_dates = prev_idx.tz_convert(TIMEZONE).date
        curr_dates = idx.tz_convert(TIMEZONE).date
    except Exception:
        prev_dates = prev_idx.date
        curr_dates = idx.date
    is_prev_hol = pd.Series([d in holidays_set for d in prev_dates], index=idx, dtype="boolean")
    is_today_hol = pd.Series([d in holidays_set for d in curr_dates], index=idx, dtype="boolean")
    is_post = (is_today_hol != True) & (is_prev_hol == True)

    d = add_time_parts(df_future.copy())
    hours = d["hour"].astype(int).values
    ph_f = np.array([post_calls_by_hour.get(int(h), 1.0) for h in hours])

    out = df_future.copy()
    mask = is_post.values
    out[col_calls_future] = out[col_calls_future].astype(float)
    new_vals = ph_f[mask].astype(float) * out.loc[mask, col_calls_future].values.astype(float)
    bad = ~np.isfinite(new_vals)
    if bad.any():
        orig = out.loc[mask, col_calls_future].values.astype(float)
        new_vals[bad] = orig[bad]
    out.loc[mask, col_calls_future] = np.round(new_vals).astype(float)
    return out

# -------------- Outlier cap robusto --------------
def _baseline_median_mad(df_hist, col=TARGET_CALLS):
    d = add_time_parts(df_hist[[col]].copy())
    g = d.groupby(["dow","hour"])[col]
    base = g.median().rename("med").to_frame()
    mad  = g.apply(lambda x: np.median(np.abs(x - np.median(x)))).rename("mad")
    base = base.join(mad)
    if base["mad"].isna().all():
        base["mad"] = 0
    base["mad"] = base["mad"].replace(0, base["mad"].median() if not np.isnan(base["mad"].median()) else 1.0)
    return base.reset_index()

def apply_outlier_cap(df_future, base_median_mad, holidays_set,
                      col_calls_future="calls",
                      k_weekday=K_WEEKDAY, k_weekend=K_WEEKEND):
    if df_future.empty:
        return df_future
    d = add_time_parts(df_future.copy())
    d = d.merge(base_median_mad, on=["dow","hour"], how="left")

    limits = d["med"].astype(float).fillna(0.0) + d["mad"].astype(float).fillna(0.0) * \
             np.where(d["dow"].isin([5,6]), float(k_weekend), float(k_weekday))

    idx = df_future.index
    try:
        curr_dates = idx.tz_convert(TIMEZONE).date
        prev_dates = (idx - pd.Timedelta(days=1)).tz_convert(TIMEZONE).date
    except Exception:
        curr_dates = idx.date
        prev_dates = (idx - pd.Timedelta(days=1)).date

    is_hol = pd.Series([d in holidays_set for d in curr_dates], index=idx, dtype="boolean")
    is_prev_hol = pd.Series([d in holidays_set for d in prev_dates], index=idx, dtype="boolean")
    is_post = (is_hol != True) & (is_prev_hol == True)
    mask = ((is_hol != True) & (is_post != True)).values

    out = df_future.copy()
    out[col_calls_future] = out[col_calls_future].astype(float)

    # mínimo seguro con NaN-safe
    new_vals = np.minimum(out.loc[mask, col_calls_future].values.astype(float),
                          limits[mask].astype(float))
    bad = ~np.isfinite(new_vals)
    if bad.any():
        orig = out.loc[mask, col_calls_future].values.astype(float)
        new_vals[bad] = orig[bad]
    out.loc[mask, col_calls_future] = new_vals
    out[col_calls_future] = out[col_calls_future].fillna(0.0).astype(float)
    return out

# ---------------- Forecast principal ----------------
def forecast_120d(df_hist_calls: pd.DataFrame,
                  horizon_days: int,
                  holidays_set: set):
    """
    df_hist_calls: DataFrame YA indexado por ts, con TARGET_CALLS y (si existen) columnas TMO auxiliares.
    Retorna DF horario con: calls, tmo_s (luego se overridea en main.py), agentes_requeridos.
    También escribe prediccion_horaria.json y prediccion_diaria.json (compatibilidad).
    """
    # Artefactos
    m_pl = tf.keras.models.load_model(PLANNER_MODEL, compile=False)
    sc_pl = joblib.load(PLANNER_SCALER)
    cols_pl = _load_cols(PLANNER_COLS)

    m_tmo = tf.keras.models.load_model(TMO_MODEL, compile=False)
    sc_tmo = joblib.load(TMO_SCALER)
    cols_tmo = _load_cols(TMO_COLS)

    # Base histórica
    df = ensure_ts(df_hist_calls)
    if getattr(df.index, "tz", None) is None:
        df.index = df.index.tz_localize(TIMEZONE)

    if TARGET_CALLS not in df.columns:
        raise ValueError(f"Falta columna {TARGET_CALLS} en historical_data.")

    aux_cols = [
        TARGET_TMO,
        "proporcion_comercial","proporcion_tecnica",
        "tmo_comercial","tmo_tecnico",
        "feriados","es_dia_de_pago",
    ]
    keep = [TARGET_CALLS] + [c for c in aux_cols if c in df.columns]
    df = df[keep].copy()
    df = df.dropna(subset=[TARGET_CALLS])

    for c in ["tmo_comercial","tmo_tecnico","proporcion_comercial","proporcion_tecnica",TARGET_TMO]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").ffill()
    for c in ["feriados","es_dia_de_pago"]:
        if c in df.columns:
            df[c] = df[c].astype("boolean").ffill()

    last_ts = df.index.max()

    # Ventana reciente
    start_hist = last_ts - pd.Timedelta(days=HIST_WINDOW_DAYS)
    df_recent = df.loc[df.index >= start_hist].copy()
    if df_recent.empty:
        df_recent = df.copy()

    # Horizonte futuro
    future_ts = pd.date_range(last_ts + pd.Timedelta(hours=1),
                              periods=horizon_days*24, freq="h", tz=TIMEZONE)

    # Planner de llamadas (iterativo)
    if "feriados" in df_recent.columns:
        dfp = df_recent[[TARGET_CALLS, "feriados"]].copy()
        dfp["feriados"] = dfp["feriados"].astype("boolean")
    else:
        dfp = df_recent[[TARGET_CALLS]].copy()
    dfp[TARGET_CALLS] = pd.to_numeric(dfp[TARGET_CALLS], errors="coerce").ffill().fillna(0.0)

    for ts in future_ts:
        tmp = pd.concat([dfp, pd.DataFrame(index=[ts])])
        tmp[TARGET_CALLS] = tmp[TARGET_CALLS].ffill()
        if "feriados" in tmp.columns:
            tmp.loc[ts, "feriados"] = bool(_is_holiday(ts, holidays_set))
        tmp = add_lags_mas(tmp, TARGET_CALLS)
        tmp = add_time_parts(tmp)
        X = dummies_and_reindex(tmp.tail(1), cols_pl)
        yhat = float(m_pl.predict(sc_pl.transform(X), verbose=0).flatten()[0])
        dfp.loc[ts, TARGET_CALLS] = max(0.0, yhat)
        if "feriados" in dfp.columns:
            dfp.loc[ts, "feriados"] = bool(_is_holiday(ts, holidays_set))

    pred_calls = dfp.loc[future_ts, TARGET_CALLS]

    # TMO interno (compatibilidad; main.py lo sobreescribe)
    base_tmo = pd.DataFrame(index=future_ts)
    base_tmo[TARGET_CALLS] = pred_calls.values
    if {"proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico"}.issubset(df.columns):
        last_vals = df.ffill().iloc[[-1]][["proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico"]]
    else:
        last_vals = pd.DataFrame([[0,0,0,0]],
                                 columns=["proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico"])
    for c in ["proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico"]:
        base_tmo[c] = float(last_vals[c].iloc[0]) if c in last_vals.columns else 0.0
    if "feriados" in df.columns:
        base_tmo["feriados"] = pd.Series([bool(_is_holiday(ts, holidays_set)) for ts in base_tmo.index],
                                         index=base_tmo.index, dtype="boolean")
    base_tmo = add_time_parts(base_tmo)
    Xt = dummies_and_reindex(base_tmo, cols_tmo)
    y_tmo = m_tmo.predict(sc_tmo.transform(Xt), verbose=0).flatten()
    y_tmo = np.maximum(0, y_tmo)

    # Curva base
    df_hourly = pd.DataFrame(index=future_ts)
    df_hourly["calls"] = np.round(pred_calls).astype(float)
    df_hourly["tmo_s"] = np.round(y_tmo).astype(float)

    # Ajustes por feriados
    if holidays_set and len(holidays_set) > 0:
        (f_calls_by_hour, f_tmo_by_hour, g_calls, g_tmo, post_calls_by_hour) = compute_holiday_factors(df, holidays_set)
        df_hourly = apply_holiday_adjustment(df_hourly, holidays_set, f_calls_by_hour, f_tmo_by_hour,
                                             col_calls_future="calls", col_tmo_future="tmo_s")
        df_hourly = apply_post_holiday_adjustment(df_hourly, holidays_set, post_calls_by_hour,
                                                  col_calls_future="calls")

    # Cap de outliers (opcional)
    if ENABLE_OUTLIER_CAP:
        base_med_mad = _baseline_median_mad(df, col=TARGET_CALLS)
        df_hourly = apply_outlier_cap(df_hourly, base_med_mad, holidays_set,
                                      col_calls_future="calls",
                                      k_weekday=K_WEEKDAY, k_weekend=K_WEEKEND)

    # Fallback si todo quedó 0
    if float(df_hourly["calls"].sum()) == 0.0 and len(df) > 0:
        base = add_time_parts(df[[TARGET_CALLS]].copy()).groupby(["dow","hour"])[TARGET_CALLS].median()
        dtmp = add_time_parts(pd.DataFrame(index=df_hourly.index))
        fb = dtmp.apply(lambda r: base.get((int(r["dow"]), int(r["hour"])), np.nan), axis=1)
        df_hourly["calls"] = np.round(pd.to_numeric(fb, errors="coerce").fillna(0.0)).astype(float)

    # Tipos finales (ints seguros)
    df_hourly["calls"] = np.round(df_hourly["calls"].astype(float)).fillna(0.0).astype(int)
    df_hourly["tmo_s"] = np.round(df_hourly["tmo_s"].astype(float)).fillna(0.0).astype(int)

    # Agentes
    try:
        df_hourly["agentes_requeridos"] = required_agents(
            traffic_calls=df_hourly["calls"].astype(float).values,
            aht_seconds=df_hourly["tmo_s"].astype(float).values
        ).astype(int)
    except Exception:
        df_hourly["agentes_requeridos"] = (df_hourly["calls"]/20).round().astype(int)

    # JSONs (compatibilidad; main.py reescribe luego)
    write_hourly_json(f"{PUBLIC_DIR}/prediccion_horaria.json", df_hourly,
                      calls_col="calls", tmo_col="tmo_s", agentes_col="agentes_requeridos")
    write_daily_json(f"{PUBLIC_DIR}/prediccion_diaria.json", df_hourly,
                     calls_col="calls", tmo_col="tmo_s")
    return df_hourly

