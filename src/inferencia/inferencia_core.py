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

# Ventana reciente para lags/MA (no afecta last_ts)
HIST_WINDOW_DAYS = 90

# ======= NUEVO: Guardrail Outliers (config) =======
ENABLE_OUTLIER_CAP = True   # <- ponlo en False si quieres desactivarlo
K_WEEKDAY = 6.0             # techos +K*MAD en lun-vie
K_WEEKEND = 7.0             # techos +K*MAD en sáb-dom


def _load_cols(path: str):
    with open(path, "r") as f:
        return json.load(f)


# ========= Helpers de FERIADOS (PORTADOS + EXTENDIDOS) =========
def _safe_ratio(num, den, fallback=1.0):
    num = float(num) if num is not None and not np.isnan(num) else np.nan
    den = float(den) if den is not None and not np.isnan(den) and den != 0 else np.nan
    if np.isnan(num) or np.isnan(den) or den == 0:
        return fallback
    return num / den


def _series_is_holiday(idx, holidays_set):
    tz = getattr(idx, "tz", None)
    idx_dates = idx.tz_convert(TIMEZONE).date if tz is not None else idx.date
    return pd.Series([d in holidays_set for d in idx_dates], index=idx, dtype=bool)


def compute_holiday_factors(df_hist, holidays_set,
                            col_calls=TARGET_CALLS, col_tmo=TARGET_TMO):
    """
    Calcula factores por HORA (mediana feriado vs normal) + factores globales,
    y además factores para el DÍA POST-FERIADO por hora.
    Basado en tu forecast3m.py.
    """
    cols = [col_calls]
    if col_tmo in df_hist.columns:
        cols.append(col_tmo)

    dfh = add_time_parts(df_hist[cols].copy())
    dfh["is_holiday"] = _series_is_holiday(dfh.index, holidays_set)

    # Medianas por hora (feriado vs normal)
    med_hol_calls = dfh[dfh["is_holiday"]].groupby("hour")[col_calls].median()
    med_nor_calls = dfh[~dfh["is_holiday"]].groupby("hour")[col_calls].median()

    if col_tmo in dfh.columns:
        med_hol_tmo = dfh[dfh["is_holiday"]].groupby("hour")[col_tmo].median()
        med_nor_tmo = dfh[~dfh["is_holiday"]].groupby("hour")[col_tmo].median()
        g_hol_tmo = dfh[dfh["is_holiday"]][col_tmo].median()
        g_nor_tmo = dfh[~dfh["is_holiday"]][col_tmo].median()
        global_tmo_factor = _safe_ratio(g_hol_tmo, g_nor_tmo, fallback=1.00)
    else:
        med_hol_tmo = med_nor_tmo = None
        global_tmo_factor = 1.00

    g_hol_calls = dfh[dfh["is_holiday"]][col_calls].median()
    g_nor_calls = dfh[~dfh["is_holiday"]][col_calls].median()
    global_calls_factor = _safe_ratio(g_hol_calls, g_nor_calls, fallback=0.75)

    factors_calls_by_hour = {
        int(h): _safe_ratio(med_hol_calls.get(h, np.nan),
                            med_nor_calls.get(h, np.nan),
                            fallback=global_calls_factor)
        for h in range(24)
    }

    if med_hol_tmo is not None:
        factors_tmo_by_hour = {
            int(h): _safe_ratio(med_hol_tmo.get(h, np.nan),
                                med_nor_tmo.get(h, np.nan),
                                fallback=global_tmo_factor)
            for h in range(24)
        }
    else:
        factors_tmo_by_hour = {int(h): 1.0 for h in range(24)}

    # Límites (más permisivo en llamadas, para no cortar picos reales)
    factors_calls_by_hour = {h: float(np.clip(v, 0.10, 1.60)) for h, v in factors_calls_by_hour.items()}
    factors_tmo_by_hour   = {h: float(np.clip(v, 0.70, 1.50)) for h, v in factors_tmo_by_hour.items()}

    # ---- NEW: factores del DÍA POST-FERIADO por hora ----
    dfh = dfh.copy()
    dfh["is_post_hol"] = (~dfh["is_holiday"]) & (dfh["is_holiday"].shift(1).fillna(False))
    med_post_calls = dfh[dfh["is_post_hol"]].groupby("hour")[col_calls].median()
    post_calls_by_hour = {
        int(h): _safe_ratio(med_post_calls.get(h, np.nan),
                            med_nor_calls.get(h, np.nan),
                            fallback=1.05)  # leve alza por defecto
        for h in range(24)
    }
    # Más margen en horas punta del rebote
    post_calls_by_hour = {h: float(np.clip(v, 0.90, 1.80)) for h, v in post_calls_by_hour.items()}

    return (factors_calls_by_hour, factors_tmo_by_hour,
            global_calls_factor, global_tmo_factor, post_calls_by_hour)


def apply_holiday_adjustment(df_future, holidays_set,
                             factors_calls_by_hour, factors_tmo_by_hour,
                             col_calls_future="calls", col_tmo_future="tmo_s"):
    """
    Aplica factores por hora SOLO en horas/fechas feriado (idéntico al original).
    """
    d = add_time_parts(df_future.copy())
    is_hol = _series_is_holiday(d.index, holidays_set)

    hours = d["hour"].astype(int).values
    call_f = np.array([factors_calls_by_hour.get(int(h), 1.0) for h in hours])
    tmo_f  = np.array([factors_tmo_by_hour.get(int(h), 1.0) for h in hours])

    out = df_future.copy()
    mask = is_hol.values
    out.loc[mask, col_calls_future] = np.round(out.loc[mask, col_calls_future].astype(float) * call_f[mask]).astype(int)
    out.loc[mask, col_tmo_future]   = np.round(out.loc[mask, col_tmo_future].astype(float)   * tmo_f[mask]).astype(int)
    return out


def apply_post_holiday_adjustment(df_future, holidays_set, post_calls_by_hour,
                                  col_calls_future="calls"):
    """
    Ajuste para el DÍA POST-FERIADO: si el día anterior fue feriado, aplicar factor por hora.
    """
    idx = df_future.index
    prev_idx = (idx - pd.Timedelta(days=1))
    try:
        prev_dates = prev_idx.tz_convert(TIMEZONE).date
        curr_dates = idx.tz_convert(TIMEZONE).date
    except Exception:
        prev_dates = prev_idx.date
        curr_dates = idx.date

    is_prev_hol = pd.Series([d in holidays_set for d in prev_dates], index=idx, dtype=bool)
    is_today_hol = pd.Series([d in holidays_set for d in curr_dates], index=idx, dtype=bool)
    is_post = (~is_today_hol) & (is_prev_hol)

    d = add_time_parts(df_future.copy())
    hours = d["hour"].astype(int).values
    ph_f = np.array([post_calls_by_hour.get(int(h), 1.0) for h in hours])

    out = df_future.copy()
    mask = is_post.values
    out.loc[mask, col_calls_future] = np.round(out.loc[mask, col_calls_future].astype(float) * ph_f[mask]).astype(int)
    return out
# ===========================================================

# ========= NUEVO: Guardrail de outliers por (dow,hour) ======
def _baseline_median_mad(df_hist, col=TARGET_CALLS):
    """
    Baseline robusto por (dow,hour): mediana y MAD.
    """
    d = add_time_parts(df_hist[[col]].copy())
    g = d.groupby(["dow", "hour"])[col]
    base = g.median().rename("med").to_frame()
    mad = g.apply(lambda x: np.median(np.abs(x - np.median(x)))).rename("mad")
    base = base.join(mad)
    # fallback si alguna combinación no tiene MAD
    if base["mad"].isna().all():
        base["mad"] = 0
    base["mad"] = base["mad"].replace(0, base["mad"].median() if not np.isnan(base["mad"].median()) else 1.0)
    return base.reset_index()  # columnas: dow, hour, med, mad


def apply_outlier_cap(df_future, base_median_mad, holidays_set,
                      col_calls_future="calls",
                      k_weekday=K_WEEKDAY, k_weekend=K_WEEKEND):
    """
    Capa picos: pred <= mediana + K*MAD (K diferente en finde).
    No actúa en feriados ni post-feriados.
    """
    if df_future.empty:
        return df_future

    d = add_time_parts(df_future.copy())
    # flags feriado/post-feriado
    prev_idx = (d.index - pd.Timedelta(days=1))
    try:
        curr_dates = d.index.tz_convert(TIMEZONE).date
        prev_dates = prev_idx.tz_convert(TIMEZONE).date
    except Exception:
        curr_dates = d.index.date
        prev_dates = prev_idx.date
    is_hol = pd.Series([dt in holidays_set for dt in curr_dates], index=d.index, dtype=bool) if holidays_set else pd.Series(False, index=d.index)
    is_prev_hol = pd.Series([dt in holidays_set for dt in prev_dates], index=d.index, dtype=bool) if holidays_set else pd.Series(False, index=d.index)
    is_post_hol = (~is_hol) & (is_prev_hol)

    # merge (dow,hour) -> med, mad
    base = base_median_mad.copy()
    capped = d.merge(base, on=["dow","hour"], how="left")
    capped["mad"] = capped["mad"].fillna(capped["mad"].median() if not np.isnan(capped["mad"].median()) else 1.0)
    capped["med"] = capped["med"].fillna(capped["med"].median() if not np.isnan(capped["med"].median()) else 0.0)

    # K por día de semana
    is_weekend = capped["dow"].isin([5,6]).values
    K = np.where(is_weekend, k_weekend, k_weekday).astype(float)

    # techo
    upper = capped["med"].values + K * capped["mad"].values

    # máscara: solo cuando NO es feriado ni post-feriado
    mask = (~is_hol.values) & (~is_post_hol.values) & (capped[col_calls_future].astype(float).values > upper)
    capped.loc[mask, col_calls_future] = np.round(upper[mask]).astype(int)

    out = df_future.copy()
    out[col_calls_future] = capped[col_calls_future].astype(int).values
    return out
# ===========================================================


def _is_holiday(ts, holidays_set: set) -> int:
    if not holidays_set:
        return 0
    try:
        d = ts.tz_convert(TIMEZONE).date()
    except Exception:
        d = ts.date()
    return 1 if d in holidays_set else 0


def forecast_120d(df_hist_calls: pd.DataFrame, horizon_days: int = 120, holidays_set: set | None = None):
    """
    - Parser robusto (igual al repo bueno).
    - Filtro dropna(subset=[TARGET_CALLS]) (sin cap a hoy).
    - Horizonte = 1h después de last_ts.
    - Planner iterativo con 'feriados' también en FUTURO.
    - TMO horario (con 'feriados' futuro si aplica).
    - Ajuste post-forecast por FERIADOS + POST-FERIADOS.
    - (Opcional) CAP de OUTLIERS por (dow,hour) con mediana+MAD.
    - Erlang C y salidas JSON.
    """
    # === Artefactos ===
    m_pl = tf.keras.models.load_model(PLANNER_MODEL, compile=False)
    sc_pl = joblib.load(PLANNER_SCALER)
    cols_pl = _load_cols(PLANNER_COLS)

    m_tmo = tf.keras.models.load_model(TMO_MODEL, compile=False)
    sc_tmo = joblib.load(TMO_SCALER)
    cols_tmo = _load_cols(TMO_COLS)

    # === Base histórica ===
    df = ensure_ts(df_hist_calls)

    if TARGET_CALLS not in df.columns:
        raise ValueError(f"Falta columna {TARGET_CALLS} en historical_data.csv")

    df = df[[TARGET_CALLS, TARGET_TMO] if TARGET_TMO in df.columns else [TARGET_CALLS]].copy()
    df = df.dropna(subset=[TARGET_CALLS])

    # ff de auxiliares (si existen en histórico)
    for aux in ["feriados", "es_dia_de_pago", "tmo_comercial", "tmo_tecnico",
                "proporcion_comercial", "proporcion_tecnica"]:
        if aux in df.columns:
            df[aux] = df[aux].ffill()

    last_ts = df.index.max()

    start_hist = last_ts - pd.Timedelta(days=HIST_WINDOW_DAYS)
    df_recent = df.loc[df.index >= start_hist].copy()
    if df_recent.empty:
        df_recent = df.copy()

    # ===== Horizonte futuro =====
    future_ts = pd.date_range(
        last_ts + pd.Timedelta(hours=1),
        periods=horizon_days * 24,
        freq="h",
        tz=TIMEZONE
    )

    # ===== Planner iterativo (con 'feriados' futuro) =====
    if "feriados" in df_recent.columns:
        dfp = df_recent[[TARGET_CALLS, "feriados"]].copy()
    else:
        dfp = df_recent[[TARGET_CALLS]].copy()

    dfp[TARGET_CALLS] = pd.to_numeric(dfp[TARGET_CALLS], errors="coerce").ffill().fillna(0.0)

    for ts in future_ts:
        tmp = pd.concat([dfp, pd.DataFrame(index=[ts])])
        tmp[TARGET_CALLS] = tmp[TARGET_CALLS].ffill()

        if "feriados" in tmp.columns:
            tmp.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)

        tmp = add_lags_mas(tmp, TARGET_CALLS)
        tmp = add_time_parts(tmp)

        X = dummies_and_reindex(tmp.tail(1), cols_pl)
        yhat = float(m_pl.predict(sc_pl.transform(X), verbose=0).flatten()[0])
        dfp.loc[ts, TARGET_CALLS] = max(0.0, yhat)

        if "feriados" in dfp.columns:
            dfp.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)

    pred_calls = dfp.loc[future_ts, TARGET_CALLS]

    # ===== TMO por hora =====
    base_tmo = pd.DataFrame(index=future_ts)
    base_tmo[TARGET_CALLS] = pred_calls.values

    if {"proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico"}.issubset(df.columns):
        last_vals = df.ffill().iloc[[-1]][["proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico"]]
    else:
        last_vals = pd.DataFrame([[0,0,0,0]], columns=["proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico"])

    for c in ["proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico"]:
        base_tmo[c] = float(last_vals[c].iloc[0]) if c in last_vals.columns else 0.0

    if "feriados" in df.columns:
        base_tmo["feriados"] = [_is_holiday(ts, holidays_set) for ts in base_tmo.index]

    base_tmo = add_time_parts(base_tmo)
    Xt = dummies_and_reindex(base_tmo, cols_tmo)
    y_tmo = m_tmo.predict(sc_tmo.transform(Xt), verbose=0).flatten()
    y_tmo = np.maximum(0, y_tmo)

    # ===== Curva base (sin ajuste) =====
    df_hourly = pd.DataFrame(index=future_ts)
    df_hourly["calls"] = np.round(pred_calls).astype(int)
    df_hourly["tmo_s"] = np.round(y_tmo).astype(int)

    # ===== AJUSTE POR FERIADOS =====
    if holidays_set and len(holidays_set) > 0:
        (f_calls_by_hour, f_tmo_by_hour,
         g_calls, g_tmo, post_calls_by_hour) = compute_holiday_factors(df, holidays_set)

        # Feriados
        df_hourly = apply_holiday_adjustment(
            df_hourly, holidays_set,
            f_calls_by_hour, f_tmo_by_hour,
            col_calls_future="calls", col_tmo_future="tmo_s"
        )

        # Post-feriado (día siguiente)
        df_hourly = apply_post_holiday_adjustment(
            df_hourly, holidays_set, post_calls_by_hour,
            col_calls_future="calls"
        )

    # ===== (OPCIONAL) CAP de OUTLIERS =====
    if ENABLE_OUTLIER_CAP:
        base_mad = _baseline_median_mad(df, col=TARGET_CALLS)
        df_hourly = apply_outlier_cap(
            df_hourly, base_mad, holidays_set,
            col_calls_future="calls",
            k_weekday=K_WEEKDAY, k_weekend=K_WEEKEND
        )

    # ===== Erlang por hora =====
    df_hourly["agents_prod"] = 0
    for ts in df_hourly.index:
        a, _ = required_agents(float(df_hourly.at[ts, "calls"]), float(df_hourly.at[ts, "tmo_s"]))
        df_hourly.at[ts, "agents_prod"] = int(a)
    df_hourly["agents_sched"] = df_hourly["agents_prod"].apply(schedule_agents)

    # ===== Salidas =====
    write_hourly_json(f"{PUBLIC_DIR}/prediccion_horaria.json",
                      df_hourly, "calls", "tmo_s", "agents_sched")
    write_daily_json(f"{PUBLIC_DIR}/prediccion_diaria.json",
                     df_hourly, "calls", "tmo_s")

    return df_hourly
