# src/inferencia/inferencia_core.py
# =============================================================================
# Núcleo de inferencia — Planner (llamadas) + TMO autoregresivo + Erlang + Export
# Ajustes:
#  - Flags ENV para control fino del planner (feriados, clip).
#  - TMO AR con blend a mediana por hora y cap por cuantil horario.
#  - Regularización (MAD, límites) + calibración opcional.
#  - JSON pretty (indent=2).
# Compat: horizon_days (alias de horizonte_dias), holidays_set (alias feriados).
# =============================================================================

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Iterable, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import load as joblib_load

# Módulos del repo
from src.inferencia.features import (
    ensure_ts,
    add_lags_mas,
    dummies_and_reindex,
)
from src.inferencia.erlang import required_agents, schedule_agents

# =============================================================================
# Constantes / Flags (tuneables por ENV)
# =============================================================================
TZ = "America/Santiago"
PUBLIC_DIR = Path("public")
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

# Artefactos esperados en models/
PLANNER_MODEL_NAME = "modelo_planner.keras"
PLANNER_SCALER_NAME = "scaler_planner.pkl"
PLANNER_COLS_NAME = "training_columns_planner.json"

TMO_MODEL_NAME = "modelo_tmo.keras"
TMO_SCALER_NAME = "scaler_tmo.pkl"
TMO_COLS_NAME = "training_columns_tmo.json"

# Columnas objetivo típicas en tus datasets
TARGET_CALLS_CANDIDATES = ["calls", "recibidos_nacional", "recibidos", "llamadas", "q_llamadas_general"]
TARGET_TMO_CANDIDATES   = ["tmo_general", "tmo", "aht"]
FERIADOS_CANDIDATES     = ["feriados", "feriado", "is_holiday", "holiday", "es_feriado"]

# Planner flags
PLANNER_HOLIDAY_ADJ = int(os.getenv("PLANNER_HOLIDAY_ADJ", "0"))  # 0=off (default), 1=on
PLANNER_CLIP_MULT   = float(os.getenv("PLANNER_CLIP_MULT", "0"))  # 0=off, >0 multiplica mediana para techo

# TMO regularización / blending
TMO_MAD_K       = float(os.getenv("TMO_MAD_K", "3.5"))
TMO_MIN_S       = float(os.getenv("TMO_MIN_S", "80"))
TMO_MAX_S       = float(os.getenv("TMO_MAX_S", "900"))
TMO_BLEND_ALPHA = float(os.getenv("TMO_BLEND_ALPHA", "0.75"))  # 0..1 (peso del modelo)
TMO_HOURLY_QHI  = float(os.getenv("TMO_HOURLY_QHI", "0.95"))   # 0 desactiva cap por cuantil

# =============================================================================
# Helpers generales
# =============================================================================

def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)  # pretty-print

def _safe_ratio(a, b):
    return np.array(a, dtype=float) / (np.array(b, dtype=float) + 1e-6)

def _normalize_calls_and_holidays(df_calls: pd.DataFrame) -> pd.DataFrame:
    df = df_calls.copy()
    cols_lower = {c.lower(): c for c in df.columns}

    src_calls = next((cols_lower[c] for c in TARGET_CALLS_CANDIDATES if c in cols_lower), None)
    if src_calls is None:
        raise KeyError(f"No se encontró columna de llamadas. Candidatas: {TARGET_CALLS_CANDIDATES}. Columns={list(df.columns)}")
    if src_calls != "calls":
        df = df.rename(columns={src_calls: "calls"})

    src_h = next((cols_lower[c] for c in FERIADOS_CANDIDATES if c in cols_lower), None)
    if src_h is None:
        df["feriados"] = 0
    elif src_h != "feriados":
        df = df.rename(columns={src_h: "feriados"})

    df["calls"] = pd.to_numeric(df["calls"], errors="coerce")
    df["feriados"] = pd.to_numeric(df["feriados"], errors="coerce").fillna(0).astype(int)
    return df

def _normalize_tmo(df_tmo: pd.DataFrame) -> pd.DataFrame:
    df = df_tmo.copy()
    cols_lower = {c.lower(): c for c in df.columns}
    src_tmo = next((cols_lower[c] for c in TARGET_TMO_CANDIDATES if c in cols_lower), None)
    if src_tmo is None and "tmo_s" not in df.columns:
        raise KeyError(f"No se encontró columna de TMO. Candidatas: {TARGET_TMO_CANDIDATES}. Columns={list(df.columns)}")
    if src_tmo and src_tmo != "tmo_s":
        df = df.rename(columns={src_tmo: "tmo_s"})
    df["tmo_s"] = pd.to_numeric(df["tmo_s"], errors="coerce")
    return df

def _build_horizon_index(last_ts: pd.Timestamp, horas: int) -> pd.DatetimeIndex:
    start = last_ts + pd.Timedelta(hours=1)
    return pd.date_range(start=start, periods=horas, freq="h", tz=TZ)

def _calendar_series_from_holidays_set(idx: pd.DatetimeIndex, holidays_set: Iterable[Any]) -> pd.Series:
    dates = set()
    for d in holidays_set:
        if isinstance(d, pd.Timestamp):
            dates.add(d.date())
        else:
            try:
                dates.add(pd.Timestamp(d).date())
            except Exception:
                pass
    if not dates:
        return pd.Series(0, index=idx, dtype=int)
    return pd.Series([1 if ts.date() in dates else 0 for ts in idx], index=idx, dtype=int)

def _calendar_series_for_horizon(idx: pd.DatetimeIndex, feriados_df: pd.DataFrame | None) -> pd.Series:
    if feriados_df is None or "feriados" not in feriados_df.columns:
        return pd.Series(0, index=idx, dtype=int)
    fer = ensure_ts(feriados_df)[["feriados"]].astype(int)
    fer = fer.reindex(idx, method="nearest", tolerance=pd.Timedelta("1H")).fillna(0).astype(int)["feriados"]
    fer.index = idx
    return fer

def compute_holiday_factors(calls_hist: pd.Series, feriados_hist: pd.Series) -> dict[int, float]:
    df = pd.DataFrame({"calls": calls_hist, "fer": feriados_hist.reindex(calls_hist.index).fillna(0).astype(int)}).dropna()
    df["hour"] = df.index.hour
    factors = {}
    for h, grp in df.groupby("hour"):
        med_n = grp.loc[grp["fer"] == 0, "calls"].median()
        med_f = grp.loc[grp["fer"] == 1, "calls"].median()
        if pd.isna(med_n) or med_n == 0:
            factors[h] = 1.0
        else:
            factors[h] = float(med_f / med_n) if not pd.isna(med_f) else 1.0
    return factors

def apply_holiday_adjustment(calls_series: pd.Series, feriados_series: pd.Series, hour_factors: dict[int, float]) -> pd.Series:
    out = []
    for ts, v in calls_series.items():
        out.append(v * hour_factors.get(ts.hour, 1.0) if int(feriados_series.get(ts, 0)) == 1 else v)
    return pd.Series(out, index=calls_series.index)

# =============================================================================
# Regularización / calibración del TMO
# =============================================================================

def _mad_cap(series: pd.Series, k: float = 3.5) -> pd.Series:
    med = series.median()
    mad = (series - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return series.clip(lower=series.quantile(0.01), upper=series.quantile(0.99))
    lower = med - k * 1.4826 * mad
    upper = med + k * 1.4826 * mad
    return series.clip(lower=lower, upper=upper)

def _clamp_tmo_bounds(s: pd.Series, min_s: float = 80.0, max_s: float = 900.0) -> pd.Series:
    return s.clip(lower=min_s, upper=max_s)

def _load_calibration_factor(default: float | None = None, path: str | Path = "models/tmo_calibration.json") -> float | None:
    env_val = os.getenv("TMO_CAL_FACTOR")
    if env_val:
        try:
            return float(env_val)
        except Exception:
            pass
    try:
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "factor" in data:
                return float(data["factor"])
    except Exception:
        pass
    return default

def _apply_calibration(series: pd.Series, factor: float | None, lo: float = 0.7, hi: float = 1.3) -> pd.Series:
    if factor is None:
        return series
    f = float(max(lo, min(hi, factor)))
    return series * f

# =============================================================================
# Carga de artefactos
# =============================================================================

def _load_planner(models_dir: str | Path):
    models_dir = Path(models_dir)
    model = tf.keras.models.load_model(models_dir / PLANNER_MODEL_NAME)
    scaler = joblib_load(models_dir / PLANNER_SCALER_NAME)
    train_cols = json.loads((models_dir / PLANNER_COLS_NAME).read_text(encoding="utf-8"))
    return model, scaler, train_cols

def _load_tmo(models_dir: str | Path):
    models_dir = Path(models_dir)
    model = tf.keras.models.load_model(models_dir / TMO_MODEL_NAME)
    scaler = joblib_load(models_dir / TMO_SCALER_NAME)
    train_cols = json.loads((models_dir / TMO_COLS_NAME).read_text(encoding="utf-8"))
    return model, scaler, train_cols

# =============================================================================
# Planner (llamadas) – inferencia feed-forward
# =============================================================================

def _planner_row(ts: pd.Timestamp, calls_hist: dict[pd.Timestamp, float], feriado: int) -> dict:
    def lag(h: int): return float(calls_hist.get(ts - pd.Timedelta(hours=h), np.nan))
    def ma(w: int):
        end = ts - pd.Timedelta(hours=1)
        vals = [calls_hist.get(end - pd.Timedelta(hours=i), np.nan) for i in range(w)]
        vals = [float(v) for v in vals if pd.notna(v)]
        return float(np.mean(vals)) if vals else np.nan

    return {
        "lag_24": lag(24), "lag_48": lag(48), "lag_72": lag(72), "lag_168": lag(168),
        "ma_24": ma(24), "ma_72": ma(72), "ma_168": ma(168),
        "sin_hour": float(np.sin(2 * np.pi * ts.hour / 24)),
        "cos_hour": float(np.cos(2 * np.pi * ts.hour / 24)),
        "sin_dow": float(np.sin(2 * np.pi * ts.dayofweek / 7)),
        "cos_dow": float(np.cos(2 * np.pi * ts.dayofweek / 7)),
        "feriados": int(feriado),
        "es_dia_de_pago": int(ts.day in {1, 2, 15, 16, 29, 30, 31}),
        "dow": ts.dayofweek, "month": ts.month, "hour": ts.hour,
    }

def predict_calls_horizonte(
    idx_horizonte: pd.DatetimeIndex,
    calls_hist_series: pd.Series,
    feriados_series: pd.Series,
    models_dir: str | Path = "models",
) -> pd.Series:
    model, scaler, train_cols = _load_planner(models_dir)
    calls_hist = calls_hist_series.to_dict()
    preds = []
    for ts in idx_horizonte:
        row = _planner_row(ts, calls_hist, int(feriados_series.get(ts, 0)))
        X = dummies_and_reindex(pd.DataFrame([row], index=[ts]), train_cols)
        if X.isna().any().any():
            X = X.ffill(axis=1).infer_objects(copy=False).fillna(0)
        y_hat = float(model.predict(scaler.transform(X.values), verbose=0).ravel()[0])
        preds.append((ts, y_hat))
        calls_hist[ts] = y_hat  # feed-forward
    s = pd.Series(dict(preds)).sort_index()
    # Guardrail opcional por ENV
    if PLANNER_CLIP_MULT and PLANNER_CLIP_MULT > 0:
        med = s.median()
        s = s.clip(lower=0, upper=max(med * PLANNER_CLIP_MULT, s.quantile(0.999)))
    return s

# =============================================================================
# TMO — inferencia feed-forward (autoregresivo)
# =============================================================================

def _predict_tmo_feedforward(
    idx_h: pd.DatetimeIndex,
    df_tmo_hist: pd.DataFrame,     # 'tmo_s'; opcionalmente q_llamadas_* para proporciones
    df_calls_pred: pd.Series,      # llamadas predichas (ajustadas si corresponde)
    holidays_set: set | None,
    m_tmo, sc_tmo, cols_tmo: list[str],
) -> pd.Series:
    d_hist = df_tmo_hist.copy()
    if "q_llamadas_general" in d_hist.columns:
        d_hist["proporcion_comercial"] = _safe_ratio(d_hist.get("q_llamadas_comercial", 0), d_hist["q_llamadas_general"])
        d_hist["proporcion_tecnica"]   = _safe_ratio(d_hist.get("q_llamadas_tecnico",   0), d_hist["q_llamadas_general"])
    p_com = float(d_hist.get("proporcion_comercial", pd.Series([0.55])).iloc[-1]) if "proporcion_comercial" in d_hist.columns else 0.55
    p_tec = float(d_hist.get("proporcion_tecnica",  pd.Series([0.45])).iloc[-1]) if "proporcion_tecnica"   in d_hist.columns else 0.45

    # Estado de TMO para lags/MAs (historia + pred)
    tmo_state = d_hist["tmo_s"].astype(float).to_dict()

    preds = []
    for ts in idx_h:
        base = {
            "feriados": 1 if (holidays_set and ts.date() in holidays_set) else 0,
            "dow": ts.dayofweek, "month": ts.month, "hour": ts.hour, "day": ts.day,
            "sin_hour": float(np.sin(2 * np.pi * ts.hour / 24)),
            "cos_hour": float(np.cos(2 * np.pi * ts.hour / 24)),
            "sin_dow": float(np.sin(2 * np.pi * ts.dayofweek / 7)),
            "cos_dow": float(np.cos(2 * np.pi * ts.dayofweek / 7)),
            "es_dia_de_pago": int(ts.day in {1, 2, 15, 16, 29, 30, 31}),
            "recibidos_nacional": float(df_calls_pred.get(ts, np.nan)),
            "proporcion_comercial": p_com,
            "proporcion_tecnica":  p_tec,
        }
        df_row = pd.DataFrame([base], index=[ts])

        # Calcula lags/MAs sobre estado (hist+pred)
        df_for_lags = pd.DataFrame({"tmo_s": pd.Series(tmo_state)}).sort_index()
        df_for_lags.loc[ts] = np.nan
        df_for_lags = add_lags_mas(df_for_lags, "tmo_s").loc[[ts]]

        X_row = pd.concat([df_row, df_for_lags], axis=1)
        X = dummies_and_reindex(X_row, cols_tmo)
        if X.isna().any().any():
            X = X.ffill(axis=1).infer_objects(copy=False).fillna(0)

        y_hat = float(m_tmo.predict(sc_tmo.transform(X.values), verbose=0).ravel()[0])
        preds.append((ts, y_hat))
        tmo_state[ts] = y_hat  # feed-forward

    return pd.Series(dict(preds)).sort_index()

# =============================================================================
# INFERENCIA PRINCIPAL
# =============================================================================

def forecast_120d(
    historico_llamadas: pd.DataFrame,        # ts + (calls/recibidos_*/llamadas) + (feriados opcional)
    historico_tmo: pd.DataFrame,             # ts + (tmo_s o tmo_general/aht)
    feriados_df: pd.DataFrame | None = None, # alternativa a holidays_set: ts + feriados (0/1)
    models_dir: str | Path = "models",
    horizonte_dias: int = 120,
    # --- alias de compatibilidad usados por src/main.py ---
    horizon_days: int | None = None,
    holidays_set: Iterable[Any] | None = None,
    exportar_json: bool = True,
) -> pd.DataFrame:
    if horizon_days is not None:
        try:
            horizonte_dias = int(horizon_days)
        except Exception:
            pass

    # Normalizar históricos
    df_calls_hist = ensure_ts(historico_llamadas)
    df_calls_hist = _normalize_calls_and_holidays(df_calls_hist)

    df_tmo_hist = ensure_ts(historico_tmo)
    df_tmo_hist = _normalize_tmo(df_tmo_hist)

    # Horizonte: usa SIEMPRE última hora de llamadas (evita que cambie por gap en TMO)
    last_ts_calls = df_calls_hist.index.max()
    last_ts_tmo   = df_tmo_hist.index.max()
    idx_h = _build_horizon_index(last_ts_calls, horas=horizonte_dias * 24)

    # Para TMO AR: si el TMO histórico quedó atrás, extiendo su estado con forward-fill
    if last_ts_tmo < last_ts_calls:
        tail_val = float(df_tmo_hist.loc[:last_ts_tmo, "tmo_s"].iloc[-1])
        extender = pd.date_range(start=last_ts_tmo + pd.Timedelta(hours=1), end=last_ts_calls, freq="h", tz=TZ)
        if len(extender) > 0:
            df_ext = pd.DataFrame({"tmo_s": tail_val}, index=extender)
            df_tmo_hist = pd.concat([df_tmo_hist, df_ext]).sort_index()

    # Serie de feriados por hora
    if holidays_set is not None:
        fer_h = _calendar_series_from_holidays_set(idx_h, holidays_set)
    else:
        fer_h = _calendar_series_for_horizon(idx_h, feriados_df)

    # LLAMADAS — planner feed-forward
    m_pl, sc_pl, cols_pl = _load_planner(models_dir)
    calls_hist_series = df_calls_hist["calls"].astype(float).sort_index()

    calls_pred_raw = predict_calls_horizonte(
        idx_horizonte=idx_h,
        calls_hist_series=calls_hist_series,
        feriados_series=fer_h,
        models_dir=models_dir,
    )

    # Ajuste de feriados (OPCIONAL). Por defecto OFF para no cambiar tus valores clásicos.
    if PLANNER_HOLIDAY_ADJ == 1:
        factors_h = compute_holiday_factors(
            calls_hist=calls_hist_series,
            feriados_hist=df_calls_hist["feriados"].astype(int),
        )
        calls_pred = apply_holiday_adjustment(calls_pred_raw, fer_h, factors_h)
    else:
        calls_pred = calls_pred_raw

    # TMO — feed-forward (autoregresivo)
    m_tmo, sc_tmo, cols_tmo = _load_tmo(models_dir)
    df_tmo_sorted = df_tmo_hist.sort_index()

    tmo_pred = _predict_tmo_feedforward(
        idx_h=idx_h,
        df_tmo_hist=df_tmo_sorted,
        df_calls_pred=calls_pred,
        holidays_set=set(holidays_set) if holidays_set is not None else None,
        m_tmo=m_tmo,
        sc_tmo=sc_tmo,
        cols_tmo=cols_tmo,
    )

    # Regularización + límites
    tmo_pred = _mad_cap(tmo_pred, k=TMO_MAD_K)
    tmo_pred = _clamp_tmo_bounds(tmo_pred, min_s=TMO_MIN_S, max_s=TMO_MAX_S)

    # Anclaje por hora del día (blend a mediana histórica por hora)
    try:
        hist_by_hour = df_tmo_hist["tmo_s"].groupby(df_tmo_hist.index.hour).median()
        anchor = tmo_pred.index.map(lambda ts: hist_by_hour.get(ts.hour, np.nan))
        anchor = pd.Series(anchor.values, index=tmo_pred.index).fillna(tmo_pred.median())
        if 0.0 <= TMO_BLEND_ALPHA <= 1.0 and TMO_BLEND_ALPHA < 1.0:
            tmo_pred = TMO_BLEND_ALPHA * tmo_pred + (1.0 - TMO_BLEND_ALPHA) * anchor
    except Exception:
        # si no pudimos calcular ancla, seguimos con pred tal cual
        pass

    # Cap por cuantil horario histórico (p. ej. 95%)
    if 0.0 < TMO_HOURLY_QHI < 1.0:
        try:
            qhi_by_hour = df_tmo_hist["tmo_s"].groupby(df_tmo_hist.index.hour).quantile(TMO_HOURLY_QHI)
            cap_series = tmo_pred.index.map(lambda ts: qhi_by_hour.get(ts.hour, np.nan))
            cap_series = pd.Series(cap_series.values, index=tmo_pred.index).fillna(df_tmo_hist["tmo_s"].quantile(TMO_HOURLY_QHI))
            tmo_pred = np.minimum(tmo_pred.values, cap_series.values)
            tmo_pred = pd.Series(tmo_pred, index=cap_series.index)
        except Exception:
            pass

    # Calibración opcional
    cal_factor = _load_calibration_factor(default=None)
    tmo_pred = _apply_calibration(tmo_pred, factor=cal_factor, lo=0.7, hi=1.3)

    # Ensamblar DF final
    df_pred = pd.DataFrame(index=idx_h)
    df_pred["calls"] = calls_pred
    df_pred["tmo_s"] = tmo_pred

    # Erlang C (agentes) — robusto a salidas vector/array
    agents_prod, agents_sched = [], []
    for ts, row in df_pred.iterrows():
        a_raw = required_agents(float(row["calls"]), float(row["tmo_s"]))
        a_scalar = float(np.asarray(a_raw).reshape(-1)[0])
        a_prod = int(np.ceil(a_scalar))

        sched_raw = schedule_agents(a_scalar)
        sched_scalar = float(np.asarray(sched_raw).reshape(-1)[0])
        a_sched = int(np.ceil(sched_scalar))

        agents_prod.append(a_prod)
        agents_sched.append(a_sched)

    df_pred["agents_prod"] = agents_prod
    df_pred["agents_sched"] = agents_sched

    # Export JSONs (pretty)
    if exportar_json:
        out_h = [
            {
                "ts": ts.isoformat(),
                "llamadas_hora": float(df_pred.at[ts, "calls"]),
                "tmo_hora": float(df_pred.at[ts, "tmo_s"]),
                "agents_prod": int(df_pred.at[ts, "agents_prod"]),
                "agentes_requeridos": int(df_pred.at[ts, "agents_sched"]),
            }
            for ts in df_pred.index
        ]
        _write_json(PUBLIC_DIR / "prediccion_horaria.json", out_h)

        df_daily = pd.DataFrame({
            "fecha": df_pred.index.tz_convert(TZ).normalize(),
            "calls": df_pred["calls"].values,
            "tmo_s": df_pred["tmo_s"].values,
        }).groupby("fecha").agg(
            llamadas_diarias=("calls", "sum"),
            tmo_diario=("tmo_s", "mean"),
        ).reset_index()

        out_d = [
            {
                "fecha": pd.Timestamp(r["fecha"]).date().isoformat(),
                "llamadas_diarias": float(r["llamadas_diarias"]),
                "tmo_diario": float(r["tmo_diario"]),
            }
            for _, r in df_daily.iterrows()
        ]
        _write_json(PUBLIC_DIR / "prediccion_diaria.json", out_d)

    return df_pred

# =============================================================================
# Fin de archivo
# =============================================================================

