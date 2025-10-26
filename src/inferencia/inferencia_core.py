# src/inferencia/inferencia_core.py
# =============================================================================
# Núcleo de inferencia — Planner (llamadas) + TMO autoregresivo + Erlang + Export
# Compat: horizon_days (alias de horizonte_dias), holidays_set (alias feriados).
# TMO feed-forward (autoregresivo) alineado a entrenamiento + regularización/calibración.
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
    add_time_parts,
    add_lags_mas,
    dummies_and_reindex,
)
from src.inferencia.erlang import required_agents, schedule_agents

# =============================================================================
# Constantes
# =============================================================================
TZ = "America/Santiago"
PUBLIC_DIR = Path("public")
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

# Nombres esperados en artefactos (dentro de models/)
PLANNER_MODEL_NAME = "modelo_planner.keras"
PLANNER_SCALER_NAME = "scaler_planner.pkl"
PLANNER_COLS_NAME = "training_columns_planner.json"

TMO_MODEL_NAME = "modelo_tmo.keras"
TMO_SCALER_NAME = "scaler_tmo.pkl"
TMO_COLS_NAME = "training_columns_tmo.json"

# Columnas objetivo típicas en tus datasets
TARGET_CALLS_CANDIDATES = [
    "calls",
    "recibidos_nacional",
    "recibidos",
    "llamadas",
    "q_llamadas_general",
]
TARGET_TMO_CANDIDATES = [
    "tmo_general",
    "tmo",
    "aht",
]
FERIADOS_CANDIDATES = [
    "feriados",
    "feriado",
    "is_holiday",
    "holiday",
    "es_feriado",
]

# =============================================================================
# Helpers generales
# =============================================================================

def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def _safe_ratio(a, b):
    return np.array(a, dtype=float) / (np.array(b, dtype=float) + 1e-6)

def _normalize_calls_and_holidays(df_calls: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura columnas 'calls' (float) y 'feriados' (int 0/1).
    No altera 'ts' (lo maneja ensure_ts).
    """
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
    """
    Asegura columna 'tmo_s' (segundos, float).
    Si no existe, intenta detectar un nombre alternativo y lo renombra.
    """
    df = df_tmo.copy()
    cols_lower = {c.lower(): c for c in df.columns}
    src_tmo = next((cols_lower[c] for c in TARGET_TMO_CANDIDATES if c in cols_lower), None)
    if src_tmo is None and "tmo_s" not in df.columns:
        # Si ya viene tmo_s, respetar
        raise KeyError(f"No se encontró columna de TMO. Candidatas: {TARGET_TMO_CANDIDATES}. Columns={list(df.columns)}")
    if src_tmo and src_tmo != "tmo_s":
        df = df.rename(columns={src_tmo: "tmo_s"})
    df["tmo_s"] = pd.to_numeric(df["tmo_s"], errors="coerce")
    return df

def _build_horizon_index(last_ts: pd.Timestamp, horas: int) -> pd.DatetimeIndex:
    # Usar 'h' para evitar FutureWarning
    start = last_ts + pd.Timedelta(hours=1)
    return pd.date_range(start=start, periods=horas, freq="h", tz=TZ)

def _calendar_series_from_holidays_set(idx: pd.DatetimeIndex, holidays_set: Iterable[Any]) -> pd.Series:
    """
    Convierte un conjunto/lista de fechas (YYYY-MM-DD, date, Timestamp) a 0/1 por hora.
    """
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
    """
    Genera serie 0/1 de feriados para el horizonte desde un DataFrame con 'ts' y 'feriados'.
    Si no hay DF, devuelve todo 0.
    """
    if feriados_df is None or "feriados" not in feriados_df.columns:
        return pd.Series(0, index=idx, dtype=int)
    fer = ensure_ts(feriados_df)[["feriados"]].astype(int)
    # Rellena por cercanía horaria
    fer = fer.reindex(idx, method="nearest", tolerance=pd.Timedelta("1H")).fillna(0).astype(int)["feriados"]
    fer.index = idx
    return fer

def compute_holiday_factors(calls_hist: pd.Series, feriados_hist: pd.Series) -> dict[int, float]:
    """
    Factor por hora (0..23): mediana(feriado)/mediana(no feriado).
    Se aplica SOLO a llamadas en el horizonte.
    """
    df = pd.DataFrame(
        {"calls": calls_hist, "fer": feriados_hist.reindex(calls_hist.index).fillna(0).astype(int)}
    ).dropna()
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
    """
    Multiplica llamadas por factor de la hora si es feriado (TMO NO se ajusta).
    """
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

    tp = {
        "dow": ts.dayofweek,
        "month": ts.month,
        "hour": ts.hour,
        "day": ts.day,
        "sin_hour": float(np.sin(2 * np.pi * ts.hour / 24)),
        "cos_hour": float(np.cos(2 * np.pi * ts.hour / 24)),
        "sin_dow": float(np.sin(2 * np.pi * ts.dayofweek / 7)),
        "cos_dow": float(np.cos(2 * np.pi * ts.dayofweek / 7)),
        "es_dia_de_pago": int(ts.day in {1, 2, 15, 16, 29, 30, 31}),
    }

    return {
        "lag_24": lag(24), "lag_48": lag(48), "lag_72": lag(72), "lag_168": lag(168),
        "ma_24": ma(24), "ma_72": ma(72), "ma_168": ma(168),
        "sin_hour": tp["sin_hour"], "cos_hour": tp["cos_hour"],
        "sin_dow": tp["sin_dow"], "cos_dow": tp["cos_dow"],
        "feriados": int(feriado), "es_dia_de_pago": tp["es_dia_de_pago"],
        "dow": tp["dow"], "month": tp["month"], "hour": tp["hour"],
    }

def predict_calls_horizonte(
    idx_horizonte: pd.DatetimeIndex,
    calls_hist_series: pd.Series,
    feriados_series: pd.Series,
    models_dir: str | Path = "models",
    cap_k: float = 3.5,
) -> pd.Series:
    model, scaler, train_cols = _load_planner(models_dir)
    calls_hist = calls_hist_series.to_dict()
    preds = []
    for ts in idx_horizonte:
        row = _planner_row(ts, calls_hist, int(feriados_series.get(ts, 0)))
        X = dummies_and_reindex(pd.DataFrame([row], index=[ts]), train_cols)
        if X.isna().any().any():
            X = X.fillna(method="ffill", axis=1).fillna(0)
        y_hat = float(model.predict(scaler.transform(X.values), verbose=0).ravel()[0])
        preds.append((ts, y_hat))
        calls_hist[ts] = y_hat  # feed-forward
    s = pd.Series(dict(preds)).sort_index()
    # Guardrail suave; el planner suele ser estable, pero aplicamos cap por si acaso
    med = s.median()
    s = s.clip(lower=0, upper=max(med * 5, s.quantile(0.995)))
    return s

# =============================================================================
# TMO — inferencia feed-forward (autoregresivo)
# =============================================================================

def _predict_tmo_feedforward(
    idx_h: pd.DatetimeIndex,
    df_tmo_hist: pd.DataFrame,     # Debe tener 'tmo_s'; opcionalmente q_llamadas_* para proporciones
    df_calls_pred: pd.Series,      # llamadas predichas (ya ajustadas por feriado)
    holidays_set: set | None,
    m_tmo, sc_tmo, cols_tmo: list[str],
) -> pd.Series:
    """
    Predice TMO por hora autoregresivo: lags/mas del TMO se construyen con historia + predicciones.
    Features replican entrenamiento: cíclicas, calendario, feriados, exógena calls, proporciones internas.
    """
    # Proporciones: si el dataset trae cantidades por tipo, calculamos proporciones; si no, fallback
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

        # Construir lags/MAs sobre el estado de TMO
        df_for_lags = pd.DataFrame({"tmo_s": pd.Series(tmo_state)}).sort_index()
        df_for_lags.loc[ts] = np.nan
        df_for_lags = add_lags_mas(df_for_lags, "tmo_s").loc[[ts]]

        # Ensamblar fila final
        X_row = pd.concat([df_row, df_for_lags], axis=1)
        X = dummies_and_reindex(X_row, cols_tmo)
        if X.isna().any().any():
            X = X.fillna(method="ffill", axis=1).fillna(0)

        y_hat = float(m_tmo.predict(sc_tmo.transform(X.values), verbose=0).ravel()[0])
        preds.append((ts, y_hat))
        tmo_state[ts] = y_hat  # feed-forward

    return pd.Series(dict(preds)).sort_index()

# =============================================================================
# INFERENCIA PRINCIPAL
# =============================================================================

def forecast_120d(
    historico_llamadas: pd.DataFrame,        # Debe contener ts + (calls/recibidos_*/llamadas) + (feriados opcional)
    historico_tmo: pd.DataFrame,             # Debe contener ts + (tmo_s o tmo_general/aht)
    feriados_df: pd.DataFrame | None = None, # Alternativa a holidays_set: ts + feriados (0/1)
    models_dir: str | Path = "models",
    horizonte_dias: int = 120,
    # --- alias de compatibilidad usados por src/main.py ---
    horizon_days: int | None = None,
    holidays_set: Iterable[Any] | None = None,
    exportar_json: bool = True,
) -> pd.DataFrame:
    """
    Devuelve DataFrame horario con: calls, tmo_s, agents_prod, agents_sched.
    También escribe public/prediccion_horaria.json y prediccion_diaria.json si exportar_json=True.
    """
    # Compat: si llega horizon_days, tiene prioridad
    if horizon_days is not None:
        try:
            horizonte_dias = int(horizon_days)
        except Exception:
            pass

    # 0) Normalizar históricos (ts + columnas)
    df_calls_hist = ensure_ts(historico_llamadas)
    df_calls_hist = _normalize_calls_and_holidays(df_calls_hist)

    df_tmo_hist = ensure_ts(historico_tmo)
    df_tmo_hist = _normalize_tmo(df_tmo_hist)

    # 1) Horizonte
    last_ts = min(df_calls_hist.index.max(), df_tmo_hist.index.max())
    idx_h = _build_horizon_index(last_ts, horas=horizonte_dias * 24)

    # 2) Serie de feriados por hora
    if holidays_set is not None:
        fer_h = _calendar_series_from_holidays_set(idx_h, holidays_set)
    else:
        fer_h = _calendar_series_for_horizon(idx_h, feriados_df)

    # 3) LLAMADAS — planner feed-forward + ajuste de feriados SOLO a calls
    m_pl, sc_pl, cols_pl = _load_planner(models_dir)
    calls_hist_series = df_calls_hist["calls"].astype(float).sort_index()

    # factores de feriado por hora a partir del histórico real
    factors_h = compute_holiday_factors(calls_hist=calls_hist_series, feriados_hist=df_calls_hist["feriados"].astype(int))

    calls_pred_raw = predict_calls_horizonte(
        idx_horizonte=idx_h,
        calls_hist_series=calls_hist_series,
        feriados_series=fer_h,
        models_dir=models_dir,
        cap_k=3.5,
    )
    calls_pred = apply_holiday_adjustment(calls_pred_raw, fer_h, factors_h)

    # 4) TMO — feed-forward (autoregresivo) alineado a entrenamiento
    m_tmo, sc_tmo, cols_tmo = _load_tmo(models_dir)
    # asegurar columnas auxiliares (si existen) para proporciones internas
    # loader_tmo ya suele traer q_llamadas_*; si no existen, usamos fallback 0.55/0.45 dentro de la función
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

    # 5) Regularización y calibración (seguro + opcional)
    mad_k  = float(os.getenv("TMO_MAD_K", "3.5"))
    tmo_lo = float(os.getenv("TMO_MIN_S", "80"))
    tmo_hi = float(os.getenv("TMO_MAX_S", "900"))

    tmo_pred = _mad_cap(tmo_pred, k=mad_k)
    tmo_pred = _clamp_tmo_bounds(tmo_pred, min_s=tmo_lo, max_s=tmo_hi)

    cal_factor = _load_calibration_factor(default=None)  # None => sin calibración
    tmo_pred = _apply_calibration(tmo_pred, factor=cal_factor, lo=0.7, hi=1.3)

    # 6) Ensamblar DF final
    df_pred = pd.DataFrame(index=idx_h)
    df_pred["calls"] = calls_pred
    df_pred["tmo_s"] = tmo_pred

    # 7) Erlang C (agentes)
    agents_prod, agents_sched = [], []
    for ts, row in df_pred.iterrows():
        a = required_agents(calls=float(row["calls"]), aht_s=float(row["tmo_s"]))
        agents_prod.append(int(np.ceil(a)))
        agents_sched.append(int(np.ceil(schedule_agents(a))))
    df_pred["agents_prod"] = agents_prod
    df_pred["agents_sched"] = agents_sched

    # 8) Export JSONs
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


