# src/inferencia/inferencia_core.py
# =============================================================================
# Núcleo de inferencia — Planner (llamadas) + TMO autoregresivo + Erlang + Export
# Acepta alias de compatibilidad: horizon_days y holidays_set.
# =============================================================================

from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Sequence, Tuple, Iterable, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import load as joblib_load

# --- Import de módulos locales ---
from src.inferencia.tmo_ar_inference import predict_tmo_from_models_dir
from src.inferencia.erlang import required_agents, schedule_agents

# =============================================================================
# Utilidades básicas
# =============================================================================

TZ = "America/Santiago"
PUBLIC_DIR = Path("public")
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

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

def _ensure_ts_index(df: pd.DataFrame, tz: str = TZ) -> pd.DataFrame:
    """Asegura índice horario 'ts' con tz y ordenado."""
    if "ts" not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Se requiere columna 'ts' o un DatetimeIndex.")
    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], errors="coerce", utc=False)
        df = df.loc[ts.notna()].copy()
        df["ts"] = ts.loc[ts.notna()]
        df = df.sort_values("ts").set_index("ts")
    else:
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce", utc=False)
        df = df.loc[df.index.notna()].sort_index()
    # Local-time aware (no UTC), respetando cambios de hora CL
    df.index = df.index.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT", errors="coerce") \
                         .tz_convert(tz)
    df = df.loc[df.index.notna()]
    return df

def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

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

# =============================================================================
# Planner (llamadas) — inferencia feed-forward con artefactos del training
# =============================================================================

def _load_planner(models_dir: str | Path = "models"):
    models_dir = Path(models_dir)
    planner_model = tf.keras.models.load_model(models_dir / "modelo_planner.keras")
    scaler = joblib_load(models_dir / "scaler_planner.pkl")
    with open(models_dir / "training_columns_planner.json", "r", encoding="utf-8") as f:
        training_cols = json.load(f)
    return planner_model, scaler, training_cols

def _planner_row(ts: pd.Timestamp, calls_hist: dict[pd.Timestamp, float], feriado: int) -> dict:
    """Replica features del entrenamiento del planner:
       - lags: 24, 48, 72, 168
       - MAs: 24, 72, 168
       - calendario/dummies + sin/cos
       - feriados + es_dia_de_pago
    """
    def lag(h: int) -> float:
        key = ts - pd.Timedelta(hours=h)
        return float(calls_hist.get(key, np.nan))

    def ma(win_h: int) -> float:
        end = ts - pd.Timedelta(hours=1)
        vals = []
        for i in range(win_h):
            key = end - pd.Timedelta(hours=i)
            v = calls_hist.get(key, np.nan)
            if pd.notna(v):
                vals.append(float(v))
        return float(np.mean(vals)) if vals else np.nan

    tp = _time_parts(ts)
    row = {
        "lag_24": lag(24), "lag_48": lag(48), "lag_72": lag(72), "lag_168": lag(168),
        "ma_24": ma(24), "ma_72": ma(72), "ma_168": ma(168),
        "sin_hour": tp["sin_hour"], "cos_hour": tp["cos_hour"],
        "sin_dow": tp["sin_dow"], "cos_dow": tp["cos_dow"],
        "feriados": int(feriado),
        "es_dia_de_pago": tp["es_dia_de_pago"],
        "dow": tp["dow"], "month": tp["month"], "hour": tp["hour"],
    }
    return row

def predict_calls_horizonte(
    idx_horizonte: pd.DatetimeIndex,
    calls_hist_series: pd.Series,     # histórico real de llamadas por hora (hasta t0-1h)
    feriados_series: pd.Series,       # 0/1 por hora para el horizonte
    models_dir: str | Path = "models",
    cap_k: float = 3.5,
) -> pd.Series:
    """Predice llamadas por hora feed-forward usando el Planner entrenado."""
    model, scaler, train_cols = _load_planner(models_dir)
    calls_hist = calls_hist_series.to_dict()
    preds = []
    for ts in idx_horizonte:
        fer = int(feriados_series.get(ts, 0))
        row = _planner_row(ts, calls_hist, fer)
        X = pd.DataFrame([row])
        X = pd.get_dummies(X, columns=["dow", "month", "hour"], drop_first=False)
        X = X.reindex(columns=list(train_cols), fill_value=0)
        if X.isna().any().any():
            X = X.fillna(method="ffill", axis=1).fillna(0)
        Xs = scaler.transform(X.values)
        y_hat = float(model.predict(Xs, verbose=0).ravel()[0])
        preds.append((ts, y_hat))
        calls_hist[ts] = y_hat  # feed-forward
    s = pd.Series(dict(preds)).sort_index()
    s = _mad_cap(s, k=cap_k)
    s = _clamp(s, low=0, high=float(s.quantile(0.999)) if len(s) > 10 else s.max())
    return s

# =============================================================================
# Feriados
# =============================================================================

def compute_holiday_factors(
    calls_hist: pd.Series,
    feriados_hist: pd.Series,
) -> dict[int, float]:
    """Factor por hora (0..23): mediana(feriado)/mediana(no feriado)."""
    df = pd.DataFrame({
        "calls": calls_hist,
        "fer": feriados_hist.reindex(calls_hist.index).fillna(0).astype(int)
    }).dropna()
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

def apply_holiday_adjustment(
    calls_series: pd.Series, feriados_series: pd.Series, hour_factors: dict[int, float]
) -> pd.Series:
    """Multiplica llamadas por factor de la hora si es feriado (TMO NO se ajusta)."""
    adj = []
    for ts, v in calls_series.items():
        if int(feriados_series.get(ts, 0)) == 1:
            f = hour_factors.get(ts.hour, 1.0)
            adj.append(v * f)
        else:
            adj.append(v)
    return pd.Series(adj, index=calls_series.index)

def _build_horizon_index(last_ts: pd.Timestamp, horas: int) -> pd.DatetimeIndex:
    start = last_ts + pd.Timedelta(hours=1)
    idx = pd.date_range(start=start, periods=horas, freq="H", tz=TZ)
    return idx

def _calendar_series_for_horizon(idx: pd.DatetimeIndex, feriados_df: pd.DataFrame | None) -> pd.Series:
    """Genera serie 0/1 de feriados para el horizonte desde un DataFrame; si no, 0."""
    if feriados_df is None or "feriados" not in feriados_df.columns:
        return pd.Series(0, index=idx, dtype=int)
    fer = _ensure_ts_index(feriados_df)[["feriados"]].astype(int)
    fer = fer.reindex(idx, method="nearest", tolerance=pd.Timedelta("1H")).fillna(0).astype(int)["feriados"]
    fer.index = idx
    return fer

def _calendar_series_from_holidays_set(idx: pd.DatetimeIndex, holidays_set: Iterable[Any]) -> pd.Series:
    """
    Construye 0/1 por hora a partir de un conjunto/lista de feriados (fechas).
    Acepta: date/datetime/pandas Timestamp/str (YYYY-MM-DD).
    """
    # Normalizar a set de fechas (date)
    dates = set()
    for d in holidays_set:
        if isinstance(d, pd.Timestamp):
            dates.add(d.date())
        else:
            try:
                dates.add(pd.Timestamp(d).date())
            except Exception:
                # ignora elementos inválidos silenciosamente
                pass
    if not dates:
        return pd.Series(0, index=idx, dtype=int)
    ser = pd.Series([1 if ts.date() in dates else 0 for ts in idx], index=idx, dtype=int)
    return ser

# =============================================================================
# INFERENCIA PRINCIPAL
# =============================================================================

def forecast_120d(
    historico_llamadas: pd.DataFrame,     # columnas: ts, recibidos (o calls), feriados (0/1)
    historico_tmo: pd.DataFrame,          # columnas: ts, tmo_s (segundos)
    feriados_df: pd.DataFrame | None = None,     # opcional: serie/DF por hora o diario
    models_dir: str | Path = "models",
    horizonte_dias: int = 120,
    # --- ALIAS DE COMPATIBILIDAD ---
    horizon_days: int | None = None,             # alias de horizonte_dias
    holidays_set: Iterable[Any] | None = None,   # alias alternativo a feriados_df (lista/set de fechas)
    exportar_json: bool = True,
) -> pd.DataFrame:
    """
    Devuelve un DataFrame horario con: calls, tmo_s, agents_prod, agents_sched.
    Exporta los JSON si exportar_json=True.
    """
    # Compatibilidad: si llega horizon_days, tiene prioridad.
    if horizon_days is not None:
        try:
            horizonte_dias = int(horizon_days)
        except Exception:
            pass

    # 0) Normalizar insumos
    df_calls_hist = historico_llamadas.copy()
    # Normalizar nombre de columna calls
    if "recibidos" in df_calls_hist.columns and "calls" not in df_calls_hist.columns:
        df_calls_hist = df_calls_hist.rename(columns={"recibidos": "calls"})
    if "feriados" not in df_calls_hist.columns:
        df_calls_hist["feriados"] = 0

    df_calls_hist = _ensure_ts_index(df_calls_hist)
    df_tmo_hist   = _ensure_ts_index(historico_tmo)
    if "tmo_s" not in df_tmo_hist.columns:
        tmo_col = next((c for c in df_tmo_hist.columns if "tmo" in c.lower()), None)
        if not tmo_col:
            raise ValueError("No se encontró columna de TMO en historico_tmo.")
        df_tmo_hist = df_tmo_hist.rename(columns={tmo_col: "tmo_s"})

    last_ts = min(df_calls_hist.index.max(), df_tmo_hist.index.max())
    horizonte_horas = horizonte_dias * 24
    idx_horizonte = _build_horizon_index(last_ts, horas=horizonte_horas)

    # 1) Calendario de feriados para el horizonte
    if holidays_set is not None:
        fer_h = _calendar_series_from_holidays_set(idx_horizonte, holidays_set)
    else:
        fer_h = _calendar_series_for_horizon(idx_horizonte, feriados_df)

    # 2) Predicción de LLAMADAS (feed-forward con Planner)
    calls_hist_series = df_calls_hist["calls"].astype(float).sort_index()

    # Factores por hora a partir del histórico real (para ajustar SOLO llamadas en feriados)
    hour_factors = compute_holiday_factors(
        calls_hist=calls_hist_series,
        feriados_hist=df_calls_hist["feriados"].astype(int)
    )

    calls_pred_raw = predict_calls_horizonte(
        idx_horizonte=idx_horizonte,
        calls_hist_series=calls_hist_series,
        feriados_series=fer_h,
        models_dir=models_dir,
        cap_k=3.5
    )
    # Ajuste de feriados SOLO a llamadas
    calls_pred = apply_holiday_adjustment(calls_pred_raw, fer_h, hour_factors)

    # 3) Predicción de TMO AUTORREGRESIVO (feed-forward)
    tmo_hist_series = df_tmo_hist["tmo_s"].astype(float).sort_index()
    tmo_pred = predict_tmo_from_models_dir(
        idx_horizonte=idx_horizonte,
        serie_tmo_hist=tmo_hist_series,
        calls_series=calls_pred,       # ya ajustadas por feriado
        feriados_series=fer_h,
        models_dir=models_dir,
        fill_prop=(0.55, 0.45),        # si puedes, reemplaza por proporciones reales recientes
        cap_k=3.5,
        clamp_bounds=(80, 900),
        strict_columns_check=False
    )

    # 4) Ensamblar DF horario final (previo a Erlang)
    df_pred = pd.DataFrame(index=idx_horizonte)
    df_pred["calls"] = calls_pred
    df_pred["tmo_s"] = tmo_pred

    # 5) Erlang C: agentes productivos y programados
    agents_prod = []
    agents_sched = []
    for ts, row in df_pred.iterrows():
        a = required_agents(calls=float(row["calls"]), aht_s=float(row["tmo_s"]))
        agents_prod.append(int(np.ceil(a)))
        agents_sched.append(int(np.ceil(schedule_agents(a))))

    df_pred["agents_prod"] = agents_prod
    df_pred["agents_sched"] = agents_sched

    # 6) Export (horario y diario)
    if exportar_json:
        # Horario
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

        # Diario (sum calls, mean tmo)
        df_daily = pd.DataFrame({
            "fecha": df_pred.index.tz_convert(TZ).normalize(),
            "calls": df_pred["calls"].values,
            "tmo_s": df_pred["tmo_s"].values
        }).groupby("fecha").agg(
            llamadas_diarias=("calls", "sum"),
            tmo_diario=("tmo_s", "mean")
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
# Ejemplo de uso (comentado)
# =============================================================================
# if __name__ == "__main__":
#     df_calls = pd.read_csv("data/historical_data.csv")   # ts, recibidos, feriados
#     df_tmo   = pd.read_csv("data/HISTORICO_TMO.csv")     # ts, tmo_s
#     df_fer   = pd.read_csv("data/Feriados_Chilev2.csv")  # ts, feriados
#     res = forecast_120d(df_calls, df_tmo, df_fer,
#                         models_dir="models", horizon_days=120, exportar_json=True)
#     print(res.head())

