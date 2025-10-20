# src/inferencia/inferencia_core.py
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from typing import Optional, Set, Dict, Tuple

from .features import ensure_ts, add_time_parts, add_lags_mas, dummies_and_reindex
from .erlang import required_agents, schedule_agents
from .utils_io import write_daily_json, write_hourly_json

# ==============================
# Configuración general
# ==============================
TIMEZONE = "America/Santiago"
PUBLIC_DIR = "public"

PLANNER_MODEL = "models/modelo_planner.keras"
PLANNER_SCALER = "models/scaler_planner.pkl"
PLANNER_COLS  = "models/training_columns_planner.json"

TMO_MODEL = "models/modelo_tmo.keras"
TMO_SCALER = "models/scaler_tmo.pkl"
TMO_COLS   = "models/training_columns_tmo.json"

# Nombre canónico que usa el planner de llamadas
TARGET_CALLS = "q_llamadas_general"
TARGET_TMO   = "tmo_general"

# Aliases aceptados para la columna de llamadas en el histórico
CALLS_ALIASES = [
    "q_llamadas_general",
    "recibidos_nacional",
    "llamadas",
    "llamadas_total",
    "q_general",
    "calls"
]

# ====== Config de POST-FERIADOS (afecta llamadas y TMO) ======
POST_HOLIDAY_HOURS = 48  # ventana post-feriado para “rebote”

# ====== Config de ventanas y límites ======
HIST_WINDOW_DAYS = 90

# ======= Guardrail Outliers (config) =======
ENABLE_OUTLIER_CAP = True   # <- ponlo en False si quieres desactivarlo
K_WEEKDAY = 6.0             # techos +K*MAD en lun-vie
K_WEEKEND = 7.0             # techos +K*MAD en sáb-dom


# ==============================
# Utilidades internas
# ==============================
def _load_cols(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _safe_ratio(num, den, fallback=np.nan):
    try:
        v = float(num) / float(den)
        if not np.isfinite(v):
            return fallback
        return v
    except Exception:
        return fallback


def _is_holiday(ts, holidays_set: Optional[Set]):
    if holidays_set is None:
        return 0
    try:
        d = ts.tz_convert(TIMEZONE).date()
    except Exception:
        d = ts.date()
    return 1 if d in holidays_set else 0


def _is_post_holiday(ts, holidays_set: Optional[Set]):
    if holidays_set is None:
        return 0
    try:
        d = ts.tz_convert(TIMEZONE).date()
    except Exception:
        d = ts.date()
    return 1 if (d - pd.Timedelta(days=1)) in holidays_set or (d - pd.Timedelta(days=2)) in holidays_set else 0


def _weekday_mad_caps(s: pd.Series, k: float):
    med = np.nanmedian(s)
    mad = np.nanmedian(np.abs(s - med)) + 1e-9
    return med + k * 1.4826 * mad


def _build_outlier_caps(df_recent: pd.DataFrame, col: str) -> Dict[Tuple[int, int], float]:
    """
    Crea techos horario*dow para 'col' con K distintos en weekday/weekend.
    Si no hay datos, devuelve infinito (sin recorte).
    """
    if df_recent is None or df_recent.empty or col not in df_recent.columns:
        return {(dow, h): float("inf") for dow in range(7) for h in range(24)}

    tmp = df_recent.copy()
    tmp["dow"] = tmp.index.dayofweek
    tmp["hour"] = tmp.index.hour

    caps = {}
    for dow in range(7):
        is_weekend = (dow >= 5)
        k = K_WEEKEND if is_weekend else K_WEEKDAY
        for h in range(24):
            s = pd.to_numeric(tmp.loc[(tmp["dow"] == dow) & (tmp["hour"] == h), col], errors="coerce")
            caps[(dow, h)] = float(_weekday_mad_caps(s, k))
    return caps


def _apply_post_holiday_adjustment(series: pd.Series, holidays_set: Optional[Set], hours_window: int = POST_HOLIDAY_HOURS, factor_calls=1.08, factor_tmo=1.03):
    if holidays_set is None or series.empty:
        return series

    idx = series.index
    is_post = []
    for ts in idx:
        is_post.append(1 if _is_post_holiday(ts, holidays_set) else 0)
    is_post = np.array(is_post, dtype=float)

    kernel = np.ones(min(hours_window, len(is_post)), dtype=float)
    post_signal = np.convolve(is_post, kernel, mode="same")
    post_signal = post_signal / (post_signal.max() + 1e-9)

    if series.name == TARGET_CALLS:
        f = factor_calls
    else:
        f = factor_tmo

    adjusted = series.values * (1.0 + (f - 1.0) * post_signal)
    return pd.Series(adjusted, index=series.index, name=series.name)


def _holiday_hour_factors(df_recent: pd.DataFrame, col: str, holidays_set):
    """
    Calcula factores (feriado vs normal) por hora para ajustar el forecast.
    Si la columna 'col' no existe o el DF está vacío, devuelve factores neutros (1.0).
    """
    if df_recent is None or df_recent.empty or col not in df_recent.columns:
        return {int(h): 1.0 for h in range(24)}

    df_recent = df_recent.copy()
    if "feriados" not in df_recent.columns:
        if holidays_set is None:
            return {int(h): 1.0 for h in range(24)}
        df_recent["feriados"] = [_is_holiday(ts, holidays_set) for ts in df_recent.index]

    df_recent["hour"] = df_recent.index.hour
    df_recent[col] = pd.to_numeric(df_recent[col], errors="coerce")

    med_nor = df_recent.loc[df_recent["feriados"] == 0].groupby("hour")[col].median().to_dict()
    med_hol = df_recent.loc[df_recent["feriados"] == 1].groupby("hour")[col].median().to_dict()

    if not med_nor:
        return {int(h): 1.0 for h in range(24)}

    global_factor = _safe_ratio(
        np.nanmedian(list(med_hol.values())) if med_hol else np.nan,
        np.nanmedian(list(med_nor.values())),
        fallback=1.0
    )
    if not np.isfinite(global_factor):
        global_factor = 1.0

    return {
        int(h): _safe_ratio(med_hol.get(h, np.nan) if med_hol else np.nan,
                            med_nor.get(h, np.nan),
                            fallback=global_factor)
        for h in range(24)
    }


def _apply_holiday_adjustments(pred_calls, pred_tmo, df_recent_calls, df_recent_tmo, holidays_set):
    # Llamadas: factores por hora desde histórico de llamadas
    factors_calls_by_hour = _holiday_hour_factors(df_recent_calls, TARGET_CALLS, holidays_set)

    # TMO: si no hay columna TARGET_TMO en histórico, factores neutros
    if df_recent_tmo is None or df_recent_tmo.empty or (TARGET_TMO not in df_recent_tmo.columns):
        factors_tmo_by_hour = {int(h): 1.0 for h in range(24)}
    else:
        factors_tmo_by_hour = _holiday_hour_factors(df_recent_tmo, TARGET_TMO, holidays_set)

    # Limitar factores
    factors_calls_by_hour = {h: float(np.clip(v, 0.10, 1.60)) for h, v in factors_calls_by_hour.items()}
    factors_tmo_by_hour   = {h: float(np.clip(v, 0.70, 1.50)) for h, v in factors_tmo_by_hour.items()}

    calls_adj = []
    tmo_adj   = []
    for ts, c, t in zip(pred_calls.index, pred_calls.values, pred_tmo.values):
        h = int(ts.hour)
        calls_adj.append(c * factors_calls_by_hour.get(h, 1.0))
        tmo_adj.append(t * factors_tmo_by_hour.get(h, 1.0))

    calls_adj = pd.Series(calls_adj, index=pred_calls.index, name=pred_calls.name)
    tmo_adj   = pd.Series(tmo_adj,   index=pred_tmo.index,   name=pred_tmo.name)

    # Rebote post-feriado
    calls_adj = _apply_post_holiday_adjustment(calls_adj, holidays_set, POST_HOLIDAY_HOURS, factor_calls=1.08, factor_tmo=1.03)
    tmo_adj   = _apply_post_holiday_adjustment(tmo_adj,   holidays_set, POST_HOLIDAY_HOURS, factor_calls=1.06, factor_tmo=1.03)

    return calls_adj, tmo_adj


# ==============================
# Núcleo de inferencia
# ==============================
def forecast_120d(df_hist_calls: pd.DataFrame, horizon_days: int = 120, holidays_set: Optional[Set] = None):
    """
    - Planner de llamadas con histórico de llamadas.
    - TMO EXCLUSIVO desde data/TMO_HISTORICO.csv con proceso ITERATIVO (igual que llamadas).
    - Ajustes por feriados y post-feriados separados (llamadas y TMO).
    - (Opcional) CAP de OUTLIERS por (dow,hour) con mediana+MAD (separado por serie).
    - Erlang C y salidas JSON.
    """
    # === Artefactos ===
    m_pl = tf.keras.models.load_model(PLANNER_MODEL, compile=False)
    sc_pl = joblib.load(PLANNER_SCALER)
    cols_pl = _load_cols(PLANNER_COLS)

    m_tmo = tf.keras.models.load_model(TMO_MODEL, compile=False)
    sc_tmo = joblib.load(TMO_SCALER)
    cols_tmo = _load_cols(TMO_COLS)

    # ===== Parse & limpieza (HISTÓRICO DE LLAMADAS) =====
    df = df_hist_calls.copy()
    df = ensure_ts(df)

    # normalizar nombres
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # aceptar alias de columna para llamadas
    if TARGET_CALLS not in df.columns:
        found = None
        for a in CALLS_ALIASES:
            if a in df.columns:
                found = a
                break
        if found is not None:
            df = df.rename(columns={found: TARGET_CALLS})
        else:
            raise ValueError(
                f"No se encontró ninguna columna de llamadas. "
                f"Esperaba una de {CALLS_ALIASES} en el histórico."
            )

    df[TARGET_CALLS] = pd.to_numeric(df[TARGET_CALLS], errors="coerce")
    df = df.dropna(subset=[TARGET_CALLS])
    df = df.sort_index()

    # forward fill de auxiliares de llamadas si existen
    if "feriados" in df.columns:
        df["feriados"] = df["feriados"].ffill()

    last_ts = df.index.max()
    start_hist_calls = last_ts - pd.Timedelta(days=HIST_WINDOW_DAYS)
    df_recent_calls = df.loc[df.index >= start_hist_calls].copy()
    if df_recent_calls.empty:
        df_recent_calls = df.copy()

    # ===== Horizonte futuro común =====
    future_ts = pd.date_range(
        last_ts + pd.Timedelta(hours=1),
        periods=horizon_days * 24,
        freq="h",
        tz=TIMEZONE
    )

    # ===== Planner iterativo (LLAMADAS) =====
    if "feriados" in df_recent_calls.columns:
        dfp = df_recent_calls[[TARGET_CALLS, "feriados"]].copy()
    else:
        dfp = df_recent_calls[[TARGET_CALLS]].copy()
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

    # ===== TMO EXCLUSIVO DESDE data/TMO_HISTORICO.csv (ITERATIVO) =====
    TMO_FILE = os.path.join("data", "TMO_HISTORICO.csv")
    if not os.path.exists(TMO_FILE):
        raise FileNotFoundError(f"No se encontró {TMO_FILE}. El TMO debe provenir exclusivamente de este archivo.")

    df_tmo_hist = pd.read_csv(TMO_FILE, low_memory=False)
    # asegurar timestamp
    if "ts" not in df_tmo_hist.columns:
        df_tmo_hist = df_tmo_hist.reset_index().rename(columns={"index": "ts"})
    df_tmo_hist = ensure_ts(df_tmo_hist)
    df_tmo_hist = df_tmo_hist.sort_index()

    # normalizar nombres
    df_tmo_hist.columns = [c.lower().strip().replace(" ", "_") for c in df_tmo_hist.columns]

    # construir tmo_general si es posible (ponderado por cantidades)
    if "tmo_general" not in df_tmo_hist.columns and all(c in df_tmo_hist.columns for c in
        ["tmo_comercial","q_comercial","tmo_tecnico","q_tecnico","q_general"]):
        df_tmo_hist["tmo_general"] = (
            pd.to_numeric(df_tmo_hist["tmo_comercial"], errors="coerce") * pd.to_numeric(df_tmo_hist["q_comercial"], errors="coerce") +
            pd.to_numeric(df_tmo_hist["tmo_tecnico"],   errors="coerce") * pd.to_numeric(df_tmo_hist["q_tecnico"],   errors="coerce")
        ) / (pd.to_numeric(df_tmo_hist["q_general"], errors="coerce") + 1e-6)

    # si tampoco se pudo construir, intentar alias comunes
    if "tmo_general" not in df_tmo_hist.columns:
        for alt in ["tmo", "aht", "duracion_promedio"]:
            if alt in df_tmo_hist.columns:
                df_tmo_hist["tmo_general"] = pd.to_numeric(df_tmo_hist[alt], errors="coerce")
                break

    # proporciones si existen (no imprescindibles para el modelo)
    if "q_llamadas_comercial" in df_tmo_hist.columns and "q_llamadas_general" in df_tmo_hist.columns:
        df_tmo_hist["proporcion_comercial"] = pd.to_numeric(df_tmo_hist["q_llamadas_comercial"], errors="coerce") / (pd.to_numeric(df_tmo_hist["q_llamadas_general"], errors="coerce") + 1e-6)
        if "q_llamadas_tecnico" in df_tmo_hist.columns:
            df_tmo_hist["proporcion_tecnica"] = pd.to_numeric(df_tmo_hist["q_llamadas_tecnico"], errors="coerce") / (pd.to_numeric(df_tmo_hist["q_llamadas_general"], errors="coerce") + 1e-6)

    # validar que exista TARGET_TMO en el histórico para iterar
    if TARGET_TMO not in df_tmo_hist.columns:
        raise ValueError(
            f"No se encontró '{TARGET_TMO}' ni fue posible construirlo desde componentes/alias en {TMO_FILE}."
        )

    # Reciente de TMO
    start_hist_tmo = df_tmo_hist.index.max() - pd.Timedelta(days=HIST_WINDOW_DAYS)
    df_recent_tmo = df_tmo_hist.loc[df_tmo_hist.index >= start_hist_tmo].copy()
    if df_recent_tmo.empty:
        df_recent_tmo = df_tmo_hist.copy()

    # marca de feriados (si no existe en tmo_hist, la creamos on-the-fly para el cálculo/iteración)
    if "feriados" not in df_recent_tmo.columns:
        df_recent_tmo["feriados"] = [_is_holiday(ts, holidays_set) for ts in df_recent_tmo.index]

    # ===== Planner iterativo (TMO) =====
    if "feriados" in df_recent_tmo.columns:
        dft = df_recent_tmo[[TARGET_TMO, "feriados"]].copy()
    else:
        dft = df_recent_tmo[[TARGET_TMO]].copy()
    dft[TARGET_TMO] = pd.to_numeric(dft[TARGET_TMO], errors="coerce").ffill().fillna(1.0)

    for ts in future_ts:
        tmp_t = pd.concat([dft, pd.DataFrame(index=[ts])])
        tmp_t[TARGET_TMO] = tmp_t[TARGET_TMO].ffill()

        if "feriados" in tmp_t.columns:
            tmp_t.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)

        tmp_t = add_lags_mas(tmp_t, TARGET_TMO)
        tmp_t = add_time_parts(tmp_t)

        X_t = dummies_and_reindex(tmp_t.tail(1), cols_tmo)
        yhat_t = float(m_tmo.predict(sc_tmo.transform(X_t), verbose=0).flatten()[0])
        dft.loc[ts, TARGET_TMO] = max(1.0, yhat_t)  # TMO mínimo 1s para estabilidad

        if "feriados" in dft.columns:
            dft.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)

    pred_tmo = dft.loc[future_ts, TARGET_TMO]
    pred_tmo.name = TARGET_TMO

    # ===== Ajustes por feriados y post-feriados (separados) =====
    # (ya incluimos 'feriados' como feature en la iteración, pero mantenemos ajuste suave + caps)
    pred_calls_adj, pred_tmo_adj = _apply_holiday_adjustments(
        pred_calls, pred_tmo, df_recent_calls, df_recent_tmo, holidays_set
    )

    # ===== CAP de outliers por (dow,hour) =====
    if ENABLE_OUTLIER_CAP:
        caps_calls = _build_outlier_caps(df_recent_calls, TARGET_CALLS)
        caps_tmo   = _build_outlier_caps(df_recent_tmo,   TARGET_TMO)

        calls_capped = []
        tmo_capped   = []
        for ts, c, t in zip(pred_calls_adj.index, pred_calls_adj.values, pred_tmo_adj.values):
            key = (int(ts.dayofweek), int(ts.hour))
            c_cap = caps_calls.get(key, np.inf)
            t_cap = caps_tmo.get(key,   np.inf)

            calls_capped.append(min(float(c), float(c_cap)))
            tmo_capped.append(min(float(t), float(t_cap)))

        pred_calls_adj = pd.Series(calls_capped, index=pred_calls_adj.index, name=pred_calls_adj.name)
        pred_tmo_adj   = pd.Series(tmo_capped,   index=pred_tmo_adj.index,   name=pred_tmo_adj.name)

    # ===== Erlang (agentes requeridos) =====
    df_erlang = pd.DataFrame({
        TARGET_CALLS: pred_calls_adj.values,
        TARGET_TMO:   pred_tmo_adj.values
    }, index=future_ts)

    df_erlang["aht"] = df_erlang[TARGET_TMO]  # si tu AHT=TMO; si no, ajusta aquí
    df_erlang["sla_s"] = 22     # SLA en segundos (parametrizable)
    df_erlang["target"] = 0.8   # 80% dentro de 22s (parametrizable)
    df_erlang["interval"] = 3600  # 1h

    req_agents = []
    for ts, row in df_erlang.iterrows():
        req = required_agents(
            calls=float(row[TARGET_CALLS]),
            aht=float(row["aht"]),
            interval_seconds=int(row["interval"]),
            sla_s=int(row["sla_s"]),
            target=float(row["target"])
        )
        req_agents.append(req)
    df_erlang["agents_required"] = req_agents

    # ===== Salidas JSON =====
    write_hourly_json(df_erlang[[TARGET_CALLS, TARGET_TMO, "agents_required"]], os.path.join(PUBLIC_DIR, "forecast_hourly.json"))
    write_daily_json(df_erlang[[TARGET_CALLS, TARGET_TMO, "agents_required"]],  os.path.join(PUBLIC_DIR, "forecast_daily.json"))

    return df_erlang
