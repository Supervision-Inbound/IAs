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
PLANNER_COLS  = "models/training_columns_planner.json"

TMO_MODEL = "models/modelo_tmo.keras"
TMO_SCALER = "models/scaler_tmo.pkl"
TMO_COLS   = "models/training_columns_tmo.json"

TARGET_CALLS = "q_llamadas_general"
TARGET_TMO   = "tmo_general"

# ====== Config de POST-FERIADOS (afecta llamadas y TMO) ======
POST_HOLIDAY_HOURS = 48  # ventana post-feriado para “rebote”

# ====== Config de ventanas y límites ======
HIST_WINDOW_DAYS = 90

# ======= NUEVO: Guardrail Outliers (config) =======
ENABLE_OUTLIER_CAP = True   # <- ponlo en False si quieres desactivarlo
K_WEEKDAY = 6.0             # techos +K*MAD en lun-vie
K_WEEKEND = 7.0             # techos +K*MAD en sáb-dom


def _load_cols(path: str):
    with open(path, "r") as f:
        return json.load(f)


# ========= Helpers de FERIADOS (PORTADOS + EXTENDIDOS) =========
def _safe_ratio(num, den, fallback=np.nan):
    try:
        v = float(num) / float(den)
        if not np.isfinite(v):
            return fallback
        return v
    except Exception:
        return fallback


def _is_holiday(ts, holidays_set):
    if holidays_set is None:
        return 0
    try:
        d = ts.tz_convert(TIMEZONE).date()
    except Exception:
        d = ts.date()
    return 1 if d in holidays_set else 0


def _is_post_holiday(ts, holidays_set):
    if holidays_set is None:
        return 0
    try:
        d = ts.tz_convert(TIMEZONE).date()
    except Exception:
        d = ts.date()
    # post-feriado si hubo feriado en las 48h previas
    prev_days = [d - pd.Timedelta(days=1), d - pd.Timedelta(days=2)]
    return 1 if (d - pd.Timedelta(days=1)) in holidays_set or (d - pd.Timedelta(days=2)) in holidays_set else 0


def _weekday_mad_caps(s: pd.Series, k: float):
    """Devuelve techo por hora basado en mediana+K*MAD para controlar outliers."""
    med = np.nanmedian(s)
    mad = np.nanmedian(np.abs(s - med)) + 1e-9
    return med + k * 1.4826 * mad


def _build_outlier_caps(df_recent: pd.DataFrame, col: str):
    """Crea techos horario*dow para col con K distintos en weekday/weekend."""
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


def _apply_post_holiday_adjustment(series: pd.Series, holidays_set: set | None, hours_window: int = POST_HOLIDAY_HOURS, factor_calls=1.08, factor_tmo=1.03):
    """Ajuste post-feriado: leve aumento en horas posteriores, configurable."""
    if holidays_set is None or series.empty:
        return series

    idx = series.index
    is_post = []
    for ts in idx:
        if _is_post_holiday(ts, holidays_set):
            is_post.append(1)
        else:
            is_post.append(0)
    is_post = np.array(is_post, dtype=float)

    # Suaviza con ventana móvil para extender efecto
    kernel = np.ones(min(hours_window, len(is_post)), dtype=float)
    post_signal = np.convolve(is_post, kernel, mode="same")
    post_signal = post_signal / (post_signal.max() + 1e-9)

    # Interpola un factor por hora
    if series.name == TARGET_CALLS:
        f = factor_calls
    else:
        f = factor_tmo

    adjusted = series.values * (1.0 + (f - 1.0) * post_signal)
    return pd.Series(adjusted, index=series.index, name=series.name)


def _holiday_hour_factors(df_recent: pd.DataFrame, col: str, holidays_set: set | None):
    """
    Calcula factores (feriado vs normal) por hora para ajustar el forecast.
    """
    if holidays_set is None or df_recent.empty:
        return {int(h): 1.0 for h in range(24)}

    df_recent = df_recent.copy()
    if "feriados" not in df_recent.columns:
        df_recent["feriados"] = 0

    df_recent["hour"] = df_recent.index.hour
    df_recent[col] = pd.to_numeric(df_recent[col], errors="coerce")

    med_nor = df_recent.loc[df_recent["feriados"] == 0].groupby("hour")[col].median().to_dict()
    med_hol = df_recent.loc[df_recent["feriados"] == 1].groupby("hour")[col].median().to_dict()

    global_factor = _safe_ratio(np.nanmedian(list(med_hol.values())), np.nanmedian(list(med_nor.values())), fallback=1.0)
    if not np.isfinite(global_factor):
        global_factor = 1.0

    if med_hol:
        factors = {
            int(h): _safe_ratio(med_hol.get(h, np.nan), med_nor.get(h, np.nan), fallback=global_factor)
            for h in range(24)
        }
    else:
        factors = {int(h): 1.0 for h in range(24)}
    return factors


def _apply_holiday_adjustments(pred_calls: pd.Series, pred_tmo: pd.Series, df_recent: pd.DataFrame, holidays_set: set | None):
    """Ajusta por feriados y post-feriados con factores por hora y caps razonables."""
    # Factores por hora
    factors_calls_by_hour = _holiday_hour_factors(df_recent, TARGET_CALLS, holidays_set)
    factors_tmo_by_hour   = _holiday_hour_factors(df_recent, TARGET_TMO, holidays_set)

    # Factor global de TMO por hora si tenemos medianas
    med_nor_tmo = df_recent.loc[df_recent.get("feriados", 0) == 0].groupby(df_recent.index.hour)[TARGET_TMO].median().to_dict() if TARGET_TMO in df_recent.columns else {}
    med_hol_tmo = df_recent.loc[df_recent.get("feriados", 0) == 1].groupby(df_recent.index.hour)[TARGET_TMO].median().to_dict() if TARGET_TMO in df_recent.columns else {}

    global_tmo_factor = _safe_ratio(np.nanmedian(list(med_hol_tmo.values())), np.nanmedian(list(med_nor_tmo.values())), fallback=1.0)
    if not np.isfinite(global_tmo_factor):
        global_tmo_factor = 1.0

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

    # ---- NEW: aplicar factores por hora según el índice temporal
    calls_adj = []
    tmo_adj   = []
    for ts, c, t in zip(pred_calls.index, pred_calls.values, pred_tmo.values):
        h = int(ts.hour)
        calls_adj.append(c * factors_calls_by_hour.get(h, 1.0))
        tmo_adj.append(t * factors_tmo_by_hour.get(h, 1.0))

    calls_adj = pd.Series(calls_adj, index=pred_calls.index, name=pred_calls.name)
    tmo_adj   = pd.Series(tmo_adj,   index=pred_tmo.index,   name=pred_tmo.name)

    # Post-feriados (rebote)
    calls_adj = _apply_post_holiday_adjustment(calls_adj, holidays_set, POST_HOLIDAY_HOURS, factor_calls=1.08, factor_tmo=1.03)
    tmo_adj   = _apply_post_holiday_adjustment(tmo_adj,   holidays_set, POST_HOLIDAY_HOURS, factor_calls=1.06, factor_tmo=1.03)

    return calls_adj, tmo_adj


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

    # ===== Parse & limpieza =====
    df = df_hist_calls.copy()
    df = ensure_ts(df)
    if TARGET_CALLS not in df.columns:
        raise ValueError(f"No se encontró la columna {TARGET_CALLS} en el histórico de llamadas.")

    df[TARGET_CALLS] = pd.to_numeric(df[TARGET_CALLS], errors="coerce")
    df = df.dropna(subset=[TARGET_CALLS])
    df = df.sort_index()

    # forward fill de auxiliares (si existen en histórico)
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

    # === Semilla TMO desde data/TMO_HISTORICO.csv (igual que ejemplo.py) ===
    import os
    TMO_FILE = os.path.join("data", "TMO_HISTORICO.csv")
    last_vals = None
    if os.path.exists(TMO_FILE):
        try:
            df_tmo_hist = pd.read_csv(TMO_FILE, low_memory=False)
            # Normalizar columnas y timestamp
            if "ts" not in df_tmo_hist.columns:
                df_tmo_hist = df_tmo_hist.reset_index().rename(columns={"index":"ts"})
            df_tmo_hist = ensure_ts(df_tmo_hist)
            df_tmo_hist.columns = [c.lower().strip().replace(' ', '_') for c in df_tmo_hist.columns]
            # Asegurar tmo_general si es posible (ponderado)
            if "tmo_general" not in df_tmo_hist.columns and all(c in df_tmo_hist.columns for c in ["tmo_comercial","q_comercial","tmo_tecnico","q_tecnico","q_general"]):
                df_tmo_hist["tmo_general"] = (
                    df_tmo_hist["tmo_comercial"]*df_tmo_hist["q_comercial"] +
                    df_tmo_hist["tmo_tecnico"]  *df_tmo_hist["q_tecnico"]
                ) / (df_tmo_hist["q_general"] + 1e-6)
            # Proporciones si existen
            if "q_llamadas_comercial" in df_tmo_hist.columns and "q_llamadas_general" in df_tmo_hist.columns:
                df_tmo_hist["proporcion_comercial"] = df_tmo_hist["q_llamadas_comercial"] / (df_tmo_hist["q_llamadas_general"] + 1e-6)
                if "q_llamadas_tecnico" in df_tmo_hist.columns:
                    df_tmo_hist["proporcion_tecnica"]  = df_tmo_hist["q_llamadas_tecnico"] / (df_tmo_hist["q_llamadas_general"] + 1e-6)
            # Extraer última fila con campos clave si existen
            needed = ["proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico"]
            available = [c for c in needed if c in df_tmo_hist.columns]
            if available:
                last_vals = df_tmo_hist.sort_index().iloc[[-1]][available].copy()
                # Completar faltantes con 0 si algunas columnas no están
                for c in needed:
                    if c not in last_vals.columns:
                        last_vals[c] = 0.0
        except Exception:
            last_vals = None

    # Fallback al comportamiento previo si no se pudo leer el archivo
    if last_vals is None:
        if {"proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico"}.issubset(df.columns):
            last_vals = df.ffill().iloc[[-1]][["proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico"]]
        else:
            last_vals = pd.DataFrame([[0,0,0,0]], columns=["proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico"])

    for c in ["proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico"]:
        base_tmo[c] = float(last_vals[c].iloc[0]) if c in last_vals.columns else 0.0

    # Construcción de features de TMO
    df_tmo_feats = base_tmo.copy()
    df_tmo_feats = add_time_parts(df_tmo_feats)
    df_tmo_feats = dummies_and_reindex(df_tmo_feats, cols_tmo)

    y_tmo = m_tmo.predict(sc_tmo.transform(df_tmo_feats), verbose=0).flatten()
    y_tmo = np.clip(y_tmo, 1.0, None)  # TMO no puede ser <= 0

    pred_tmo = pd.Series(y_tmo, index=future_ts, name=TARGET_TMO)

    # ===== Ajustes por feriados y post-feriados =====
    pred_calls_adj, pred_tmo_adj = _apply_holiday_adjustments(pred_calls, pred_tmo, df_recent, holidays_set)

    # ===== CAP de outliers por (dow,hour) =====
    if ENABLE_OUTLIER_CAP:
        caps_calls = _build_outlier_caps(df_recent, TARGET_CALLS)
        caps_tmo   = _build_outlier_caps(df_recent, TARGET_TMO) if TARGET_TMO in df_recent.columns else {}

        # aplica caps
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
    # NOTA: Usa pred_calls_adj y pred_tmo_adj
    df_erlang = pd.DataFrame({
        TARGET_CALLS: pred_calls_adj.values,
        TARGET_TMO:   pred_tmo_adj.values
    }, index=future_ts)

    df_erlang["aht"] = df_erlang[TARGET_TMO]  # si tu AHT=TMO; si no, ajusta aquí
    df_erlang["sla_s"] = 22     # SLA en segundos (puedes parametrizar)
    df_erlang["target"] = 0.8   # 80% dentro de 22s (puedes parametrizar)
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

    # Si tienes lógica de schedule_agents, puedes aplicarla aquí (opcional)
    # df_schedule = schedule_agents(df_erlang["agents_required"], shift_len=8, max_shifts=... )

    # ===== Salidas JSON =====
    write_hourly_json(df_erlang[[TARGET_CALLS, TARGET_TMO, "agents_required"]], PUBLIC_DIR + "/forecast_hourly.json")
    write_daily_json(df_erlang[[TARGET_CALLS, TARGET_TMO, "agents_required"]],  PUBLIC_DIR + "/forecast_daily.json")

    return df_erlang

