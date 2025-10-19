# src/inferencia/inferencia_core.py
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from .features import ensure_ts, add_time_parts, add_lags_mas, dummies_and_reindex
from .erlang import required_agents, schedule_agents
from .utils_io import write_daily_json, write_hourly_json
from ..data.loader_tmo import load_historico_tmo

TIMEZONE = "America/Santiago"
PUBLIC_DIR = "public"

PLANNER_MODEL = "models/modelo_planner.keras"
PLANNER_SCALER = "models/scaler_planner.pkl"
PLANNER_COLS = "models/training_columns_planner.json"

TMO_MODEL = "models/modelo_tmo.keras"
TMO_SCALER = "models/scaler_tmo.pkl"
TMO_COLS = "models/training_columns_tmo.json"

TARGET_CALLS = "recibidos_nacional"

def _load_cols(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _is_holiday(ts, holidays_set):
    if holidays_set is None:
        return False
    try:
        d = ts.tz_convert(TIMEZONE).date() if ts.tzinfo else ts.date()
    except Exception:
        d = ts.date()
    return (str(d) in holidays_set)

def compute_holiday_factors(df: pd.DataFrame, holidays_set: set,
                            k_weekday: float = 1.00,
                            k_weekend: float = 1.00):
    """
    Calcula factores por feriado (plantilla). Mantiene compatibilidad con el flujo existente.
    Devuelve tuplas de factores para llamadas y TMO.
    """
    d = df.copy()
    d = add_time_parts(d)
    base = (d.groupby(["dow","hour"])
              .agg(calls=("recibidos_nacional","median"),
                   tmo=("tmo_general","median"))
              .reset_index())
    base.rename(columns={"calls":"med_calls","tmo":"med_tmo"}, inplace=True)
    capped = d.merge(base, on=["dow","hour"], how="left")
    # factores (simplificados; tu lógica original puede ser más extensa)
    f_calls_by_hour = {h:1.0 for h in range(24)}
    f_tmo_by_hour = {h:1.0 for h in range(24)}
    g_calls = 1.0
    g_tmo = 1.0
    post_calls_by_hour = {h:1.0 for h in range(24)}
    return f_calls_by_hour, f_tmo_by_hour, g_calls, g_tmo, post_calls_by_hour

def apply_holiday_adjustment(df_hourly: pd.DataFrame, holidays_set: set,
                             f_calls_by_hour, f_tmo_by_hour, g_calls, g_tmo, post_calls_by_hour):
    out = df_hourly.copy()
    idx = out.index
    is_hol = pd.Series([_is_holiday(ts, holidays_set) for ts in idx], index=idx)
    # ejemplo simple (tu versión original puede ser más sofisticada)
    for ts in idx:
        h = getattr(ts, "hour", pd.Timestamp(ts).hour)
        if is_hol.loc[ts]:
            out.at[ts, "calls"] = out.at[ts, "calls"] * f_calls_by_hour.get(h,1.0) * g_calls
            out.at[ts, "tmo_s"] = out.at[ts, "tmo_s"] * f_tmo_by_hour.get(h,1.0) * g_tmo
        else:
            out.at[ts, "calls"] = out.at[ts, "calls"] * post_calls_by_hour.get(h,1.0)
    return out

def inferencia_core(df_hist_calls: pd.DataFrame,
                    holidays_json_path: str = "data/Feriados_Chilev2.csv",
                    horizon_days: int = 14):
    """
    Recibe histórico de llamadas (hosting) y genera predicciones + agentes. Escribe salidas JSON.
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
        # normaliza nombre 'recibidos' -> TARGET_CALLS
        for c in df.columns:
            if c.lower().strip() == "recibidos":
                df = df.rename(columns={c: TARGET_CALLS})
                break

    # agregación por ts
    agg = {TARGET_CALLS: "sum"}
    if "feriados" in df.columns:
        agg["feriados"] = "max"

    df = (df.groupby("ts", as_index=True)
            .agg(agg)
            .sort_index())

    # completar feriados a 0 si no existe
    if "feriados" not in df.columns:
        df["feriados"] = 0

    # ===== Horizonte futuro =====
    last_ts = df.index.max()
    start_hist = last_ts - pd.Timedelta(days=120)  # recorte para lags/MA recientes
    df_recent = df.loc[df.index >= start_hist].copy()
    if df_recent.empty:
        df_recent = df.copy()

    future_ts = pd.date_range(
        last_ts + pd.Timedelta(hours=1),
        periods=horizon_days * 24,
        freq="h",
        tz=TIMEZONE
    )

    # ===== Planner iterativo (con 'feriados' futuro) =====
    dfp = df_recent.copy()
    dfp = add_time_parts(dfp)
    dfp = add_lags_mas(dfp, TARGET_CALLS)

    def _predict_calls_row(row_feats):
        Xr = dummies_and_reindex(row_feats, cols_pl)
        y = m_pl.predict(sc_pl.transform(Xr), verbose=0).flatten()[0]
        return max(0.0, float(y))

    # Predicción por paso (usa feriados del calendario)
    holidays_set = None
    try:
        # El repo trae CSV de feriados; si fuera JSON, adáptalo aquí.
        # Asumimos columna 'Fecha' en YYYY-MM-DD.
        hdf = pd.read_csv(holidays_json_path)
        # Normalizamos a set de fechas string "YYYY-MM-DD"
        if "Fecha" in hdf.columns:
            holidays_set = set(pd.to_datetime(hdf["Fecha"], dayfirst=True, errors="coerce").dropna().dt.date.astype(str))
        else:
            # compat: si hay una columna 'date'
            if "date" in hdf.columns:
                holidays_set = set(pd.to_datetime(hdf["date"], errors="coerce").dropna().dt.date.astype(str))
    except Exception:
        holidays_set = None

    preds = []
    for ts in future_ts:
        row = {}
        # construir fila con lags/MA desde dfp
        # copiamos últimas 168h para asegurar rolling y lags
        d = dfp.copy().tail(168)
        # agregamos punto ficticio ts-1h con NaN y luego imputación
        # (flujo simple; tu versión original podría diferir)
        last_row = d.iloc[[-1]].copy()
        last_row.index = [ts - pd.Timedelta(hours=1)]
        d = pd.concat([d, last_row]).sort_index()
        d = add_time_parts(d)
        d = add_lags_mas(d, TARGET_CALLS)
        row_feats = d.iloc[[-1]].copy()
        if holidays_set:
            row_feats["feriados"] = int(_is_holiday(ts, holidays_set))
        Xr = dummies_and_reindex(row_feats, cols_pl)
        yhat = m_pl.predict(sc_pl.transform(Xr), verbose=0).flatten()[0]
        yhat = max(0.0, float(yhat))
        preds.append((ts, yhat))

        # actualizar dfp con la predicción para siguientes lags
        dfp.loc[ts, TARGET_CALLS] = yhat
        if holidays_set:
            dfp.loc[ts, "feriados"] = _is_holiday(ts, holidays_set)

    # Serie futura de llamadas
    pred_calls = pd.Series({ts: y for ts, y in preds})
    pred_calls.index = pd.DatetimeIndex(pred_calls.index).tz_convert(TIMEZONE)

    # ===== TMO por hora =====
    base_tmo = pd.DataFrame(index=future_ts)
    base_tmo[TARGET_CALLS] = pred_calls.values

    # Si el histórico de hosted trae proporciones y tmo por tipo, reutilizamos el último valor conocido
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
            f_calls_by_hour, f_tmo_by_hour, g_calls, g_tmo, post_calls_by_hour
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

    # ======= Exportar TMO histórico (últimos 90 días) sin tocar llamadas =======
    try:
        # 1) Cargar histórico TMO usando el loader ya existente
        df_tmo_hist = load_historico_tmo("data")  # index=ts, incluye tmo_general (o lo calcula ponderado)

        # 2) Ventana de 90 días respecto del último timestamp del pipeline
        start_90 = last_ts - pd.Timedelta(days=90)
        df_tmo_90 = df_tmo_hist.loc[(df_tmo_hist.index >= start_90) & (df_tmo_hist.index <= last_ts)].copy()

        # 3) TMO horario -> { ts, tmo_hora }
        df_tmo_hourly = (
            df_tmo_90[["tmo_general"]]
            .rename(columns={"tmo_general": "tmo_hora"})
            .reset_index()
        )
        df_tmo_hourly["ts"] = pd.to_datetime(df_tmo_hourly["ts"], errors="coerce")
        df_tmo_hourly = df_tmo_hourly.dropna(subset=["ts"])
        df_tmo_hourly["ts"] = df_tmo_hourly["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

        # 4) TMO diario ponderado por Q_LLAMADAS_GENERAL -> { ts (YYYY-MM-DD), tmo_dia }
        tmp = df_tmo_90.reset_index().rename(columns={"index": "ts"})
        tmp["ts"] = pd.to_datetime(tmp["ts"], errors="coerce")
        tmp = tmp.dropna(subset=["ts"])
        tmp["fecha"] = tmp["ts"].dt.date.astype(str)

        if "q_llamadas_general" in tmp.columns:
            tmp["peso"] = tmp["q_llamadas_general"].astype(float).clip(lower=0)
        else:
            tmp["peso"] = 0.0

        def _ponderado(g):
            num = (g["tmo_general"].astype(float) * g["peso"]).sum()
            den = float(g["peso"].sum())
            if den <= 0:
                return float(g["tmo_general"].astype(float).mean())
            return num / den

        df_tmo_daily = (
            tmp.groupby("fecha", as_index=False)
               .apply(lambda g: pd.Series({"tmo_dia": _ponderado(g)}))
        )

        # 5) Guardar salidas TMO en public/ (llamadas quedan intactas)
        from .utils_io import write_json
        write_json(f"{PUBLIC_DIR}/tmo_horario.json",
                   df_tmo_hourly[["ts", "tmo_hora"]].to_dict(orient="records"))
        write_json(f"{PUBLIC_DIR}/tmo_diario.json",
                   df_tmo_daily[["fecha", "tmo_dia"]].rename(columns={"fecha": "ts"}).to_dict(orient="records"))

        # 6) Debug liviano
        write_json(f"{PUBLIC_DIR}/debug_tmo_hist.json",
                   {"rows_hourly": int(len(df_tmo_hourly)),
                    "rows_daily": int(len(df_tmo_daily))})
    except Exception as e:
        # No interrumpir el pipeline de llamadas si falla el TMO histórico
        print(f"[WARN] TMO histórico (90d) no exportado: {e}")

    return df_hourly


