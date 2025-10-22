# src/main.py
import os
import argparse
import pandas as pd
import numpy as np

from inferencia.features import ensure_ts, add_time_parts
from inferencia.inferencia_core import forecast_120d, TIMEZONE, TARGET_TMO
from inferencia.erlang import required_agents
from inferencia.utils_io import write_daily_json, write_hourly_json

PUBLIC_DIR = "public"
TARGET_CALLS = "recibidos_nacional"

# -------------------------
# Helpers de carga y DDL TMO
# -------------------------
def _read_any(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe archivo: {path}")
    low = path.lower()
    if low.endswith(".csv"):
        try:
            return pd.read_csv(path, low_memory=False)
        except Exception:
            return pd.read_csv(path, delimiter=";", low_memory=False)
    elif low.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    else:
        # intento genérico
        return pd.read_csv(path, low_memory=False)

def _load_holidays(holidays_file: str) -> set:
    """
    Lee un CSV/XLS con una columna de fechas (nombre flexible) y arma un set de date().
    No pisa la columna 'feriados' del histórico; solo sirve para los ajustes en inferencia_core.
    """
    if not os.path.exists(holidays_file):
        return set()
    df = _read_any(holidays_file)
    # buscar columna de fecha probable
    for cand in ["fecha", "date", "dia", "día", "day"]:
        if cand in {c.lower().strip(): c for c in df.columns}:
            col = {c.lower().strip(): c for c in df.columns}[cand]
            dt = pd.to_datetime(df[col], errors="coerce", dayfirst=True).dt.date
            return set(d for d in dt.dropna().tolist())
    # si solo hay una columna, intentar parsearla
    if df.shape[1] == 1:
        dt = pd.to_datetime(df.iloc[:, 0], errors="coerce", dayfirst=True).dt.date
        return set(d for d in dt.dropna().tolist())
    # fallback vacío
    return set()

def _coerce_num(s):
    return pd.to_numeric(s, errors="coerce")

def _tmo_ponderado_row(row) -> float:
    """
    Calcula TMO general ponderado:
    - Si existen q_llamadas_comercial/tecnico y tmo_comercial/tecnico -> ponderación por cantidades
    - Si existen proporciones y tmo por tipo -> ponderación por proporción
    - Si existe TARGET_TMO ya calculado -> usarlo
    """
    # Opción 1: por cantidades
    qc = row.get("q_llamadas_comercial", np.nan)
    qt = row.get("q_llamadas_tecnico", np.nan)
    tg = row.get("q_llamadas_general", np.nan)
    tc = row.get("tmo_comercial", np.nan)
    tt = row.get("tmo_tecnico", np.nan)

    qc = np.nan if pd.isna(qc) else float(qc)
    qt = np.nan if pd.isna(qt) else float(qt)
    tg = np.nan if pd.isna(tg) else float(tg)
    tc = np.nan if pd.isna(tc) else float(tc)
    tt = np.nan if pd.isna(tt) else float(tt)

    if (not pd.isna(qc)) and (not pd.isna(qt)) and (not pd.isna(tc)) and (not pd.isna(tt)):
        denom = (qc + qt) if pd.isna(tg) else tg
        if denom and denom > 0:
            return (tc * (qc / denom) + tt * (qt / denom))

    # Opción 2: por proporciones
    pc = row.get("proporcion_comercial", np.nan)
    pt = row.get("proporcion_tecnica", np.nan)
    pc = np.nan if pd.isna(pc) else float(pc)
    pt = np.nan if pd.isna(pt) else float(pt)
    if (not pd.isna(pc)) and (not pd.isna(pt)) and (not pd.isna(tc)) and (not pd.isna(tt)):
        s = pc + pt
        if s > 0:
            return (tc * pc + tt * pt) / s

    # Opción 3: TARGET_TMO ya viene
    tg = row.get(TARGET_TMO, np.nan)
    if not pd.isna(tg):
        return float(tg)

    return np.nan

def _build_tmo_future_from_ddl(df_tmo_hist: pd.DataFrame, future_index: pd.DatetimeIndex) -> pd.Series:
    """
    Construye TMO horario futuro desde histórico (DDL):
    - Calcula TMO ponderado por fila (si hay cantidades/proporciones).
    - Arma perfil por (dow, hour) usando últimas 6–8 semanas.
    - Proyecta a futuro mapeando (dow, hour).
    - Fallback: mediana por hour, luego mediana global, luego 0.
    """
    if df_tmo_hist is None or df_tmo_hist.empty:
        return pd.Series(np.zeros(len(future_index), dtype=float), index=future_index)

    d = df_tmo_hist.copy()
    d = ensure_ts(d)  # tz=America/Santiago, drop horas ambiguas (modo original)
    if d.empty:
        return pd.Series(np.zeros(len(future_index), dtype=float), index=future_index)

    # calcular TMO ponderado por fila
    d_cols = {c.lower().strip(): c for c in d.columns}
    # intentar asegurar nombres básicos
    # (si no existen, _tmo_ponderado_row usará TARGET_TMO si está)
    # compute row-wise
    tmo_p = d.apply(_tmo_ponderado_row, axis=1)
    d["tmo_pond"] = _coerce_num(tmo_p)

    # quedarnos con ventana reciente para el perfil (evita drift)
    if not d.index.is_monotonic_increasing:
        d = d.sort_index()
    last_ts = d.index.max()
    d_recent = d.loc[d.index >= (last_ts - pd.Timedelta(days=56))].copy()
    if d_recent["tmo_pond"].dropna().empty:
        # usar una ventana más amplia si no hay datos
        d_recent = d.copy()

    d_recent = add_time_parts(d_recent)
    # perfil por (dow, hour): mediana robusta
    prof = d_recent.groupby(["dow", "hour"])["tmo_pond"].median()

    # fallback por hora (agregando todos los días) y global
    med_by_hour = d_recent.groupby("hour")["tmo_pond"].median()
    med_global = float(d_recent["tmo_pond"].median()) if not d_recent["tmo_pond"].dropna().empty else 0.0

    # armar serie futura
    fut = pd.DataFrame(index=future_index)
    fut = add_time_parts(fut)
    vals = []
    for _, r in fut.iterrows():
        key = (int(r["dow"]), int(r["hour"]))
        v = prof.get(key, np.nan)
        if pd.isna(v):
            v = med_by_hour.get(int(r["hour"]), np.nan)
        if pd.isna(v):
            v = med_global
        vals.append(float(v) if np.isfinite(v) else 0.0)

    s = pd.Series(vals, index=future_index, dtype=float)
    # limpiar extremos
    s = s.clip(lower=0).fillna(0.0)
    # redondeo similar a la salida original (entero)
    s = np.round(s).astype(float)
    return s

# -------------------------
# Flujo principal
# -------------------------
def main(horizonte_dias: int):
    # Paths (configurables por ENV)
    HISTORICAL_FILE = os.getenv("HISTORICAL_FILE", "data/historical_data.csv")
    HISTORICAL_TMO_FILE = os.getenv("HISTORICAL_TMO_FILE", "data/TMO_HISTORICO.csv")
    HOLIDAYS_FILE = os.getenv("HOLIDAYS_FILE", "data/Feriados_Chilev2.csv")

    os.makedirs(PUBLIC_DIR, exist_ok=True)

    # 1) Cargar histórico de llamadas (NO tocar 'feriados' si ya viene)
    dfh = _read_any(HISTORICAL_FILE)
    dfh = ensure_ts(dfh)

    # 2) Holidays set (solo para ajustes dentro de inferencia_core)
    holidays_set = _load_holidays(HOLIDAYS_FILE)

    # 3) Forecast base (llamadas originales + tmo_s interno provisional)
    #    OJO: forecast_120d ya escribe JSONs base; luego los sobrescribimos con TMO override.
    df_hourly = forecast_120d(dfh, horizonte_dias, holidays_set)

    # 4) Override de TMO final desde DDL/HISTÓRICO TMO (si existe)
    tmo_override_aplicado = False
    if os.path.exists(HISTORICAL_TMO_FILE):
        try:
            df_tmo_hist = _read_any(HISTORICAL_TMO_FILE)
            tmo_future = _build_tmo_future_from_ddl(df_tmo_hist, df_hourly.index)
            # asegurar Series
            if not isinstance(tmo_future, pd.Series):
                tmo_future = pd.Series(tmo_future, index=df_hourly.index, dtype=float)

            # override TMO (salida final)
            df_hourly["tmo_s"] = pd.to_numeric(tmo_future, errors="coerce").fillna(df_hourly["tmo_s"]).astype(float)

            # Recalcular agentes con TMO final
            df_hourly["agentes_requeridos"] = required_agents(
                traffic_calls=df_hourly["calls"].astype(float).values,
                aht_seconds=df_hourly["tmo_s"].astype(float).values
            ).astype(int)

            tmo_override_aplicado = True
        except Exception as e:
            print(f"[TMO] Fallback activado: no se pudo aplicar override desde {HISTORICAL_TMO_FILE}. Error: {e}")

    if not tmo_override_aplicado:
        print("[TMO] Se mantiene TMO del modelo interno (no hubo DDL válido).")

    # 5) Persistir JSONs FINALES (sobrescriben los del paso base)
    df_hourly["calls"] = pd.to_numeric(df_hourly["calls"], errors="coerce").fillna(0.0).astype(int)
    df_hourly["tmo_s"] = pd.to_numeric(df_hourly["tmo_s"], errors="coerce").fillna(0.0).astype(int)
    if "agentes_requeridos" not in df_hourly.columns:
        df_hourly["agentes_requeridos"] = (df_hourly["calls"] / 20).round().astype(int)

    write_hourly_json(f"{PUBLIC_DIR}/prediccion_horaria.json", df_hourly,
                      calls_col="calls", tmo_col="tmo_s", agentes_col="agentes_requeridos")
    write_daily_json(f"{PUBLIC_DIR}/prediccion_diaria.json", df_hourly,
                     calls_col="calls", tmo_col="tmo_s")

    print("✅ Listo: llamadas (lógica original) + TMO override DDL aplicado." if tmo_override_aplicado
          else "✅ Listo: llamadas (lógica original); TMO interno (sin override).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizonte", type=int, default=int(os.getenv("HORIZONTE_DIAS", 120)))
    args = parser.parse_args()
    main(args.horizonte)
