# src/main.py
import argparse
import os
import numpy as np
import pandas as pd

from src.inferencia.inferencia_core import forecast_120d
from src.inferencia.features import ensure_ts

from src.data.loader_tmo import load_historico_tmo
from src.inferencia.tmo_from_historico import forecast_tmo_from_historico
from src.inferencia.utils_io import write_hourly_json, write_daily_json

DATA_FILE = "data/historical_data.csv"
HOLIDAYS_FILE = "data/Feriados_Chilev2.csv"
TMO_HIST_FILE = "data/HISTORICO_TMO.csv"

TARGET_CALLS_NEW = "recibidos_nacional"
TARGET_TMO_NEW   = "tmo_general"
TZ = "America/Santiago"

def smart_read_historical(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    return pd.read_csv(path, delimiter=';', low_memory=False)

def parse_tmo_to_seconds(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().replace(",", ".")
    if s.replace(".", "", 1).isdigit():
        try: return float(s)
        except: return np.nan
    parts = s.split(":")
    try:
        if len(parts) == 3: return float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
        if len(parts) == 2: return float(parts[0])*60   + float(parts[1])
        return float(s)
    except:
        return np.nan

def load_holidays(csv_path: str) -> set:
    if not os.path.exists(csv_path): return set()
    fer = pd.read_csv(csv_path)
    cols_map = {c.lower().strip(): c for c in fer.columns}
    fecha_col = next((cols_map[k] for k in ("fecha","date","dia","día") if k in cols_map), None)
    if not fecha_col: return set()
    fechas = pd.to_datetime(fer[fecha_col].astype(str), dayfirst=True, errors="coerce").dropna().dt.date
    return set(fechas)

def mark_holidays_index(dt_index, holidays_set: set) -> pd.Series:
    tz = getattr(dt_index, "tz", None)
    idx_dates = dt_index.tz_convert(TZ).date if tz is not None else dt_index.date
    return pd.Series([d in holidays_set for d in idx_dates], index=dt_index, dtype=int, name="feriados")

def add_es_dia_de_pago(df_idx: pd.DataFrame) -> pd.Series:
    dias = [1,2,15,16,29,30,31]
    return pd.Series(df_idx.index.day.isin(dias).astype(int), index=df_idx.index, name="es_dia_de_pago")

def safe_reset_index_ts(df: pd.DataFrame, prefer: str = "ts") -> pd.DataFrame:
    name = prefer
    i = 1
    while name in df.columns:
        name = f"{prefer}_{i}"
        i += 1
    return df.reset_index(names=name)

def main(horizonte_dias: int):
    os.makedirs("public", exist_ok=True)

    # 1) Histórico principal
    dfh = smart_read_historical(DATA_FILE)
    dfh.columns = dfh.columns.str.strip()

    if TARGET_CALLS_NEW not in dfh.columns:
        for cand in ["recibidos_nacional","recibidos","total_llamadas","llamadas"]:
            if cand in dfh.columns:
                dfh = dfh.rename(columns={cand: TARGET_CALLS_NEW})
                break

    if TARGET_TMO_NEW not in dfh.columns:
        tmo_source = None
        for cand in ["tmo (segundos)","tmo_seg","tmo","tmo_general"]:
            if cand in dfh.columns:
                tmo_source = cand; break
        if tmo_source:
            dfh[TARGET_TMO_NEW] = dfh[tmo_source].apply(parse_tmo_to_seconds)

    dfh = ensure_ts(dfh)

    # 2) Merge HISTORICO_TMO
    if os.path.exists(TMO_HIST_FILE):
        df_tmo = load_historico_tmo(TMO_HIST_FILE)
        dfh = dfh.join(df_tmo, how="left")
        if "tmo_general" in dfh.columns:
            dfh[TARGET_TMO_NEW] = dfh["tmo_general"].combine_first(dfh.get(TARGET_TMO_NEW))

    # 3) Calendario
    holidays_set = load_holidays(HOLIDAYS_FILE)
    if "feriados" not in dfh.columns:
        dfh["feriados"] = mark_holidays_index(dfh.index, holidays_set).values
    dfh["feriados"] = pd.to_numeric(dfh["feriados"], errors="coerce").fillna(0).astype(int)
    if "es_dia_de_pago" not in dfh.columns:
        dfh["es_dia_de_pago"] = add_es_dia_de_pago(dfh).values

    # 4) Forzar numérico + ffill
    for c in [
        TARGET_TMO_NEW, "feriados", "es_dia_de_pago",
        "proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico",
        "q_llamadas_general","q_llamadas_comercial","q_llamadas_tecnico",
    ]:
        if c in dfh.columns:
            dfh[c] = pd.to_numeric(dfh[c], errors="coerce").ffill()

    # 5) Forecast (planner + TMO interno que luego se sobreescribe)
    df_hourly = forecast_120d(
        safe_reset_index_ts(dfh),
        horizon_days=horizonte_dias,
        holidays_set=holidays_set
    )

    # 6) OVERRIDE TMO con HISTORICO_TMO
    future_idx = df_hourly.index
    tmo_future = forecast_tmo_from_historico(
        historico_path=TMO_HIST_FILE,
        future_idx=future_idx,
        lookback_days=90,
    )

    # Diagnóstico simple
    try:
        if os.path.exists(TMO_HIST_FILE):
            _df_raw = pd.read_csv(TMO_HIST_FILE, low_memory=False)
            print(f"[TMO HIST] filas leídas: {len(_df_raw)}")
    except Exception as e:
        print("[TMO HIST] Diagnóstico lectura falló:", e)

    # Fallback si quedó 0/NaN
    vals = np.nan_to_num(tmo_future.values, nan=0.0)
    if (vals == 0).all():
        print("[TMO] Fallback activado: perfil futuro vacío o todo 0.")
        last_ts = dfh.index.max()
        df_win = dfh.loc[dfh.index >= last_ts - pd.Timedelta(days=90)].copy()

        def _to_num(s):
            return pd.to_numeric(s, errors="coerce") if s is not None else None

        gm = _to_num(df_win.get("tmo_general"))
        if gm is None or gm.dropna().empty:
            tc = _to_num(df_win.get("tmo_comercial"))
            tt = _to_num(df_win.get("tmo_tecnico"))
            qc = _to_num(df_win.get("q_llamadas_comercial"))
            qt = _to_num(df_win.get("q_llamadas_tecnico"))
            qg = _to_num(df_win.get("q_llamadas_general"))
            pc = _to_num(df_win.get("proporcion_comercial"))
            pt = _to_num(df_win.get("proporcion_tecnica"))
            gm = None
            if tc is not None and tt is not None and qc is not None and qt is not None and qg is not None:
                num = tc.astype(float)*qc.astype(float) + tt.astype(float)*qt.astype(float)
                den = qg.astype(float).replace(0, np.nan)
                gm = (num/den)
            elif tc is not None and tt is not None and pc is not None and pt is not None:
                w = (pc.astype(float).fillna(0.0) + pt.astype(float).fillna(0.0)).replace(0, np.nan)
                gm = (tc.astype(float)*pc.astype(float) + tt.astype(float)*pt.astype(float)) / w
            elif tc is not None and tt is not None:
                gm = pd.concat([tc, tt], axis=1).median(axis=1, skipna=True)

        if gm is not None and not gm.dropna().empty:
            prof = (pd.DataFrame({"tmo": gm})
                      .assign(dow=lambda x: x.index.dayofweek,
                              hour=lambda x: x.index.hour)
                      .groupby(["dow","hour"])["tmo"].median().rename("tmo_mediana").reset_index())
            mm = pd.DataFrame({"ts": future_idx})
            mm["dow"] = mm["ts"].dt.dayofweek   # <-- FIX: usar .dt
            mm["hour"] = mm["ts"].dt.hour       # <-- FIX: usar .dt
            out = mm.merge(prof, on=["dow","hour"], how="left")
            med_global = out["tmo_mediana"].median(skipna=True)
            out["tmo_mediana"] = out["tmo_mediana"].fillna(float(med_global) if np.isfinite(med_global) else 0.0)
            tmo_future = pd.Series(out["tmo_mediana"].values, index=future_idx, name="tmo_s").fillna(0.0)
        else:
            print("[TMO] Fallback sin datos: usando 0.")

    df_hourly["tmo_s"] = pd.to_numeric(tmo_future.values, errors="coerce").fillna(0.0).astype(float)

    # 7) Agentes y JSONs
    try:
        from src.inferencia.erlang import required_agents
        df_hourly["agentes_requeridos"] = required_agents(
            traffic_calls=df_hourly["calls"].astype(float).values,
            aht_seconds=df_hourly["tmo_s"].astype(float).values
        ).astype(int)
    except Exception:
        df_hourly["agentes_requeridos"] = (df_hourly["calls"] / 20).round().astype(int)

    write_hourly_json("public/prediccion_horaria.json", df_hourly,
                      calls_col="calls", tmo_col="tmo_s", agentes_col="agentes_requeridos")
    write_daily_json("public/prediccion_diaria.json", df_hourly,
                     calls_col="calls", tmo_col="tmo_s")

    # (opcional) alertas clima
    try:
        from src.alertas_clima import generar_alertas
        generar_alertas(df_hourly[["calls"]])
    except Exception as e:
        print("⚠️ Alertas clima no generadas:", e)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizonte", type=int, default=120)
    args = ap.parse_args()
    main(args.horizonte)

