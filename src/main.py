# src/main.py
import argparse, os, json
import numpy as np
import pandas as pd

from src.inferencia.inferencia_core import forecast_120d
from src.inferencia.features import ensure_ts
from src.data.loader_tmo import load_historico_tmo  # ← usa tu loader real de TMO

DATA_FILE = "data/historical_data.csv"
HOLIDAYS_FILE = "data/Feriados_Chilev2.csv"
TMO_HIST_FILE = "data/TMO_HISTORICO.csv"   # ← nombre real en tu repo
TZ = "America/Santiago"

TARGET_CALLS_NEW = "recibidos_nacional"
TARGET_TMO_NEW = "tmo_general"

# ---------------- utilidades de lectura ----------------
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
        if len(parts) == 2: return float(parts[0])*60 + float(parts[1])
        return float(s)
    except:
        return np.nan

def load_holidays(csv_path: str) -> set:
    if not os.path.exists(csv_path): return set()
    fer = pd.read_csv(csv_path)
    cols_map = {c.lower().strip(): c for c in fer.columns}
    fecha_col = None
    for cand in ["fecha","date","dia","día"]:
        if cand in cols_map:
            fecha_col = cols_map[cand]; break
    if not fecha_col: return set()
    fechas = pd.to_datetime(fer[fecha_col].astype(str), dayfirst=True, errors="coerce").dropna().dt.date
    return set(fechas)

def mark_holidays_index(idx, holidays_set: set) -> pd.Series:
    tz = getattr(idx, "tz", None)
    idx_dates = idx.tz_convert(TZ).date if tz is not None else idx.date
    return pd.Series([d in holidays_set for d in idx_dates], index=idx, dtype=int, name="feriados")

def add_es_dia_de_pago(df_idx: pd.DataFrame) -> pd.Series:
    dias = [1,2,15,16,29,30,31]
    return pd.Series(df_idx.index.day.isin(dias).astype(int), index=df_idx.index, name="es_dia_de_pago")

# ---------------- entrypoint ----------------
def main(horizonte_dias: int):
    os.makedirs("public", exist_ok=True)

    # 1) Leer histórico de llamadas (NO mezclar TMO aquí)
    dfh = smart_read_historical(DATA_FILE)
    dfh.columns = dfh.columns.str.strip()

    if TARGET_CALLS_NEW not in dfh.columns:
        for cand in ["recibidos_nacional","recibidos","total_llamadas","llamadas"]:
            if cand in dfh.columns:
                dfh = dfh.rename(columns={cand: TARGET_CALLS_NEW}); break

    # Si viene algún TMO de “historical_data”, lo dejamos como referencia, pero no lo forzamos
    if TARGET_TMO_NEW not in dfh.columns:
        for cand in ["tmo (segundos)","tmo_seg","tmo","aht"]:
            if cand in dfh.columns:
                dfh[TARGET_TMO_NEW] = dfh[cand].apply(parse_tmo_to_seconds); break

    dfh = ensure_ts(dfh)

    # 2) Cargar TMO_HISTORICO por separado (base del TMO)
    df_tmo = None
    if os.path.exists(TMO_HIST_FILE):
        df_tmo = load_historico_tmo(TMO_HIST_FILE)  # ← devuelve index ts y columnas TMO/mezcla
        # (no lo mezclamos a dfh: se pasa aparte a forecast_120d)
        # Dump de debug mínimo
        try:
            with open("public/debug_tmo_hist.json","w",encoding="utf-8") as f:
                json.dump({
                    "rows": int(df_tmo.shape[0]),
                    "cols": list(df_tmo.columns),
                    "first_ts": str(df_tmo.index.min()) if df_tmo.shape[0] else None,
                    "last_ts": str(df_tmo.index.max()) if df_tmo.shape[0] else None
                }, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    else:
        # Fallback: dataframe vacío para que forecast use perfiles neutros
        df_tmo = pd.DataFrame(index=dfh.index.unique().sort_values())

    # 3) Calendario (feriados y día de pago SOLO en llamadas)
    holidays_set = load_holidays(HOLIDAYS_FILE)
    if "feriados" not in dfh.columns:
        dfh["feriados"] = mark_holidays_index(dfh.index, holidays_set).values
    dfh["feriados"] = pd.to_numeric(dfh["feriados"], errors="coerce").fillna(0).astype(int)
    if "es_dia_de_pago" not in dfh.columns:
        dfh["es_dia_de_pago"] = add_es_dia_de_pago(dfh).values

    # FFill columnas clave en llamadas
    for c in [TARGET_TMO_NEW, "feriados", "es_dia_de_pago"]:
        if c in dfh.columns:
            dfh[c] = dfh[c].ffill()

    # 4) Forecast: PASAR AMBOS DATAFRAMES, sin merge de TMO dentro de dfh
    df_hourly = forecast_120d(
        df_hist_calls=dfh,
        df_tmo_hist=df_tmo,
        horizon_days=horizonte_dias,
        holidays_set=holidays_set
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizonte", type=int, default=120)
    args = ap.parse_args()
    main(args.horizonte)
