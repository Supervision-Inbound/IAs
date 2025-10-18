# src/main.py
import argparse, os, json
import numpy as np
import pandas as pd

from src.inferencia.inferencia_core import forecast_120d
from src.inferencia.features import ensure_ts
from src.data.loader_tmo import load_historico_tmo

DATA_FILE = "data/historical_data.csv"
HOLIDAYS_FILE = "data/Feriados_Chilev2.csv"
TMO_HIST_FILE = "data/TMO_HISTORICO.csv"   # <- nombre real en tu repo
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

    # 1) Leer histórico base
    dfh = smart_read_historical(DATA_FILE)
    dfh.columns = dfh.columns.str.strip()

    if TARGET_CALLS_NEW not in dfh.columns:
        for cand in ["recibidos_nacional","recibidos","total_llamadas","llamadas"]:
            if cand in dfh.columns:
                dfh = dfh.rename(columns={cand: TARGET_CALLS_NEW}); break

    if TARGET_TMO_NEW not in dfh.columns:
        for cand in ["tmo (segundos)","tmo_seg","tmo","aht"]:
            if cand in dfh.columns:
                dfh[TARGET_TMO_NEW] = dfh[cand].apply(parse_tmo_to_seconds); break

    dfh = ensure_ts(dfh)

    # 2) Merge con TMO_HISTORICO.csv (por hora) sin colisión de columnas
    used_tmo_hist = False
    if os.path.exists(TMO_HIST_FILE):
        df_tmo = load_historico_tmo(TMO_HIST_FILE)  # index ts
        try:
            df_tmo.index = df_tmo.index.tz_convert(TZ)
        except Exception:
            pass

        # columnas posibles que trae el loader
        tmo_cols = [
            "q_llamadas_general","q_llamadas_comercial","q_llamadas_tecnico",
            "proporcion_comercial","proporcion_tecnica",
            "tmo_comercial","tmo_tecnico","tmo_general"
        ]

        # asignación columna a columna para evitar "columns overlap" en join
        for c in tmo_cols:
            if c in df_tmo.columns:
                if c in dfh.columns:
                    # preferimos lo que ya trae dfh y completamos huecos con df_tmo
                    dfh[c] = dfh[c].combine_first(df_tmo[c])
                else:
                    dfh[c] = df_tmo[c]

        used_tmo_hist = True

        # Debug dumps
        debug = {
            "used_tmo_hist": used_tmo_hist,
            "tmo_hist_rows": int(df_tmo.shape[0]),
            "dfh_rows": int(dfh.shape[0]),
            "new_cols": [c for c in tmo_cols if c in dfh.columns],
            "last_non_na_tmo_general": float(dfh["tmo_general"].dropna().iloc[-1]) if "tmo_general" in dfh.columns and dfh["tmo_general"].notna().any() else None
        }
        with open("public/debug_tmo_merge.json","w",encoding="utf-8") as f:
            json.dump(debug, f, ensure_ascii=False, indent=2)
        dfh.tail(48).to_csv("public/debug_tmo_tail48.csv")

    # 3) Calendario
    holidays_set = load_holidays(HOLIDAYS_FILE)
    if "feriados" not in dfh.columns:
        dfh["feriados"] = mark_holidays_index(dfh.index, holidays_set).values
    dfh["feriados"] = pd.to_numeric(dfh["feriados"], errors="coerce").fillna(0).astype(int)
    if "es_dia_de_pago" not in dfh.columns:
        dfh["es_dia_de_pago"] = add_es_dia_de_pago(dfh).values

    # 4) FFill columnas clave
    for c in [TARGET_TMO_NEW, "feriados", "es_dia_de_pago",
              "proporcion_comercial","proporcion_tecnica","tmo_comercial","tmo_tecnico"]:
        if c in dfh.columns:
            dfh[c] = dfh[c].ffill()

    # 5) Forecast
    df_hourly = forecast_120d(
        dfh.reset_index(),
        horizon_days=horizonte_dias,
        holidays_set=holidays_set
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizonte", type=int, default=120)
    args = ap.parse_args()
    main(args.horizonte)
