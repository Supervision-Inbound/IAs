# src/main.py
import argparse
import os
import numpy as np
import pandas as pd

from src.inferencia.inferencia_core import forecast_120d
from src.inferencia.features import ensure_ts
from src.data.loader_tmo import load_historico_tmo

# Rutas (ajústalas si tu repo usa otras)
DATA_FILE = "data/historical_data.csv"         # histórico ya unido (calls + feriados + clima si aplica)
HOLIDAYS_FILE = "data/Feriados_Chilev2.csv"    # feriados
TMO_HIST_FILE = "data/HISTORICO_TMO.csv"       # histórico TMO puro (para AR)

TARGET_CALLS_NEW = "recibidos_nacional"
TARGET_TMO_NEW = "tmo_general"


def parse_tmo_to_seconds(val):
    """Permite leer TMO como 'mm:ss', 'hh:mm:ss' o float."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace(",", ".")
    # Número simple
    try:
        # si es un número como "123.4"
        if s.replace(".", "", 1).isdigit():
            return float(s)
    except Exception:
        pass
    # Formato mm:ss o hh:mm:ss
    parts = s.split(":")
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(s)
    except Exception:
        return np.nan


def load_holidays(file_path):
    """Devuelve un set de fechas (YYYY-MM-DD) feriadas."""
    if not os.path.exists(file_path):
        return set()
    dfh = pd.read_csv(file_path)
    # Detecta posibles columnas comunes
    cols = [c for c in dfh.columns if "fecha" in c.lower() or "date" in c.lower()]
    if not cols:
        return set()
    col = cols[0]
    s = pd.to_datetime(dfh[col], errors="coerce").dt.date
    return set(d for d in s.dropna().tolist())


def main(horizonte_dias: int = 120):
    # 1) Cargar histórico principal
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"No se encontró {DATA_FILE}")

    dfh = pd.read_csv(DATA_FILE, low_memory=False)

    # --- Normalización mínima de campos de tiempo ANTES de ensure_ts ---
    cols_norm = {c.lower().strip(): c for c in dfh.columns}

    # Caso 1: hay una columna tipo datetime/timestamp/fechahora
    for alias in ("ts", "datetime", "datatime", "timestamp", "fecha_hora", "fechahora", "fecha y hora"):
        if alias in cols_norm:
            if alias != "ts":
                dfh = dfh.rename(columns={cols_norm[alias]: "ts"})
            break
    else:
        # Caso 2: vienen separadas 'FECHA' y 'HORA' (con cualquier capitalización)
        fecha_like = next((c for c in dfh.columns if "fecha" in c.lower()), None)
        hora_like  = next((c for c in dfh.columns if "hora"  in c.lower()), None)
        if fecha_like and hora_like and "ts" not in dfh.columns:
            dfh["ts"] = pd.to_datetime(
                dfh[fecha_like].astype(str) + " " + dfh[hora_like].astype(str),
                errors="coerce", dayfirst=True
            )

    # crea 'ts' y ordena (función robusta en features.py)
    dfh = ensure_ts(dfh)  # devuelve con índice 'ts'
    dfh = dfh.reset_index()

    # 2) Normalizar nombres y TMO si existiera
    if TARGET_CALLS_NEW not in dfh.columns:
        if "recibidos" in dfh.columns:
            dfh = dfh.rename(columns={"recibidos": TARGET_CALLS_NEW})
        else:
            raise ValueError(f"No se encontró columna de llamadas '{TARGET_CALLS_NEW}' ni 'recibidos'.")

    if TARGET_TMO_NEW in dfh.columns:
        dfh[TARGET_TMO_NEW] = dfh[TARGET_TMO_NEW].apply(parse_tmo_to_seconds)

    # 3) Cargar feriados
    holidays_set = load_holidays(HOLIDAYS_FILE)

    # 4) Cargar HISTÓRICO TMO puro (para AR de TMO)
    df_tmo_hist_only = None
    if os.path.exists(TMO_HIST_FILE):
        # loader propio del repo – asegura schema correcto y ts index
        df_tmo = load_historico_tmo(TMO_HIST_FILE)  # debe retornar index=ts y columnas tmo/proporciones
        # parsear a segundos por robustez (si tu loader ya lo hace, esto no afecta)
        if "tmo_general" in df_tmo.columns:
            df_tmo["tmo_general"] = df_tmo["tmo_general"].apply(parse_tmo_to_seconds)
        df_tmo_hist_only = df_tmo.copy().reset_index()

        # merge no destructivo sobre dfh (para completar huecos)
        dfh = dfh.set_index("ts")
        dfh = dfh.join(df_tmo, how="left")
        if TARGET_TMO_NEW in dfh.columns and "tmo_general" in dfh.columns:
            dfh[TARGET_TMO_NEW] = dfh[TARGET_TMO_NEW].combine_first(dfh["tmo_general"])
        dfh = dfh.reset_index()

    # 5) Ejecutar inferencia (con AR de TMO) – SIN tocar la lógica de llamadas
    df_hourly = forecast_120d(
        df_hist_joined=dfh,                              # con ts en columna
        df_hist_tmo_only=df_tmo_hist_only,               # TMO puro (para lags AR)
        horizon_days=horizonte_dias,
        holidays_set=holidays_set
    )

    # 6) Alertas de clima (opcional)
    try:
        from src.inferencia.alertas_clima import generar_alertas
        generar_alertas(df_hourly[["calls"]])
    except Exception as e:
        print(f"[WARN] Alertas de clima no generadas: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizonte", type=int, default=120)
    args = ap.parse_args()
    main(args.horizonte)
