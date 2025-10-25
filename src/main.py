# src/main.py
import argparse
import os
import numpy as np
import pandas as pd

from src.inferencia.inferencia_core import forecast_120d
from src.inferencia.features import ensure_ts
from src.data.loader_tmo import load_historico_tmo

# Rutas (ajústalas si tu repo usa otras)
DATA_FILE = "data/historical_data.csv"         # histórico principal (clima/feriados/otros + llamadas si existen)
HOLIDAYS_FILE = "data/Feriados_Chilev2.csv"    # feriados
TMO_HIST_FILE = "data/HISTORICO_TMO.csv"       # histórico TMO puro (para AR de TMO)

TARGET_CALLS = "recibidos_nacional"            # nombre estándar dentro del flujo
TARGET_TMO = "tmo_general"


def parse_tmo_to_seconds(val):
    """Permite leer TMO como 'mm:ss', 'hh:mm:ss' o float."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace(",", ".")
    # Número simple
    try:
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
    cols = [c for c in dfh.columns if "fecha" in c.lower() or "date" in c.lower()]
    if not cols:
        return set()
    col = cols[0]
    s = pd.to_datetime(dfh[col], errors="coerce").dt.date
    return set(d for d in s.dropna().tolist())


def normalize_time_columns(dfh: pd.DataFrame) -> pd.DataFrame:
    """Genera/normaliza 'ts' antes de ensure_ts (soporta muchos alias)."""
    cols_norm = {c.lower().strip(): c for c in dfh.columns}
    # Caso 1: una sola columna datetime/timestamp/fechahora
    for alias in ("ts", "datetime", "datatime", "timestamp", "fecha_hora", "fechahora", "fecha y hora"):
        if alias in cols_norm:
            if alias != "ts":
                dfh = dfh.rename(columns={cols_norm[alias]: "ts"})
            return dfh
    # Caso 2: 'fecha' + 'hora'
    fecha_like = next((c for c in dfh.columns if "fecha" in c.lower()), None)
    hora_like = next((c for c in dfh.columns if "hora" in c.lower()), None)
    if fecha_like and hora_like and "ts" not in dfh.columns:
        dfh["ts"] = pd.to_datetime(
            dfh[fecha_like].astype(str) + " " + dfh[hora_like].astype(str),
            errors="coerce", dayfirst=True
        )
    return dfh


def map_calls_column(dfh: pd.DataFrame) -> pd.DataFrame:
    """Mapea cualquier alias razonable de llamadas a TARGET_CALLS."""
    if TARGET_CALLS in dfh.columns:
        return dfh
    # alias posibles
    aliases = [
        "recibidos", "calls", "llamadas", "volumen", "trafico", "tráfico",
        "q_llamadas_general", "q_llamadas", "demanda", "demand", "traffic"
    ]
    lower_map = {c.lower(): c for c in dfh.columns}
    for a in aliases:
        if a in lower_map:
            return dfh.rename(columns={lower_map[a]: TARGET_CALLS})
    # si no hubo match, error claro
    raise ValueError(
        f"No se encontró columna de llamadas '{TARGET_CALLS}' ni alias comunes. "
        f"Columnas disponibles: {list(dfh.columns)}"
    )


def main(horizonte_dias: int = 120):
    # 1) Cargar histórico principal (puede o no traer llamadas)
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"No se encontró {DATA_FILE}")

    dfh = pd.read_csv(DATA_FILE, low_memory=False)
    dfh = normalize_time_columns(dfh)
    dfh = ensure_ts(dfh).reset_index()  # asegura 'ts'

    # 2) Mapear columna de llamadas (sin esto no podemos correr el planner ni erlang)
    dfh = map_calls_column(dfh)

    # 3) Normalizar TMO si existiera accidentalmente en este CSV (no lo usaremos como fuente)
    if TARGET_TMO in dfh.columns:
        dfh[TARGET_TMO] = dfh[TARGET_TMO].apply(parse_tmo_to_seconds)

    # 4) Cargar feriados
    holidays_set = load_holidays(HOLIDAYS_FILE)

    # 5) Cargar HISTÓRICO TMO puro (única fuente de lags/MA de TMO)
    if not os.path.exists(TMO_HIST_FILE):
        raise FileNotFoundError(
            f"No se encontró {TMO_HIST_FILE}. "
            "El TMO autoregresivo requiere el histórico TMO puro (mismo esquema que entrenamiento)."
        )

    df_tmo = load_historico_tmo(TMO_HIST_FILE)  # index=ts, columnas tmo/proporciones
    if TARGET_TMO in df_tmo.columns:
        df_tmo[TARGET_TMO] = df_tmo[TARGET_TMO].apply(parse_tmo_to_seconds)
    df_tmo_hist_only = df_tmo.copy().reset_index()  # <- pasamos TMO puro al core

    # (IMPORTANTE) NO fusionamos TMO al histórico principal: el core leerá TMO solo desde df_tmo_hist_only

    # 6) Ejecutar inferencia (TMO AR no se alimenta de histórico de llamadas; solo usa llamadas como exógena)
    df_hourly = forecast_120d(
        df_hist_joined=dfh,                       # histórico (con 'ts', llamadas y exógenas)
        df_hist_tmo_only=df_tmo_hist_only,        # TMO puro (para lags/MA y ffill de base)
        horizon_days=horizonte_dias,
        holidays_set=holidays_set
    )

    # 7) Alertas de clima (opcional)
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
