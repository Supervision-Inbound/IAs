# src/main.py
import argparse
import os
import numpy as np
import pandas as pd

from src.inferencia.inferencia_core import forecast_120d
from src.inferencia.features import ensure_ts
from src.data.loader_tmo import load_historico_tmo

DATA_FILE = "data/historical_data.csv"
HOLIDAYS_FILE = "data/Feriados_Chilev2.csv"

# Nombre estándar de llamadas y TMO dentro del flujo/modelos
TARGET_CALLS = "recibidos_nacional"
TARGET_TMO = "tmo_general"


def read_csv_smart(path: str) -> pd.DataFrame:
    """Lee CSV probando ',', luego ';' si detecta una sola columna con ';' en la primera fila."""
    df = pd.read_csv(path, low_memory=False)
    if df.shape[1] == 1:
        first = str(df.iloc[0, 0])
        if ';' in first:
            df = pd.read_csv(path, delimiter=';', low_memory=False)
    return df


def parse_tmo_to_seconds(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace(",", ".")
    try:
        if s.replace(".", "", 1).isdigit():
            return float(s)
    except Exception:
        pass
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
    if not os.path.exists(file_path):
        return set()
    dfh = read_csv_smart(file_path)
    cols = [c for c in dfh.columns if "fecha" in c.lower() or "date" in c.lower()]
    if not cols:
        return set()
    col = cols[0]
    s = pd.to_datetime(dfh[col], errors="coerce").dt.date
    return set(d for d in s.dropna().tolist())


def normalize_time_columns(dfh: pd.DataFrame) -> pd.DataFrame:
    cols_norm = {c.lower().strip(): c for c in dfh.columns}
    for alias in ("ts", "datetime", "datatime", "timestamp", "fecha_hora", "fechahora", "fecha y hora"):
        if alias in cols_norm:
            if alias != "ts":
                dfh = dfh.rename(columns={cols_norm[alias]: "ts"})
            return dfh

    fecha_like = next((c for c in dfh.columns if "fecha" in c.lower()), None)
    hora_like  = next((c for c in dfh.columns if "hora"  in c.lower()), None)
    # Deben ser columnas distintas
    if fecha_like and hora_like and fecha_like != hora_like and "ts" not in dfh.columns:
        dfh["ts"] = pd.to_datetime(
            dfh[fecha_like].astype(str) + " " + dfh[hora_like].astype(str),
            errors="coerce", dayfirst=True
        )
    return dfh


def map_calls_column(dfh: pd.DataFrame) -> pd.DataFrame:
    if TARGET_CALLS in dfh.columns:
        return dfh
    aliases = [
        "recibidos", "calls", "llamadas", "volumen", "trafico", "tráfico",
        "q_llamadas_general", "q_llamadas", "demanda", "demand", "traffic"
    ]
    lower_map = {c.lower(): c for c in dfh.columns}
    for a in aliases:
        if a in lower_map:
            return dfh.rename(columns={lower_map[a]: TARGET_CALLS})
    raise ValueError(
        f"No se encontró columna de llamadas '{TARGET_CALLS}' ni alias comunes. "
        f"Columnas disponibles: {list(dfh.columns)}"
    )


def resolve_tmo_path() -> str:
    """
    Intenta resolver la ruta del archivo TMO histórico.
    Prioridad:
      1) Variable de entorno TMO_FILE
      2) data/HISTORICO_TMO.csv
      3) data/TMO_HISTORICO.csv
    """
    candidates = []
    env_override = os.environ.get("TMO_FILE")
    if env_override:
        candidates.append(env_override)
    candidates += [
        "data/HISTORICO_TMO.csv",
        "data/TMO_HISTORICO.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "No se encontró archivo histórico de TMO. "
        f"Probé estas rutas: {candidates}. "
        "Puedes definir TMO_FILE en variables de entorno apuntando al archivo correcto."
    )


def main(horizonte_dias: int = 120):
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"No se encontró {DATA_FILE}")

    # LECTURA ROBUSTA (coma/semicolón)
    dfh = read_csv_smart(DATA_FILE)
    dfh = normalize_time_columns(dfh)
    dfh = ensure_ts(dfh).reset_index()

    # Planner de llamadas
    dfh = map_calls_column(dfh)

    # Si hubiera una columna TMO aquí, no la usamos como fuente; pero la normalizamos por si aparece
    if TARGET_TMO in dfh.columns:
        dfh[TARGET_TMO] = dfh[TARGET_TMO].apply(parse_tmo_to_seconds)

    holidays_set = load_holidays(HOLIDAYS_FILE)

    # Resolver ruta del TMO histórico (flexible)
    tmo_hist_path = resolve_tmo_path()

    # Cargar TMO puro (mismo esquema del entrenamiento)
    df_tmo = load_historico_tmo(tmo_hist_path)
    if TARGET_TMO in df_tmo.columns:
        df_tmo[TARGET_TMO] = df_tmo[TARGET_TMO].apply(parse_tmo_to_seconds)
    df_tmo_hist_only = df_tmo.copy().reset_index()

    # NO fusionamos TMO al histórico principal; el core usa solo df_tmo_hist_only
    df_hourly = forecast_120d(
        df_hist_joined=dfh,
        df_hist_tmo_only=df_tmo_hist_only,
        horizon_days=horizonte_dias,
        holidays_set=holidays_set
    )

    # Alertas (opcional)
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

