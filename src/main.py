# src/main.py
import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# TensorFlow puede imprimir bastante; opcionalmente silenciamos algunos warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=UserWarning)

# --- Imports del proyecto ---
from inferencia.inferencia_core import forecast_120d, TARGET_CALLS, TARGET_TMO, TIMEZONE
from inferencia.features import ensure_ts

# Rutas por defecto (Kaggle / repo local)
KAGGLE_INPUT_DIR = "/kaggle/input/data-ia"
DEFAULT_XLSX = "Hosting ia.xlsx"    # igual que en entrenamiento
PUBLIC_DIR = "public"

def _smart_read(path: Path) -> pd.DataFrame:
    """Lee .xlsx/.xls/.csv con autodetección simple."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    low = path.name.lower()
    if low.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    # CSV (coma o punto y coma)
    try:
        df = pd.read_csv(path, low_memory=False)
        if df.shape[1] == 1 and ";" in str(df.iloc[0, 0]):
            return pd.read_csv(path, delimiter=";", low_memory=False)
        return df
    except Exception:
        return pd.read_csv(path, delimiter=";", low_memory=False)

def _coerce_numeric(s):
    """Convierte serie a numérica (maneja comas)."""
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")

def _prepare_hosting_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres y columnas mínimas:
      - 'ts' desde ('fecha' + 'hora') o 'ts' directo.
      - TARGET_CALLS = 'recibidos_nacional' (si viene 'recibidos', renombrar).
      - TARGET_TMO   = 'tmo_general' (si viene 'TMO (segundos)' u otro alias, mapear).
      - 'feriados' (si no existe, asumir 0).
    Devuelve dataframe indexado por ts (tz=America/Santiago), ordenado.
    """
    df = df_raw.copy()
    # Normaliza cabeceras
    cols_map = {c.lower().strip(): c for c in df.columns}

    # 1) Asegurar ts
    df = ensure_ts(df)

    # 2) Calls
    if TARGET_CALLS not in df.columns:
        # renombrar si viene 'recibidos' o similares
        for cand in ["recibidos", "llamadas", "total_llamadas", "recibidos total", "recibidos_nacional"]:
            if cand in cols_map:
                df.rename(columns={cols_map[cand]: TARGET_CALLS}, inplace=True)
                break
    if TARGET_CALLS not in df.columns:
        raise ValueError(f"No encuentro la columna de llamadas '{TARGET_CALLS}' ni sus alias.")

    df[TARGET_CALLS] = _coerce_numeric(df[TARGET_CALLS]).fillna(0)

    # 3) TMO (segundos) -> tmo_general
    if TARGET_TMO not in df.columns:
        # buscar alias tipo "TMO (segundos)" / "TMO (s)" / "aht"
        for cand in ["tmo (segundos)", "tmo (s)", "aht", "tmo_seg", "tmo_general"]:
            if cand in cols_map:
                df.rename(columns={cols_map[cand]: TARGET_TMO}, inplace=True)
                break
    # si existe, normalizamos a float
    if TARGET_TMO in df.columns:
        df[TARGET_TMO] = _coerce_numeric(df[TARGET_TMO])

    # 4) Feriados
    if "feriados" not in df.columns:
        df["feriados"] = 0
    df["feriados"] = pd.to_numeric(df["feriados"], errors="coerce").fillna(0).astype(int)

    # Orden y salida
    df = df.sort_index()
    return df

def _build_holidays_set(df_hosting: pd.DataFrame) -> set:
    """
    Construye set de fechas (date) donde feriados==1 para usar en inferencia.
    """
    if "feriados" not in df_hosting.columns:
        return set()
    mask = df_hosting["feriados"].fillna(0).astype(int) == 1
    if not mask.any():
        return set()
    # Convertimos índice a date en la zona horaria del proyecto
    idx = df_hosting.index
    try:
        dates = idx.tz_convert(TIMEZONE).date
    except Exception:
        dates = idx.date
    return set(pd.Series(dates, index=idx)[mask].unique().tolist())

def main(horizonte_dias: int):
    # 0) Asegurar carpeta de salida
    Path(PUBLIC_DIR).mkdir(parents=True, exist_ok=True)

    # 1) Localiza el archivo de Hosting
    #    Preferencia Kaggle, si no, busca en ./data
    candidates = [
        Path(KAGGLE_INPUT_DIR) / DEFAULT_XLSX,
        Path("data") / DEFAULT_XLSX,
        Path("data") / "historical_data.csv",
        Path("data") / "Hosting.csv",
    ]
    src_path = None
    for p in candidates:
        if p.exists():
            src_path = p
            break
    if src_path is None:
        raise FileNotFoundError(
            f"No se encontró el histórico de Hosting. Revisé: {', '.join(str(c) for c in candidates)}"
        )
    print(f"[main] Leyendo histórico desde: {src_path}")
    df_raw = _smart_read(src_path)

    # 2) Preparar dataframe (ts, columnas estándar)
    df = _prepare_hosting_df(df_raw)

    # 3) Holidays set (desde la propia columna feriados)
    holidays_set = _build_holidays_set(df)
    print(f"[main] Feriados detectados (fechas únicas): {len(holidays_set)}")

    # 4) Ejecutar inferencia 120d (o lo que pida --horizonte)
    print(f"[main] Ejecutando forecast_120d (horizonte={horizonte_dias} días)...")
    df_hourly = forecast_120d(
        df_hist_joined=df,         # mismo dataset para llamadas y TMO
        df_hist_tmo_only=None,     # ya no usamos histórico TMO clásico
        horizon_days=int(horizonte_dias),
        holidays_set=holidays_set
    )

    # 5) Resumen consola
    print("\n[main] Resumen de la predicción (primeras 5 filas):")
    print(df_hourly.head().to_string())
    print("\n[main] Archivos generados en 'public/':")
    print(" - prediccion_horaria.json")
    print(" - prediccion_diaria.json")
    print("\n[main] Listo.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizonte", type=int, default=120, help="Días de horizonte a predecir (default: 120)")
    args = parser.parse_args()
    try:
        main(args.horizonte)
    except Exception as e:
        print(f"[main][ERROR] {e}", file=sys.stderr)
        sys.exit(1)
