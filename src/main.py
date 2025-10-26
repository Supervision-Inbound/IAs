# src/main.py
import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Silenciar verbosidad TF (opcional)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=UserWarning)

# ---------- Imports robustos (paquete/standalone) ----------
def _import_inferencia():
    try:
        from .inferencia.inferencia_core import forecast_120d, TARGET_CALLS, TARGET_TMO, TIMEZONE
        from .inferencia.features import ensure_ts
        return forecast_120d, TARGET_CALLS, TARGET_TMO, TIMEZONE, ensure_ts
    except Exception:
        here = Path(__file__).resolve().parent
        sys.path.append(str(here))
        from inferencia.inferencia_core import forecast_120d, TARGET_CALLS, TARGET_TMO, TIMEZONE
        from inferencia.features import ensure_ts
        return forecast_120d, TARGET_CALLS, TARGET_TMO, TIMEZONE, ensure_ts

forecast_120d, TARGET_CALLS, TARGET_TMO, TIMEZONE, ensure_ts = _import_inferencia()

# ---------- Rutas por defecto ----------
KAGGLE_INPUT_DIR = "/kaggle/input/data-ia"
DEFAULT_XLSX = "Hosting ia.xlsx"
PUBLIC_DIR = "public"

# ---------- Utilidades ----------
def _smart_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    low = path.name.lower()
    if low.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    try:
        df = pd.read_csv(path, low_memory=False)
        if df.shape[1] == 1 and ";" in str(df.iloc[0, 0]):
            return pd.read_csv(path, delimiter=";", low_memory=False)
        return df
    except Exception:
        return pd.read_csv(path, delimiter=";", low_memory=False)

def _coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")

def _canonicalize_ts_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura que 'ts' sea SOLO índice (no columna). Evita el error:
    "'ts' is both an index level and a column label".
    """
    # Si el índice ya es 'ts' y además existe la columna 'ts', la eliminamos.
    if "ts" in df.columns and (df.index.name == "ts" or "ts" in (df.index.names or [])):
        df = df.drop(columns=["ts"])
    # Si AÚN no es índice, lo establecemos.
    if df.index.name != "ts" and "ts" in df.columns:
        df = df.set_index("ts")
    # Orden temporal ascendente
    try:
        df = df.sort_index()
    except Exception:
        pass
    return df

def _prepare_hosting_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza:
      - 'ts' (índice con zona horaria que maneje ensure_ts).
      - TARGET_CALLS = 'recibidos_nacional' (si viene 'recibidos', renombrar).
      - TARGET_TMO   = 'tmo_general' (si viene 'TMO (segundos)' u otro alias, mapear).
      - 'feriados' (si no existe, 0).
    """
    df = df_raw.copy()
    cols_map = {c.lower().strip(): c for c in df.columns}

    # 1) ts normalizado (la función de features)
    df = ensure_ts(df)
    # 2) Canonizar 'ts' solo como índice
    df = _canonicalize_ts_index(df)

    # 3) llamadas
    if TARGET_CALLS not in df.columns:
        for cand in ["recibidos", "llamadas", "total_llamadas", "recibidos total", "recibidos_nacional"]:
            if cand in cols_map:
                df.rename(columns={cols_map[cand]: TARGET_CALLS}, inplace=True)
                break
    if TARGET_CALLS not in df.columns:
        raise ValueError(f"No encuentro la columna de llamadas '{TARGET_CALLS}' ni sus alias.")
    df[TARGET_CALLS] = _coerce_numeric(df[TARGET_CALLS]).fillna(0)

    # 4) TMO (segundos) -> tmo_general (si existe)
    if TARGET_TMO not in df.columns:
        for cand in ["tmo (segundos)", "tmo (s)", "aht", "tmo_seg", "tmo_general"]:
            if cand in cols_map:
                df.rename(columns={cols_map[cand]: TARGET_TMO}, inplace=True)
                break
    if TARGET_TMO in df.columns:
        df[TARGET_TMO] = _coerce_numeric(df[TARGET_TMO])

    # 5) feriados
    if "feriados" not in df.columns:
        df["feriados"] = 0
    df["feriados"] = pd.to_numeric(df["feriados"], errors="coerce").fillna(0).astype(int)

    return df.sort_index()

def _build_holidays_set(df_hosting: pd.DataFrame) -> set:
    if "feriados" not in df_hosting.columns:
        return set()
    mask = df_hosting["feriados"].fillna(0).astype(int) == 1
    if not mask.any():
        return set()
    idx = df_hosting.index
    try:
        dates = idx.tz_convert(TIMEZONE).date
    except Exception:
        dates = idx.date
    return set(pd.Series(dates, index=idx)[mask].unique().tolist())

# ---------- Main ----------
def main(horizonte_dias: int):
    Path(PUBLIC_DIR).mkdir(parents=True, exist_ok=True)

    # Localiza histórico de Hosting
    candidates = [
        Path(KAGGLE_INPUT_DIR) / DEFAULT_XLSX,
        Path("data") / DEFAULT_XLSX,
        Path("data") / "historical_data.csv",
        Path("data") / "Hosting.csv",
    ]
    src_path = next((p for p in candidates if p.exists()), None)
    if src_path is None:
        raise FileNotFoundError(
            f"No se encontró el histórico de Hosting. Revisé: {', '.join(str(c) for c in candidates)}"
        )
    print(f"[main] Leyendo histórico desde: {src_path}")
    df_raw = _smart_read(src_path)

    # Preparar dataframe
    df = _prepare_hosting_df(df_raw)

    # Holidays
    holidays_set = _build_holidays_set(df)
    print(f"[main] Feriados detectados (fechas únicas): {len(holidays_set)}")

    # Inferencia
    print(f"[main] Ejecutando forecast_120d (horizonte={horizonte_dias} días)...")
    df_hourly = forecast_120d(
        df_hist_joined=df,
        df_hist_tmo_only=None,
        horizon_days=int(horizonte_dias),
        holidays_set=holidays_set
    )

    # Resumen
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
