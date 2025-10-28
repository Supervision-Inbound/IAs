# src/main.py
import argparse
import os
import pandas as pd

from src.inferencia.inferencia_core import forecast_120d   # <-- prefijo src.
from src.inferencia.features import ensure_ts              # <-- prefijo src.

# Opcional: set de feriados (si quieres habilitarlo, coloca fechas YYYY-mm-dd)
HOLIDAYS = set()  # ejemplo: {date(2025,1,1), date(2025,5,1)}

def _read_historical():
    """
    Lee el histórico desde rutas comunes del repo/runner.
    Debe contener al menos: 'recibidos', y opcionalmente 'tmo_general',
    'feriados', 'es_dia_de_pago', y 'ts' o 'fecha'+'hora'.
    """
    candidates = [
        "historical_data.csv",
        "hosting/historical_data.csv",
        "data/historical_data.csv",
        "/mnt/data/historical_data.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p, low_memory=False)
            return df
    raise FileNotFoundError("No se encontró historical_data.csv en rutas conocidas.")

def main(horizonte_dias: int):
    df_hist = _read_historical()
    # Normaliza TS (la función interna hace el set_index)
    df_hist = ensure_ts(df_hist)

    df_hourly = forecast_120d(
        df_hist_joined=df_hist,
        horizon_days=int(horizonte_dias),
        holidays_set=HOLIDAYS if HOLIDAYS else None
    )

    # Impresión breve para logs
    print("\nPreview salida horaria:")
    print(df_hourly.head(8).to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizonte", type=int, default=120, help="Horizonte en días (por defecto 120)")
    args = parser.parse_args()
    main(args.horizonte)


