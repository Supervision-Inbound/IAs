# src/main.py
import argparse, pandas as pd, os
from inferencia.inferencia_core import forecast_120d
from inferencia.features import ensure_ts
from inferencia.alertas_clima import generar_alertas

def main(horizonte_dias: int):
    os.makedirs("public", exist_ok=True)
    # 1) Cargar histórico mínimo (ts/fecha+hora, recibidos_nacional, feriados opcional)
    dfh = pd.read_csv("data/historical_data.csv")
    dfh = ensure_ts(dfh)
    # 2) Planner + TMO + Erlang → predicción horaria (retorna DF por hora)
    df_hourly = forecast_120d(dfh.reset_index(), horizon_days=horizonte_dias)
    # 3) Alertas climáticas (usa planner para calcular uplift adicional por comuna)
    generar_alertas(df_hourly[["calls"]])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizonte", type=int, default=120)
    args = ap.parse_args()
    main(args.horizonte)

