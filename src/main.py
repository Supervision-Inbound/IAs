import os
import argparse
import pandas as pd
from src.inferencia.inferencia_core import forecast_120d
from src.inferencia.features import ensure_ts

PUBLIC_DIR = "public"
HOLIDAYS_FILE = "data/feriados.csv"
DATA_FILE = "data/historical_data.csv"

# Funciones de carga (Asumiendo que son correctas)

def main(horizonte: int):
    
    # 💥 MODIFICACIÓN CRÍTICA: Crear el directorio de salida si no existe.
    if not os.path.exists(PUBLIC_DIR):
        print(f"Creando directorio de salida: {PUBLIC_DIR}")
        # os.makedirs crea la carpeta y todos los directorios intermedios
        os.makedirs(PUBLIC_DIR, exist_ok=True)
    
    # 2. Cargar y preparar datos históricos
    try:
        print(f"Cargando datos desde: {DATA_FILE}")
        df_hist = pd.read_csv(DATA_FILE)
        df_hist = ensure_ts(df_hist)
    except Exception as e:
        print(f"Error al cargar datos históricos: {e}")
        return

    # 3. Cargar feriados
    try:
        df_feriados = pd.read_csv(HOLIDAYS_FILE)
        feriados_set = set(pd.to_datetime(df_feriados['date'], errors='coerce').dt.date.dropna())
    except Exception:
        print(f"Advertencia: No se pudo cargar el archivo de feriados en {HOLIDAYS_FILE}. Continuando sin feriados.")
        feriados_set = set()

    # 4. Ejecutar la inferencia
    df_hourly = forecast_120d(
        df_hist_joined=df_hist,
        horizon_days=horizonte,
        holidays_set=feriados_set
    )
    
    if not df_hourly.empty:
        print(f"Éxito: Predicción para {horizonte} días completada. Verifique los archivos en ./{PUBLIC_DIR}/")
    else:
        print("Advertencia: El DataFrame de predicción está vacío. Verifique la carga de modelos.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecuta la inferencia del modelo LSTM.")
    parser.add_argument("--horizonte", type=int, default=7, help="Número de días a predecir.")
    args = parser.parse_args()
    
    main(args.horizonte)
