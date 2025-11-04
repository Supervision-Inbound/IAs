# src/inferencia/inference_config.py

# --- Constantes de Inferencia ---

# Define la ruta a los artefactos del modelo
# (Ajusta esto si tus modelos están en 'models/' o 'artifacts/')
ARTIFACTS_PATH = "models" 

# --- Nombres de Archivos de Artefactos ---
MODEL_FILENAME = "modelo_planner.keras" # Ajusta si el nombre es diferente
SCALER_FILENAME = "scaler_planner.pkl" # Ajusta si el nombre es diferente
TRAINING_COLS_FILENAME = "training_columns_planner.json" # Ajusta si el nombre es diferente

# --- Parámetros del Modelo LSTM ---
# Estos valores DEBEN coincidir con los usados durante el entrenamiento
MAX_TIMESTAMPS = 168 # (7 días * 24 horas) -> Lookback
