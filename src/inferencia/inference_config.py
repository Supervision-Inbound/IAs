# src/inferencia/inference_config.py

# --- Constantes de Inferencia ---

# Define la ruta a los artefactos del modelo
# (Ajusta esto si tus modelos están en 'models/' o 'artifacts/')
ARTIFACTS_PATH = "models" 

# --- Nombres de Archivos de Artefactos ---
MODEL_FILENAME = "modelo_planner.keras" 
SCALER_FILENAME = "scaler_planner.pkl" 
TRAINING_COLS_FILENAME = "training_columns_planner.json" 

# --- Parámetros del Modelo LSTM ---
# Este valor DEBE coincidir con el usado durante el entrenamiento
MAX_TIMESTAMPS = 168 # (7 días * 24 horas) -> Lookback
