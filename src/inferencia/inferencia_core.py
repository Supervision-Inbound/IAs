# -*- coding: utf-8 -*-
import datetime
import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf

# Importación relativa (evita ModuleNotFoundError)
from . import features 

# ----------------------------------------------------------------------
# Constantes (Las que no dependen de config)
# ----------------------------------------------------------------------
TIMEZONE = features.TIMEZONE

# ----------------------------------------------------------------------
# Carga de modelos y datos
# ----------------------------------------------------------------------
def load_all_artifacts(artifacts_path: str):
    """Carga el modelo Keras, el scaler y las columnas de entrenamiento."""
    # Importación local (evita Circular Import)
    from . import inference_config as config 
    
    print("Cargando artefactos LSTM...")
    model_path = os.path.join(artifacts_path, config.MODEL_FILENAME)
    scaler_path = os.path.join(artifacts_path, config.SCALER_FILENAME)
    cols_path = os.path.join(artifacts_path, config.TRAINING_COLS_FILENAME)

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"Error al cargar el modelo Keras desde {model_path}: {e}")
        return None, None, None

    try:
        import joblib
        scaler = joblib.load(scaler_path)
    except Exception as e:
        print(f"Error al cargar el scaler desde {scaler_path}: {e}")
        return model, None, None

    try:
        with open(cols_path, 'r') as f:
            import json
            training_cols = json.load(f)
    except Exception as e:
        print(f"Error al cargar las columnas de entrenamiento (json) desde {cols_path}: {e}")
        return model, scaler, None

    return model, scaler, training_cols

# ----------------------------------------------------------------------
# Preparación de datos (Input Sequence)
# ----------------------------------------------------------------------
def prepare_input_sequence(df_historic_raw: pd.DataFrame, training_cols: list):
    """
    Prepara la secuencia histórica y la escala para la inferencia.
    """
    # Importación local
    from . import inference_config as config 
    MAX_TIMESTAMPS = config.MAX_TIMESTAMPS
    
    dfh = features.ensure_ts(df_historic_raw)
    dfh = dfh.sort_index()

    if len(dfh) < MAX_TIMESTAMPS:
        raise ValueError(f"Datos históricos insuficientes. Se requieren al menos {MAX_TIMESTAMPS} puntos.")

    df_input = dfh.tail(MAX_TIMESTAMPS).copy()
    
    df_input = features.add_time_parts(df_input)
    df_input = features.add_lags_mas(df_input, target_col="target")
    
    X_input = features.dummies_and_reindex(df_input, training_cols)
    X_input = X_input[training_cols]
    
    return df_input, X_input.values

# ----------------------------------------------------------------------
# Predicción iterativa (Step-by-Step)
# ----------------------------------------------------------------------
def iterative_forecast(
    df_historic_cleaned: pd.DataFrame, 
    X_input_sequence: np.ndarray, 
    model: tf.keras.Model, 
    scaler: object, 
    training_cols: list, 
    horizonte_pasos: int,
    holidays_set: set 
) -> pd.DataFrame:
    """
    Realiza la predicción iterativa (paso a paso) de la serie temporal.
    """
    # Importación local
    from . import inference_config as config 
    MAX_TIMESTAMPS = config.MAX_TIMESTAMPS
    
    HORIZONTE_PRED_DIAS = horizonte_pasos // 24 

    print(f"Iniciando predicción iterativa (LSTM) para {horizonte_pasos} pasos...")
    
    # 1. Preparación
    last_known_target = df_historic_cleaned["target"].iloc[-1]
    # last_known_ts ya es tz-aware (America/Santiago) gracias a ensure_ts
    last_known_ts = df_historic_cleaned.index[-1] 

    X_input_scaled = scaler.transform(X_input_sequence)
    current_sequence = np.expand_dims(X_input_scaled, axis=0)

    # 2. Inicialización
    # future_timestamps será una lista de timestamps tz-aware (America/Santiago)
    future_timestamps = [last_known_ts + datetime.timedelta(hours=h) for h in range(1, horizonte_pasos + 1)]
    dfp = pd.DataFrame(index=future_timestamps, columns=["target_pred"])
    
    dfw = df_historic_cleaned.copy()
    dfw["target"] = dfw["target"].astype(float)
    
    # 3. Bucle de predicción
    for step in range(horizonte_pasos):
        
        day = (step // 24) + 1
        if step % 24 == 0:
            print(f"Prediciendo Día {day}/{HORIZONTE_PRED_DIAS}...")

        # a. Predicción
        y_scaled = model.predict(current_sequence, verbose=0)[0, 0]
        
        dummy_features = np.zeros((1, len(training_cols)))
        dummy_features[0, 0] = y_scaled 
        y_pred_unscaled = scaler.inverse_transform(dummy_features)[0, 0]
        
        # b. Almacenar predicción
        future_ts = future_timestamps[step]
        dfp.loc[future_ts, "target_pred"] = y_pred_unscaled

        # c. Preparar el input para el siguiente paso
        new_row_data = {"target": y_pred_unscaled}
        # new_row.index ya es tz-aware (America/Santiago)
        new_row = pd.DataFrame(new_row_data, index=[future_ts])
        
        # 🚨 CORRECCIÓN: El índice ya es tz-aware. No se necesita .tz_localize
        # Simplemente asignamos el nombre.
        new_row.index.name = "ts"
        
        # Añadir feriados
        if holidays_set:
            new_row_date = new_row.index[0].date()
            new_row["feriados"] = 1 if new_row_date in holidays_set else 0
        else:
            new_row["feriados"] = 0
        
        dfw = pd.concat([dfw, new_row])
        dfw = dfw[~dfw.index.duplicated(keep='last')] 
        
        # Generar features para el nuevo input
        df_next_input = dfw.tail(MAX_TIMESTAMPS).copy()
        df_next_input = features.add_time_parts(df_next_input)
        df_next_input = features.add_lags_mas(df_next_input, target_col="target")
        
        X_next_input = features.dummies_and_reindex(df_next_input, training_cols)
        
        X_next_input_scaled = scaler.transform(X_next_input.values)
        current_sequence = np.expand_dims(X_next_input_scaled, axis=0)
        
    dfp["target_pred"] = pd.to_numeric(dfp["target_pred"], errors="coerce")
    
    return dfp

# ----------------------------------------------------------------------
# Función principal de inferencia (UNIFICADA)
# ----------------------------------------------------------------------
def forecast_120d(
    df_historic_raw: pd.DataFrame, 
    horizon_days: int = 120, # ACEPTA EL ARGUMENTO DE MAIN
    holidays_set: set | None = None, # ACEPTA EL ARGUMENTO DE MAIN
    artifacts_path: str = None 
) -> pd.DataFrame:
    """
    Carga artefactos y realiza la predicción de 120 días (2880 pasos).
    """
    # Importación local
    from . import inference_config as config 
    
    if artifacts_path is None:
        artifacts_path = config.ARTIFACTS_PATH
        
    HORIZONTE_PRED_PASOS = horizon_days * 24
    
    # 1. Cargar artefactos
    model, scaler, training_cols = load_all_artifacts(artifacts_path)
    if model is None or scaler is None or training_cols is None:
        raise RuntimeError("No se pudieron cargar todos los artefactos de inferencia.")

    # 2. Preparar secuencia de input (historia)
    df_historic_cleaned, X_input_sequence = prepare_input_sequence(
        df_historic_raw, 
        training_cols
    )
    
    # 3. Realizar predicción iterativa
    dfp = iterative_forecast(
        df_historic_cleaned, 
        X_input_sequence, 
        model, 
        scaler, 
        training_cols, 
        HORIZONTE_PRED_PASOS,
        holidays_set # Pasamos el set de feriados
    )
    
    # 4. Generación de features para el DataFrame de Predicción (dfp)
    all_cols_expected = training_cols + ["target_pred"]
    all_cols_expected = list(dict.fromkeys(all_cols_expected))

    dfp_with_future = pd.concat([df_historic_cleaned.drop(columns=["target"], errors="ignore"), dfp])
    dfp_with_future = features.add_time_parts(dfp_with_future)
    
    dfp_with_future["target"] = dfp_with_future["target_pred"].combine_first(df_historic_cleaned["target"])
    dfp_with_future = features.add_lags_mas(dfp_with_future, target_col="target")
    
    dfp_with_future = features.dummies_and_reindex(dfp_with_future, all_cols_expected)

    # CORRECCIÓN DE DUPLICADOS EN EL ÍNDICE (prevención del ValueError)
    if dfp_with_future.index.duplicated().any():
        print("Advertencia: Se encontraron y eliminaron duplicados en el índice antes del reindexado de columnas.")
        dfp_with_future = dfp_with_future[~dfp_with_future.index.duplicated(keep='last')]
    
    dfp_with_future = dfp_with_future.reindex(columns=all_cols_expected, fill_value=0.0) 
    
    # 5. Filtrar solo las predicciones y limpiar
    df_hourly = dfp_with_future.loc[dfp.index].copy()
    
    if "target_pred" not in df_hourly.columns:
        df_hourly["target_pred"] = 0.0

    return df_hourly[["target_pred"]]
