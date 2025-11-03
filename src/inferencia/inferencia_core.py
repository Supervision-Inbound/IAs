# -*- coding: utf-8 -*-
import datetime
import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf

from src.inferencia import features
from src.inferencia import inference_config as config

# ----------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------
TIMEZONE = features.TIMEZONE
MAX_TIMESTAMPS = config.MAX_TIMESTAMPS
HORIZONTE_PRED_DIAS = config.HORIZONTE_PRED_DIAS
HORIZONTE_PRED_PASOS = HORIZONTE_PRED_DIAS * 24

# ----------------------------------------------------------------------
# Carga de modelos y datos
# ----------------------------------------------------------------------
def load_all_artifacts(artifacts_path: str):
    """Carga el modelo Keras, el scaler y las columnas de entrenamiento."""
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
        scaler = pd.read_pickle(scaler_path)
    except Exception as e:
        print(f"Error al cargar el scaler desde {scaler_path}: {e}")
        return model, None, None

    try:
        with open(cols_path, 'r') as f:
            training_cols = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error al cargar las columnas de entrenamiento desde {cols_path}: {e}")
        return model, scaler, None

    return model, scaler, training_cols

# ----------------------------------------------------------------------
# Preparación de datos (Input Sequence)
# ----------------------------------------------------------------------
def prepare_input_sequence(df_historic_raw: pd.DataFrame, training_cols: list):
    """
    Prepara la secuencia histórica y la escala para la inferencia.
    """
    dfh = features.ensure_ts(df_historic_raw)
    dfh = dfh.sort_index()

    # Asegurar que solo tenemos MAX_TIMESTAMPS o más
    if len(dfh) < MAX_TIMESTAMPS:
        raise ValueError(f"Datos históricos insuficientes. Se requieren al menos {MAX_TIMESTAMPS} puntos.")

    # Tomar la secuencia más reciente para el input de LSTM
    df_input = dfh.tail(MAX_TIMESTAMPS).copy()
    
    # Crear features de tiempo, lags y MAs
    df_input = features.add_time_parts(df_input)
    df_input = features.add_lags_mas(df_input, target_col="target")
    
    # Crear dummies y reindexar a las columnas de entrenamiento
    X_input = features.dummies_and_reindex(df_input, training_cols)
    
    # Limpiar cualquier columna inesperada antes de escalar (aunque reindex ya lo hizo)
    X_input = X_input[training_cols]
    
    # Devolver el DataFrame limpio y la secuencia de input (como array)
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
    horizonte_pasos: int
) -> pd.DataFrame:
    """
    Realiza la predicción iterativa (paso a paso) de la serie temporal.
    """
    print(f"Iniciando predicción iterativa (LSTM) para {horizonte_pasos} pasos...")
    
    # 1. Preparación de secuencias iniciales
    # Tomar el último target conocido y el último índice (timestamp)
    last_known_target = df_historic_cleaned["target"].iloc[-1]
    last_known_ts = df_historic_cleaned.index[-1]

    # Tomar la secuencia de input escalada y darle la forma correcta (1, timesteps, features)
    X_input_scaled = scaler.transform(X_input_sequence)
    current_sequence = np.expand_dims(X_input_scaled, axis=0) # Shape: (1, 168, N_FEATURES)

    # 2. Inicialización del DataFrame de predicciones (dfp)
    
    # Crear la lista de Timestamps futuros
    future_timestamps = [last_known_ts + datetime.timedelta(hours=h) for h in range(1, horizonte_pasos + 1)]
    dfp = pd.DataFrame(index=future_timestamps, columns=["target_pred"])
    
    # Crear un DataFrame temporal de trabajo (dfw) con la historia + el futuro para generar features
    dfw = df_historic_cleaned.copy()
    dfw["target"] = dfw["target"].astype(float) # Asegurar float
    
    # 3. Bucle de predicción
    for step in range(horizonte_pasos):
        
        day = (step // 24) + 1
        if step % 24 == 0:
            print(f"Prediciendo Día {day}/{HORIZONTE_PRED_DIAS}...")

        # a. Predicción
        y_scaled = model.predict(current_sequence, verbose=0)[0, 0]
        y_pred_unscaled = scaler.inverse_transform([[y_scaled]])[0, 0]
        
        # b. Almacenar predicción
        future_ts = future_timestamps[step]
        dfp.loc[future_ts, "target_pred"] = y_pred_unscaled

        # c. Preparar el input para el siguiente paso
        
        # Generar el registro futuro (temporalmente con la predicción)
        new_row_data = {"target": y_pred_unscaled}
        new_row = pd.DataFrame(new_row_data, index=[future_ts])
        
        # Asegurar la TZ y el nombre del índice
        new_row.index = new_row.index.tz_localize('UTC').tz_convert(TIMEZONE)
        new_row.index.name = "ts"
        
        # Agregar la nueva fila al DataFrame de trabajo (dfw)
        dfw = pd.concat([dfw, new_row])
        dfw = dfw.sort_index().tail(MAX_TIMESTAMPS + step + 1) # Mantener historial suficiente
        
        # Generar features para el nuevo input
        df_next_input = dfw.tail(MAX_TIMESTAMPS).copy()
        df_next_input = features.add_time_parts(df_next_input)
        df_next_input = features.add_lags_mas(df_next_input, target_col="target")
        
        # Crear dummies y reindexar (usando la función de features)
        X_next_input = features.dummies_and_reindex(df_next_input, training_cols)
        
        # Escalar y preparar la secuencia
        X_next_input_scaled = scaler.transform(X_next_input.values)
        current_sequence = np.expand_dims(X_next_input_scaled, axis=0)
        
    # 4. Finalización y limpieza
    dfp["target_pred"] = pd.to_numeric(dfp["target_pred"], errors="coerce")
    
    return dfp

# ----------------------------------------------------------------------
# Función principal de inferencia
# ----------------------------------------------------------------------
def forecast_120d(
    df_historic_raw: pd.DataFrame, 
    artifacts_path: str = config.ARTIFACTS_PATH
) -> pd.DataFrame:
    """
    Carga artefactos y realiza la predicción de 120 días (2880 pasos).
    """
    
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
        HORIZONTE_PRED_PASOS
    )
    
    # 4. Generación de features para el DataFrame de Predicción (dfp)

    # Crear una lista unificada de todas las columnas esperadas
    all_cols_expected = training_cols + ["target_pred"]
    all_cols_expected = list(dict.fromkeys(all_cols_expected)) # Limpieza final de duplicados

    # Unir la historia limpia y las predicciones
    dfp_with_future = pd.concat([df_historic_cleaned.drop(columns=["target"], errors="ignore"), dfp])
    
    # Crear features de tiempo para el futuro (dfp)
    dfp_with_future = features.add_time_parts(dfp_with_future)
    
    # Crear features de lags y MAs (necesita la historia previa)
    # Usamos target_pred como la columna para lags/MAs en el futuro, y target para el pasado
    dfp_with_future["target"] = dfp_with_future["target_pred"].combine_first(dfp_with_future["target"])
    dfp_with_future = features.add_lags_mas(dfp_with_future, target_col="target")
    
    # Crear Dummies y asegurar el orden de las columnas de entrenamiento
    dfp_with_future = features.dummies_and_reindex(dfp_with_future, all_cols_expected)

    # 🚨 CORRECCIÓN FINAL: Aseguramos que el índice de filas sea único antes de reindexar columnas
    # Esto previene el error 'cannot reindex on an axis with duplicate labels' si el redondeo 
    # del tiempo introdujo duplicados que no fueron limpiados correctamente.
    if dfp_with_future.index.duplicated().any():
        print("Advertencia: Se encontraron y eliminaron duplicados en el índice antes del reindexado de columnas.")
        dfp_with_future = dfp_with_future[~dfp_with_future.index.duplicated(keep='last')]
    
    # Reindexar a las columnas esperadas (línea 304 original)
    dfp_with_future = dfp_with_future.reindex(columns=all_cols_expected, fill_value=0.0) 
    
    # 5. Filtrar solo las predicciones y limpiar
    df_hourly = dfp_with_future.loc[dfp.index].copy()
    
    # Asegurar que la columna de predicción esté presente
    if "target_pred" not in df_hourly.columns:
        df_hourly["target_pred"] = 0.0

    return df_hourly[["target_pred"]]

