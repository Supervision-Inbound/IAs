# src/inferencia/alertas_clima.py
# (Asegúrate de que las importaciones y constantes iniciales sean correctas para tu proyecto)
import pandas as pd
import numpy as np
import joblib
import os
import json

from .features import add_time_parts # Necesario para dow, hour
from .utils_io import write_json # Para escribir la salida

# --- Constantes (Asegúrate que coincidan con tu entrenamiento/inferencia) ---
CLIMA_DIR = "data" # O donde esté el archivo de clima histórico para la inferencia
CLIMA_HIST_FILE = os.path.join(CLIMA_DIR, "historico_clima.csv") # Archivo histórico usado en inferencia
BASELINES_FILE = "models/baselines_clima.pkl" # Archivo guardado en entrenamiento Fase 2
RIESGOS_MODEL = "models/modelo_riesgos.keras" # Modelo de riesgos (opcional, si se usa probabilidad)
RIESGOS_SCALER = "models/scaler_riesgos.pkl" # Scaler de riesgos (opcional)
RIESGOS_COLS = "models/training_columns_riesgos.json" # Columnas de riesgos (opcional)

PUBLIC_DIR = "public"
TIMEZONE = "America/Santiago" # Asegurar consistencia

# Umbrales (pueden ajustarse)
ANOMALIA_THRESHOLD = 2.5 # Desviaciones estándar para considerar anomalía
PCT_COMUNAS_THRESHOLD = 0.15 # 15% de comunas afectadas para alerta general
PROBABILIDAD_RIESGO_THRESHOLD = 0.6 # Umbral si se usa el modelo de riesgos

# Columnas esperadas de clima
WEATHER_METRICS = ['temperatura', 'precipitacion', 'lluvia']
# --- Fin Constantes ---

def _load_cols(path: str):
    """Carga columnas desde JSON."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"WARN: Archivo de columnas no encontrado: {path}")
        return None
    except Exception as e:
        print(f"ERROR: No se pudo cargar archivo de columnas {path}: {e}")
        return None


def _load_clima_historico(path=CLIMA_HIST_FILE):
    """Carga y preprocesa el archivo histórico de clima para inferencia."""
    if not os.path.exists(path):
        print(f"WARN: Archivo de clima histórico no encontrado en {path}. No se generarán alertas.")
        return None
    try:
        try: df = pd.read_csv(path, low_memory=False)
        except (pd.errors.ParserError, UnicodeDecodeError): df = pd.read_csv(path, delimiter=';', low_memory=False)

        df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]

        # Renombrar columnas a estándar
        column_map = {'temperatura': ['temperature_2m', 'temperatura'],
                      'precipitacion': ['precipitation', 'precipitacion', 'precipitación'],
                      'lluvia': ['rain', 'lluvia']}
        for std_name, variations in column_map.items():
            for var in variations:
                if var in df.columns and std_name not in df.columns:
                    df = df.rename(columns={var: std_name})
                    break

        # Asegurar columnas de fecha/hora
        date_col = next((c for c in df.columns if 'fecha' in c), None)
        hour_col = next((c for c in df.columns if 'hora' in c), None)
        if not date_col or not hour_col:
            print(f"WARN: No se encontraron columnas 'fecha'/'hora' en {path}. No se puede procesar clima.")
            return None

        # Parsear timestamp
        try: df["ts"] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[hour_col].astype(str), errors='coerce', dayfirst=True)
        except Exception as e: print(f"WARN: Error parseando fecha/hora en clima: {e}"); return None

        df = df.dropna(subset=["ts"])
        if df.empty: print("WARN: DataFrame de clima vacío después de parsear ts."); return None

        # Localizar timezone
        df["ts"] = df["ts"].dt.tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT')
        df = df.dropna(subset=["ts"]) # Drop NaT
        if df.empty: print("WARN: DataFrame de clima vacío después de localizar timezone."); return None

        # Asegurar columna 'comuna'
        if 'comuna' not in df.columns:
            print("WARN: No se encontró columna 'comuna' en archivo de clima. No se puede calcular baseline.")
            return None

        # Seleccionar y ordenar
        cols_to_keep = ['ts', 'comuna'] + [m for m in WEATHER_METRICS if m in df.columns]
        df = df[cols_to_keep].sort_values(['comuna', 'ts']).reset_index(drop=True)

        # Rellenar NaNs en métricas de clima (antes de add_time_parts)
        metrics_present = [m for m in WEATHER_METRICS if m in df.columns]
        for col in metrics_present:
             df[col] = pd.to_numeric(df[col], errors='coerce')
             # Relleno robusto por comuna
             df[col] = df.groupby('comuna', group_keys=False)[col].apply(lambda x: x.ffill().bfill())
        # Eliminar filas si *aún* hay NaNs después de rellenar
        df = df.dropna(subset=metrics_present, how='any')


        print(f"INFO: Clima histórico cargado y preprocesado. Filas: {len(df)}")
        return df

    except Exception as e:
        print(f"ERROR: Falló la carga/procesamiento del clima histórico {path}. Error: {e}")
        return None

def _anomalias_vs_baseline(df_clima_actual, baselines):
    """Calcula las anomalías (z-score) vs baseline por comuna/dow/hour."""
    if df_clima_actual is None or df_clima_actual.empty:
        print("INFO: No hay datos de clima actual para calcular anomalías.")
        return pd.DataFrame() # Devuelve vacío si no hay datos
    if baselines is None or baselines.empty:
        print("WARN: No hay datos de baseline cargados. No se pueden calcular anomalías.")
        return pd.DataFrame()

    print("INFO: Calculando anomalías climáticas...")
    d = add_time_parts(df_clima_actual.copy()) # Necesita 'dow', 'hour'

    # Verificar columnas en 'baselines' ANTES del merge
    expected_baseline_cols = ['comuna', 'dow', 'hour']
    metrics_present_in_clima = [m for m in WEATHER_METRICS if m in d.columns]
    for metric in metrics_present_in_clima:
        expected_baseline_cols.extend([f'{metric}_median', f'{metric}_std'])

    missing_baseline_cols = [col for col in expected_baseline_cols if col not in baselines.columns]
    if missing_baseline_cols:
        print(f"WARN: Faltan columnas esperadas en el archivo baselines: {missing_baseline_cols}")
        # Intentar continuar solo con las columnas presentes? O devolver vacío?
        # Decidimos continuar, pero las anomalías para esas métricas faltarán.
        baselines_cols_to_use = [col for col in expected_baseline_cols if col in baselines.columns]
        if len(baselines_cols_to_use) <= 3: # Si solo quedan comuna, dow, hour
             print("ERROR: No hay columnas de métricas válidas en baselines. No se pueden calcular anomalías.")
             return pd.DataFrame()
        baselines = baselines[baselines_cols_to_use]


    # --- ¡¡¡ESTA ES LA LÍNEA CORREGIDA!!! ---
    # Merge usando los nombres de columna correctos (sin '_') del baseline
    try:
        merged = d.merge(baselines, on=["comuna", "dow", "hour"], how="left")
    except KeyError as e:
         print(f"ERROR en merge de anomalías: {e}. Columnas disponibles en clima: {d.columns}. Columnas disponibles en baseline: {baselines.columns}")
         return pd.DataFrame()
    except Exception as e:
        print(f"ERROR inesperado en merge de anomalías: {e}")
        return pd.DataFrame()
    # --- FIN CORRECCIÓN ---


    # Calcular anomalías solo para métricas presentes en ambos dataframes
    anomalias = {}
    metrics_calculables = []
    for metric in metrics_present_in_clima:
        median_col = f'{metric}_median'
        std_col = f'{metric}_std'
        if median_col in merged.columns and std_col in merged.columns:
            # Rellenar NaNs post-merge (si comuna/dow/hour no estaba en baseline)
            merged[median_col] = merged[median_col].fillna(merged[metric].median())
            merged[std_col] = merged[std_col].fillna(0) # Asumir 0 std si falta baseline

            anomalia_col = f'anomalia_{metric}'
            # Evitar división por cero o std muy pequeños
            std_safe = merged[std_col].replace(0, 1e-6)
            merged[anomalia_col] = (merged[metric] - merged[median_col]) / std_safe
            anomalias[anomalia_col] = merged[anomalia_col]
            metrics_calculables.append(metric) # Marcar como calculada
        else:
            print(f"WARN: No se encontraron columnas {median_col}/{std_col} en baselines mergeados para calcular anomalía de {metric}.")


    # Crear DataFrame de salida solo con 'ts' y las anomalías calculadas
    if not anomalias:
         print("WARN: No se pudo calcular ninguna anomalía.")
         return pd.DataFrame()

    df_anomalias = pd.DataFrame(anomalias, index=merged.index)
    df_anomalias['ts'] = merged['ts'] # Añadir ts para agrupación posterior
    # Añadir comuna por si se quiere agrupar por comuna en el futuro
    df_anomalias['comuna'] = merged['comuna']


    # Devolver DataFrame con ts, comuna y columnas de anomalia_*
    return df_anomalias[['ts', 'comuna'] + list(anomalias.keys())]


def _agregar_anomalias(df_anomalias):
    """Agrega anomalías por timestamp para obtener features de riesgo."""
    if df_anomalias is None or df_anomalias.empty:
        return pd.DataFrame()

    print("INFO: Agregando anomalías por timestamp...")
    anomaly_cols = [col for col in df_anomalias.columns if col.startswith('anomalia_')]
    if not anomaly_cols:
        print("WARN: No hay columnas de anomalías para agregar.")
        return pd.DataFrame()

    n_comunas = df_anomalias['comuna'].nunique()
    if n_comunas == 0:
        print("WARN: No hay comunas en df_anomalias para agregar.")
        return pd.DataFrame()

    agg_functions = {
         col: [('max', 'max'), ('sum', 'sum'), ('pct_comunas_afectadas', lambda x: (x > ANOMALIA_THRESHOLD).sum() / n_comunas)]
         for col in anomaly_cols
    }

    df_agregado = df_anomalias.groupby('ts').agg(agg_functions).reset_index()
    # Aplanar MultiIndex de columnas correctamente
    df_agregado.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df_agregado.columns.values]
    df_agregado = df_agregado.rename(columns={'ts_':'ts'}) # Renombrar columna ts

    print(f"INFO: Anomalías agregadas. Filas: {len(df_agregado)}")
    return df_agregado

def _generar_alertas_simples(df_agregado):
    """Genera alertas basadas en umbrales simples de anomalía."""
    alertas = []
    if df_agregado is None or df_agregado.empty:
        return alertas

    print("INFO: Generando alertas simples basadas en umbrales...")
    # Iterar sobre cada hora agregada
    for _, row in df_agregado.iterrows():
        ts = row['ts']
        alertas_hora = []
        # Revisar cada métrica de anomalía agregada
        for col in df_agregado.columns:
            if col.endswith('_pct_comunas_afectadas') and pd.notna(row[col]) and row[col] >= PCT_COMUNAS_THRESHOLD:
                metrica = col.split('_')[1] # Extraer nombre de métrica (temp, precip, lluvia)
                alertas_hora.append(f"{metrica.capitalize()} anómala ({row[col]:.1%})")
            # Podrías añadir más reglas aquí, ej: max > X*threshold

        if alertas_hora:
            alertas.append({
                "ts": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "alerta": f"Alto riesgo climático: {'; '.join(alertas_hora)}"
            })
    print(f"INFO: Generadas {len(alertas)} alertas simples.")
    return alertas

# --- Función Principal ---
def generar_alertas(df_prediccion_llamadas):
    """
    Función principal para generar alertas climáticas.
    1. Carga clima histórico y baselines.
    2. Calcula anomalías para las horas futuras de la predicción.
    3. Agrega anomalías por hora.
    4. Genera alertas basadas en umbrales simples.
    5. Escribe las alertas en un JSON.
    """
    print("\n--- Iniciando Generación de Alertas Climáticas ---")
    alertas_finales = []
    output_path = os.path.join(PUBLIC_DIR, "alertas_clima.json")

    try:
        # 1. Cargar Baselines
        if not os.path.exists(BASELINES_FILE):
            print(f"WARN: Archivo de baselines no encontrado en {BASELINES_FILE}. No se pueden generar alertas.")
            write_json(output_path, []) # Escribir JSON vacío
            return
        baselines = joblib.load(BASELINES_FILE)
        # Asegurarse que las columnas del baseline estén en minúsculas y sin '_' extras al final
        baselines.columns = [c.lower().replace('_','') if c.endswith('_') else c.lower() for c in baselines.columns]
        # Renombrar columnas clave si es necesario (ej: si se guardaron con '_')
        rename_map = {'comuna_':'comuna', 'dow_':'dow', 'hour_':'hour'}
        baselines = baselines.rename(columns={k:v for k,v in rename_map.items() if k in baselines.columns})

        print("INFO: Baselines climáticos cargados.")

        # 2. Cargar y procesar clima histórico (para el rango futuro)
        df_clima_hist = _load_clima_historico(CLIMA_HIST_FILE)
        if df_clima_hist is None or df_clima_hist.empty:
             write_json(output_path, []) # Escribir JSON vacío si no hay clima
             return

        # Filtrar clima histórico solo para el rango de la predicción
        if df_prediccion_llamadas is None or df_prediccion_llamadas.empty:
             print("WARN: DataFrame de predicción de llamadas vacío. No se pueden generar alertas.")
             write_json(output_path, [])
             return

        # Asegurar índice datetime en predicción
        if not isinstance(df_prediccion_llamadas.index, pd.DatetimeIndex):
            try: df_prediccion_llamadas.index = pd.to_datetime(df_prediccion_llamadas.index)
            except: print("ERROR: Índice de predicción de llamadas inválido."); write_json(output_path, []); return

        start_pred = df_prediccion_llamadas.index.min()
        end_pred = df_prediccion_llamadas.index.max()

        # Filtrar clima histórico por el rango de fechas/horas de la predicción
        df_clima_futuro = df_clima_hist[
            (df_clima_hist['ts'] >= start_pred) & (df_clima_hist['ts'] <= end_pred)
        ].copy()

        if df_clima_futuro.empty:
            print(f"INFO: No se encontraron datos climáticos históricos para el rango futuro ({start_pred} a {end_pred}). No se generarán alertas.")
            write_json(output_path, [])
            return

        # 3. Calcular Anomalías
        df_anomalias = _anomalias_vs_baseline(df_clima_futuro, baselines)

        # 4. Agregar Anomalías
        df_agregado = _agregar_anomalias(df_anomalias)

        # 5. Generar Alertas (Método Simple)
        alertas_finales = _generar_alertas_simples(df_agregado)

        # (Opcional: Podrías añadir lógica para usar el modelo de riesgos aquí si lo deseas)

    except FileNotFoundError as e:
        print(f"ERROR: Archivo no encontrado durante generación de alertas: {e}")
    except Exception as e:
        print(f"ERROR inesperado durante la generación de alertas climáticas: {e}")
        import traceback
        traceback.print_exc() # Imprimir traceback para depuración

    # 6. Escribir JSON de Alertas (incluso si está vacío)
    print(f"INFO: Escribiendo {len(alertas_finales)} alertas en {output_path}")
    write_json(output_path, alertas_finales)
    print("--- Generación de Alertas Climáticas Finalizada ---")
