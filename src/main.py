# -*- coding: utf-8 -*-
"""
main.py
Orquestación de la inferencia v7:
- Llama a inferencia_core para:
  * Cargar artefactos (planner, risk, tmo)
  * Normalizar históricos (hosting y TMO)
  * Construir futuro horario
  * Predecir llamadas (planner) sin tocar su lógica original
  * Predecir TMO AUTORREGRESIVO (desacoplado del clima)
  * Generar JSON horaria, diaria y (opcional) alertas climáticas
Requiere: models/*.keras, *.pkl, *.json y data/*.csv
"""

import os
import json
import argparse
import warnings
import pandas as pd
import numpy as np
import joblib

from inferencia.inferencia_core import (
    TZ,
    read_data,
    ensure_ts_and_tz,
    add_time_parts,
    build_future_frame,
    forecast_120d,
    load_artifacts,
    predict_tmo_autoregressive,
    calculate_erlang_agents,
    simulate_future_weather,
    process_future_climate,
    predict_risk,
)

# Si tienes un generador de alertas aparte, úsalo:
try:
    from alertas_clima import generate_alerts_json as gen_alertas
except Exception:
    gen_alertas = None

warnings.filterwarnings("ignore")

# --- Paths base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
PUBLIC_DIR = os.path.join(ROOT_DIR, "public")

# --- Archivos de datos
HOSTING_FILE = os.path.join(DATA_DIR, "historical_data.csv")  # llamadas históricas
TMO_FILE = os.path.join(DATA_DIR, "TMO_HISTORICO.csv")        # TMO histórico puro
FERIADOS_FILE = os.path.join(DATA_DIR, "Feriados_Chilev2.csv")
CLIMA_HIST_FILE = os.path.join(DATA_DIR, "historico_clima.csv")  # si no existe, se simula

# --- Targets / nombres estándar entrenados
TARGET_CALLS = "recibidos_nacional"
TARGET_TMO = "tmo_general"


def _cargar_feriados():
    """
    Devuelve un set() de fechas de feriados (date).
    Si no se encuentra el CSV, retorna set() vacío.
    """
    try:
        df_fer = read_data(FERIADOS_FILE)
        # Se espera columna "Fecha" en dd-mm-YYYY
        if "Fecha" in df_fer.columns:
            fechas = pd.to_datetime(df_fer["Fecha"], format="%d-%m-%Y", errors="coerce").dt.date
        else:
            # fallback: intenta parsear primer col
            first_col = df_fer.columns[0]
            fechas = pd.to_datetime(df_fer[first_col], errors="coerce").dt.date
        return set(fechas.dropna())
    except Exception as e:
        print(f"[Adv] No se pudo cargar feriados ({FERIADOS_FILE}): {e}. Se asume 0.")
        return set()


def _map_calls_column(dfh: pd.DataFrame) -> pd.DataFrame:
    """
    Garantiza que exista la columna TARGET_CALLS:
    - Si viene 'recibidos', la renombra
    - Si viene con headers mal separados por ';', read_data ya intenta corregir
    """
    dfh = dfh.copy()
    cols_lower = [c.lower() for c in dfh.columns]
    # directos
    if TARGET_CALLS in dfh.columns:
        return dfh
    if "recibidos" in dfh.columns:
        dfh = dfh.rename(columns={"recibidos": TARGET_CALLS})
        return dfh

    # alias frecuentes
    alias = ["llamadas", "llamadas_recibidas", "recibidos_total"]
    for a in alias:
        if a in cols_lower:
            real = dfh.columns[cols_lower.index(a)]
            dfh = dfh.rename(columns={real: TARGET_CALLS})
            return dfh

    raise ValueError(
        f"No se encontró columna de llamadas '{TARGET_CALLS}' ni alias comunes. "
        f"Columnas disponibles: {list(dfh.columns)}"
    )


def _prepara_hosting(dfh_raw: pd.DataFrame, feriados_set: set) -> pd.DataFrame:
    """
    Normaliza el histórico de llamadas:
    - crea/ajusta ts y TZ
    - mapea columna de llamadas a TARGET_CALLS
    - rellena 'feriados'
    - agrega features temporales
    - agrupa por ts (sum llam, max feriados)
    """
    dfh = ensure_ts_and_tz(dfh_raw)
    dfh = _map_calls_column(dfh)

    # feriados
    if "feriados" not in dfh.columns:
        dfh["feriados"] = dfh["ts"].dt.date.isin(feriados_set).astype(int)
    else:
        dfh["feriados"] = pd.to_numeric(dfh["feriados"], errors="coerce").fillna(0).astype(int)

    # Agregación
    dfh = dfh.groupby("ts").agg({TARGET_CALLS: "sum", "feriados": "max"}).reset_index()

    # Partes temporales (dow, month, hour, etc.)
    dfh = add_time_parts(dfh)
    return dfh


def _prepara_tmo_hist(df_tmo: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza TMO_HISTORICO:
    - crea ts y TZ
    - asegura TARGET_TMO
    - deriva proporciones si es posible
    """
    dft = ensure_ts_and_tz(df_tmo)
    dft.columns = [c.lower().strip().replace(" ", "_") for c in dft.columns]

    # nombre estándar del objetivo
    if TARGET_TMO not in dft.columns:
        # intentar construir ponderado si están disponibles
        if all(c in dft.columns for c in ["tmo_comercial", "q_llamadas_comercial", "tmo_tecnico", "q_llamadas_tecnico", "q_llamadas_general"]):
            dft[TARGET_TMO] = (
                dft["tmo_comercial"] * dft["q_llamadas_comercial"]
                + dft["tmo_tecnico"] * dft["q_llamadas_tecnico"]
            ) / (dft["q_llamadas_general"] + 1e-6)
        elif "tmo (segundos)" in dft.columns:
            dft[TARGET_TMO] = pd.to_numeric(dft["tmo (segundos)"], errors="coerce")
        else:
            raise ValueError(f"No se encontró '{TARGET_TMO}' ni columnas para ponderarlo.")

    # proporciones operacionales si existen
    if "q_llamadas_comercial" in dft.columns and "q_llamadas_general" in dft.columns:
        dft["proporcion_comercial"] = dft["q_llamadas_comercial"] / (dft["q_llamadas_general"] + 1e-6)
    else:
        dft["proporcion_comercial"] = 0.0

    if "q_llamadas_tecnico" in dft.columns and "q_llamadas_general" in dft.columns:
        dft["proporcion_tecnica"] = dft["q_llamadas_tecnico"] / (dft["q_llamadas_general"] + 1e-6)
    else:
        dft["proporcion_tecnica"] = 0.0

    # mantener columnas clave si existen
    for c in ["tmo_comercial", "tmo_tecnico"]:
        if c not in dft.columns:
            dft[c] = 0.0

    # ordenar
    return dft.sort_values("ts").reset_index(drop=True)


def main(horizonte_dias: int):
    print("=" * 70)
    print(f"PIPELINE INFERENCIA v7  |  TZ={TZ}  |  Horizonte={horizonte_dias} días")
    print("=" * 70)

    os.makedirs(PUBLIC_DIR, exist_ok=True)

    # --- 1) Cargar artefactos (planner, risk, tmo)
    print("\n[Fase 1] Cargando artefactos...")
    planner = load_artifacts(MODEL_DIR, "planner")
    risk = load_artifacts(MODEL_DIR, "risk")
    tmo_art = load_artifacts(MODEL_DIR, "tmo")

    # Para planner, indicamos la columna objetivo estándar
    planner["target_calls"] = TARGET_CALLS

    # Baselines clima (si existen)
    try:
        baselines_clima = joblib.load(os.path.join(MODEL_DIR, "baselines_clima.pkl"))
    except Exception:
        baselines_clima = pd.DataFrame()

    # --- 2) Cargar históricos
    print("\n[Fase 2] Cargando históricos (hosting y TMO)...")
    feriados_set = _cargar_feriados()

    dfh_raw = read_data(HOSTING_FILE)
    df_hosting_proc = _prepara_hosting(dfh_raw, feriados_set)
    last_ts = df_hosting_proc["ts"].max()
    print(f"  Última hora histórica (llamadas): {last_ts}")

    dft_raw = read_data(TMO_FILE)
    df_tmo_hist = _prepara_tmo_hist(dft_raw)
    last_tmo_ts = df_tmo_hist["ts"].max()
    print(f"  Última hora histórica (TMO):      {last_tmo_ts}")

    # --- 3) Construir futuro horario base
    print("\n[Fase 3] Construyendo esqueleto futuro...")
    df_future_base = build_future_frame(last_ts, horizonte_dias, feriados_set)
    print(f"  Futuro desde {df_future_base['ts'].min()} hasta {df_future_base['ts'].max()}")

    # --- 4) Llamadas + Riesgo (clima solo para riesgo/alertas)
    print("\n[Fase 4] Predicción de llamadas y riesgo...")
    # forecast_120d: aplica clima->anomalías->riesgo y llamadas (planner)
    try:
        clima_hist_df = read_data(CLIMA_HIST_FILE)
    except FileNotFoundError:
        clima_hist_df = pd.DataFrame()

    df_future_calls = forecast_120d(
        df_hosting_proc=df_hosting_proc,
        df_future_base=df_future_base,
        holidays_set=feriados_set,
        planner_artifacts=planner,
        risk_artifacts={"model": risk["model"], "scaler": risk["scaler"], "cols": risk["cols"]},
        clima_hist_df=clima_hist_df,
        baselines_clima=baselines_clima,
    )
    # Asegurar columna llamadas_hora existe
    if "llamadas_hora" not in df_future_calls.columns:
        df_future_calls["llamadas_hora"] = 0

    # --- 5) TMO AUTORREGRESIVO (sin clima)
    print("\n[Fase 5] Predicción de TMO (autoregresivo, desacoplado del clima)...")
    df_tmo_pred = predict_tmo_autoregressive(
        df_future=df_future_calls[["ts", "dow", "month", "hour", "feriados", "llamadas_hora"]].copy(),
        df_tmo_hist=df_tmo_hist,
        model=tmo_art["model"],
        scaler=tmo_art["scaler"],
        cols_tmo=tmo_art["cols"],
        target_calls_col="llamadas_hora",
        target_tmo=TARGET_TMO,
        force_zero_anomalias=True,   # <- CLAVE: TMO sin clima
    )
    # Merge TMO
    df_future = pd.merge(df_future_calls, df_tmo_pred, on="ts", how="left")
    df_future["tmo_hora"] = pd.to_numeric(df_future["tmo_hora"], errors="coerce").fillna(0).clip(lower=0)

    # --- 6) Agentes requeridos
    print("\n[Fase 6] Cálculo de agentes (Erlang Aprox.)...")
    df_future["agentes_requeridos"] = calculate_erlang_agents(
        df_future["llamadas_hora"], df_future["tmo_hora"]
    )

    # --- 7) JSON horaria
    print("\n[Fase 7] Exportando JSON horaria...")
    df_horaria = df_future[["ts", "llamadas_hora", "tmo_hora", "agentes_requeridos"]].copy()
    df_horaria["ts"] = df_horaria["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out_horaria = os.path.join(PUBLIC_DIR, "prediccion_horaria.json")
    df_horaria.to_json(out_horaria, orient="records", indent=2, force_ascii=False)
    print(f"  OK -> {out_horaria}")

    # --- 8) JSON diaria (TMO ponderado por llamadas)
    print("\n[Fase 8] Exportando JSON diario (ponderado por llamadas)...")
    tmp = df_future.copy()
    tmp["fecha"] = tmp["ts"].dt.date
    tmp["tmo_num"] = tmp["tmo_hora"] * tmp["llamadas_hora"]
    df_d = tmp.groupby("fecha").agg(
        llamadas_totales_dia=("llamadas_hora", "sum"),
        tmo_ponderado_num=("tmo_num", "sum"),
    )
    df_d["tmo_promedio_diario"] = df_d["tmo_ponderado_num"] / (df_d["llamadas_totales_dia"] + 1e-6)
    # fallback si hay días con 0 llamadas
    if (df_d["llamadas_totales_dia"] == 0).any():
        tmo_simple = tmp.groupby("fecha")["tmo_hora"].mean().fillna(0)
        df_d["tmo_promedio_diario"] = df_d["tmo_promedio_diario"].where(
            df_d["llamadas_totales_dia"] > 0, tmo_simple
        ).fillna(0)
    df_d = df_d.reset_index()[["fecha", "llamadas_totales_dia", "tmo_promedio_diario"]]
    df_d["fecha"] = df_d["fecha"].astype(str)
    df_d["llamadas_totales_dia"] = df_d["llamadas_totales_dia"].astype(int)

    out_diaria = os.path.join(PUBLIC_DIR, "Predicion_daria.json")
    df_d.to_json(out_diaria, orient="records", indent=2, force_ascii=False)
    print(f"  OK -> {out_diaria}")

    # --- 9) Alertas (opcional)
    if gen_alertas is not None:
        print("\n[Fase 9] Generando alertas climáticas...")
        # Para generar alertas necesitamos anomalías por comuna + proba de riesgo
        # Reutilizamos simulación/processing ya hecho:
        df_weather_future = simulate_future_weather(
            clima_hist_df if isinstance(clima_hist_df, pd.DataFrame) else pd.DataFrame(),
            df_future_base["ts"].min(),
            df_future_base["ts"].max(),
        )
        df_agg_anoms, df_per_comuna_anoms = process_future_climate(
            df_weather_future, baselines_clima if isinstance(baselines_clima, pd.DataFrame) else pd.DataFrame()
        )
        df_tmp = pd.merge(df_future_base[["ts"]], df_agg_anoms, on="ts", how="left").fillna(0)
        # riesgo
        df_tmp["risk_proba"] = predict_risk(df_tmp, risk["model"], risk["scaler"], risk["cols"])
        df_risk_out = df_tmp[["ts", "risk_proba"]].copy()

        alertas = gen_alertas(df_per_comuna_anoms, df_risk_out, proba_threshold=0.5, impact_factor=100)
        out_alertas = os.path.join(PUBLIC_DIR, "alertas_climaticas.json")
        with open(out_alertas, "w", encoding="utf-8") as f:
            json.dump(alertas, f, indent=2, ensure_ascii=False)
        print(f"  OK -> {out_alertas}")
    else:
        print("\n[Fase 9] Alertas climáticas deshabilitadas (módulo no disponible).")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETADO CON ÉXITO")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia IA: llamadas y TMO autoregresivo.")
    parser.add_argument("--horizonte", type=int, default=120, help="Horizonte en días (futuro).")
    args = parser.parse_args()
    main(args.horizonte)

