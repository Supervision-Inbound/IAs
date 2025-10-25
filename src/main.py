# -*- coding: utf-8 -*-
"""
main.py
Orquestación de la inferencia v7:
- Carga artefactos (planner, risk, tmo)
- Normaliza históricos (hosting y TMO)
- Construye futuro horario
- Predice llamadas (planner) sin tocar su lógica original
- Predice TMO AUTORREGRESIVO (desacoplado del clima)
- Genera JSON horaria y diaria (y alertas si el módulo existe)
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

try:
    from alertas_clima import generate_alerts_json as gen_alertas
except Exception:
    gen_alertas = None

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
PUBLIC_DIR = os.path.join(ROOT_DIR, "public")

HOSTING_FILE = os.path.join(DATA_DIR, "historical_data.csv")
TMO_FILE = os.path.join(DATA_DIR, "TMO_HISTORICO.csv")
FERIADOS_FILE = os.path.join(DATA_DIR, "Feriados_Chilev2.csv")
CLIMA_HIST_FILE = os.path.join(DATA_DIR, "historico_clima.csv")

TARGET_CALLS = "recibidos_nacional"
TARGET_TMO = "tmo_general"


def _cargar_feriados():
    try:
        df_fer = read_data(FERIADOS_FILE)
        col = "Fecha" if "Fecha" in df_fer.columns else df_fer.columns[0]
        fechas = pd.to_datetime(df_fer[col], format="%d-%m-%Y", errors="coerce").dt.date
        return set(fechas.dropna())
    except Exception as e:
        print(f"[Adv] No se pudo cargar feriados ({FERIADOS_FILE}): {e}. Se asume 0.")
        return set()


def _map_calls_column(dfh: pd.DataFrame) -> pd.DataFrame:
    dfh = dfh.copy()
    if TARGET_CALLS in dfh.columns:
        return dfh
    if "recibidos" in dfh.columns:
        return dfh.rename(columns={"recibidos": TARGET_CALLS})
    lowers = [c.lower() for c in dfh.columns]
    for a in ["llamadas", "llamadas_recibidas", "recibidos_total"]:
        if a in lowers:
            real = dfh.columns[lowers.index(a)]
            return dfh.rename(columns={real: TARGET_CALLS})
    raise ValueError(
        f"No se encontró columna de llamadas '{TARGET_CALLS}' ni alias comunes. "
        f"Columnas disponibles: {list(dfh.columns)}"
    )


def _prepara_hosting(dfh_raw: pd.DataFrame, feriados_set: set) -> pd.DataFrame:
    dfh = ensure_ts_and_tz(dfh_raw)
    dfh = _map_calls_column(dfh)
    if "feriados" not in dfh.columns:
        dfh["feriados"] = dfh["ts"].dt.date.isin(feriados_set).astype(int)
    else:
        dfh["feriados"] = pd.to_numeric(dfh["feriados"], errors="coerce").fillna(0).astype(int)
    dfh = dfh.groupby("ts").agg({TARGET_CALLS: "sum", "feriados": "max"}).reset_index()
    dfh = add_time_parts(dfh)
    return dfh


def _prepara_tmo_hist(dft_raw: pd.DataFrame) -> pd.DataFrame:
    dft = ensure_ts_and_tz(dft_raw)
    dft.columns = [c.lower().strip().replace(" ", "_") for c in dft.columns]
    if TARGET_TMO not in dft.columns:
        if all(c in dft.columns for c in ["tmo_comercial", "q_llamadas_comercial", "tmo_tecnico", "q_llamadas_tecnico", "q_llamadas_general"]):
            dft[TARGET_TMO] = (
                dft["tmo_comercial"] * dft["q_llamadas_comercial"]
                + dft["tmo_tecnico"] * dft["q_llamadas_tecnico"]
            ) / (dft["q_llamadas_general"] + 1e-6)
        elif "tmo_(segundos)" in dft.columns:
            dft[TARGET_TMO] = pd.to_numeric(dft["tmo_(segundos)"], errors="coerce")
        elif "tmo_(segundos);hora" in dft.columns:
            dft[TARGET_TMO] = pd.to_numeric(dft["tmo_(segundos);hora"], errors="coerce")
        elif "tmo_(segundos);hora_numero" in dft.columns:
            dft[TARGET_TMO] = pd.to_numeric(dft["tmo_(segundos);hora_numero"], errors="coerce")
        elif "tmo_(segundos);tmo_general" in dft.columns:
            dft[TARGET_TMO] = pd.to_numeric(dft["tmo_(segundos);tmo_general"], errors="coerce")
        elif "tmo (segundos)" in dft.columns:
            dft[TARGET_TMO] = pd.to_numeric(dft["tmo (segundos)"], errors="coerce")
        else:
            raise ValueError(f"No se encontró '{TARGET_TMO}' ni columnas para construirlo.")
    # proporciones si existen
    if "q_llamadas_comercial" in dft.columns and "q_llamadas_general" in dft.columns:
        dft["proporcion_comercial"] = dft["q_llamadas_comercial"] / (dft["q_llamadas_general"] + 1e-6)
    else:
        dft["proporcion_comercial"] = 0.0
    if "q_llamadas_tecnico" in dft.columns and "q_llamadas_general" in dft.columns:
        dft["proporcion_tecnica"] = dft["q_llamadas_tecnico"] / (dft["q_llamadas_general"] + 1e-6)
    else:
        dft["proporcion_tecnica"] = 0.0
    for c in ["tmo_comercial", "tmo_tecnico"]:
        if c not in dft.columns:
            dft[c] = 0.0
    return dft.sort_values("ts").reset_index(drop=True)


def main(horizonte_dias: int):
    print("=" * 70)
    print(f"PIPELINE INFERENCIA v7 | TZ={TZ} | Horizonte={horizonte_dias} días")
    print("=" * 70)

    os.makedirs(PUBLIC_DIR, exist_ok=True)

    # 1) Artefactos
    print("\n[Fase 1] Cargando artefactos...")
    planner = load_artifacts(MODEL_DIR, "planner")
    risk = load_artifacts(MODEL_DIR, "riesgos")  # nombre entrenado: modelo_riesgos
    tmo_art = load_artifacts(MODEL_DIR, "tmo")

    # planner necesita saber el target de llamadas
    planner["target_calls"] = TARGET_CALLS

    # baselines clima (si existen)
    try:
        baselines_clima = joblib.load(os.path.join(MODEL_DIR, "baselines_clima.pkl"))
    except Exception:
        baselines_clima = pd.DataFrame()

    # 2) Históricos
    print("\n[Fase 2] Cargando históricos...")
    feriados_set = _cargar_feriados()

    dfh_raw = read_data(HOSTING_FILE)
    df_hosting_proc = _prepara_hosting(dfh_raw, feriados_set)
    last_ts = df_hosting_proc["ts"].max()
    print(f"  Última hora histórica (llamadas): {last_ts}")

    dft_raw = read_data(TMO_FILE)
    df_tmo_hist = _prepara_tmo_hist(dft_raw)
    last_tmo_ts = df_tmo_hist["ts"].max()
    print(f"  Última hora histórica (TMO):      {last_tmo_ts}")

    # 3) Futuro base
    print("\n[Fase 3] Construyendo futuro...")
    df_future_base = build_future_frame(last_ts, horizonte_dias, feriados_set)
    print(f"  Futuro: {df_future_base['ts'].min()} -> {df_future_base['ts'].max()}")

    # 4) Llamadas + Riesgo (clima solo para riesgo)
    print("\n[Fase 4] Predicción de llamadas y riesgo...")
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

    # 5) TMO autoregresivo (sin clima)
    print("\n[Fase 5] Predicción de TMO (autoregresivo, sin clima)...")
    df_tmo_pred = predict_tmo_autoregressive(
        df_future=df_future_calls[["ts", "dow", "month", "hour", "feriados", "llamadas_hora"]].copy(),
        df_tmo_hist=df_tmo_hist,
        model=tmo_art["model"],
        scaler=tmo_art["scaler"],
        cols_tmo=tmo_art["cols"],
        target_calls_col="llamadas_hora",
        target_tmo=TARGET_TMO,
        force_zero_anomalias=True,
    )

    # Merge
    df_future = pd.merge(df_future_calls, df_tmo_pred, on="ts", how="left")
    df_future["tmo_hora"] = pd.to_numeric(df_future["tmo_hora"], errors="coerce").fillna(0).clip(lower=0)

    # 6) Erlang
    print("\n[Fase 6] Agentes requeridos...")
    df_future["agentes_requeridos"] = calculate_erlang_agents(df_future["llamadas_hora"], df_future["tmo_hora"])

    # 7) JSON horaria
    print("\n[Fase 7] JSON horaria...")
    os.makedirs(PUBLIC_DIR, exist_ok=True)
    out_h = os.path.join(PUBLIC_DIR, "prediccion_horaria.json")
    df_h = df_future[["ts", "llamadas_hora", "tmo_hora", "agentes_requeridos"]].copy()
    df_h["ts"] = df_h["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_h.to_json(out_h, orient="records", indent=2, force_ascii=False)
    print(f"  OK -> {out_h}")

    # 8) JSON diaria ponderada
    print("\n[Fase 8] JSON diaria ponderada...")
    tmp = df_future.copy()
    tmp["fecha"] = tmp["ts"].dt.date
    tmp["tmo_num"] = tmp["tmo_hora"] * tmp["llamadas_hora"]
    d = tmp.groupby("fecha").agg(
        llamadas_totales_dia=("llamadas_hora", "sum"),
        tmo_ponderado_num=("tmo_num", "sum"),
    )
    d["tmo_promedio_diario"] = d["tmo_ponderado_num"] / (d["llamadas_totales_dia"] + 1e-6)
    # fallback si hay días con 0 llamadas
    if (d["llamadas_totales_dia"] == 0).any():
        tmo_simple = tmp.groupby("fecha")["tmo_hora"].mean().fillna(0)
        d["tmo_promedio_diario"] = d["tmo_promedio_diario"].where(d["llamadas_totales_dia"] > 0, tmo_simple).fillna(0)
    d = d.reset_index()[["fecha", "llamadas_totales_dia", "tmo_promedio_diario"]]
    d["fecha"] = d["fecha"].astype(str)
    d["llamadas_totales_dia"] = d["llamadas_totales_dia"].astype(int)
    out_d = os.path.join(PUBLIC_DIR, "Predicion_daria.json")
    d.to_json(out_d, orient="records", indent=2, force_ascii=False)
    print(f"  OK -> {out_d}")

    # 9) Alertas (si el módulo existe)
    if gen_alertas is not None:
        print("\n[Fase 9] Alertas climáticas...")
        # Reusar simulación para sacar anomalías por comuna
        df_weather_future = simulate_future_weather(
            clima_hist_df if isinstance(clima_hist_df, pd.DataFrame) else pd.DataFrame(),
            df_future_base["ts"].min(),
            df_future_base["ts"].max(),
        )
        df_agg_anoms = process_future_climate(
            df_weather_future, baselines_clima if isinstance(baselines_clima, pd.DataFrame) else pd.DataFrame()
        )
        # riesgo por ts
        df_tmp = pd.merge(df_future_base[["ts"]], df_agg_anoms, on="ts", how="left").fillna(0)
        df_tmp["risk_proba"] = predict_risk(df_tmp, risk["model"], risk["scaler"], risk["cols"])
        df_risk_out = df_tmp[["ts", "risk_proba"]].copy()

        # si tu función gen_alertas requiere por-comuna, adapta aquí
        try:
            alertas = gen_alertas(df_weather_future, df_risk_out, proba_threshold=0.5, impact_factor=100)
            out_alert = os.path.join(PUBLIC_DIR, "alertas_climaticas.json")
            with open(out_alert, "w", encoding="utf-8") as f:
                json.dump(alertas, f, indent=2, ensure_ascii=False)
            print(f"  OK -> {out_alert}")
        except Exception as e:
            print(f"  [Adv] No se pudieron generar alertas: {e}")
    else:
        print("\n[Fase 9] Alertas deshabilitadas (módulo no disponible).")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETADO CON ÉXITO")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia IA: llamadas + TMO autoregresivo.")
    parser.add_argument("--horizonte", type=int, default=120, help="Horizonte en días.")
    args = parser.parse_args()
    main(args.horizonte)


