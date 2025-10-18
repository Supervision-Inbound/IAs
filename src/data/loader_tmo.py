# src/data/loader_tmo.py
import os
import pandas as pd
import numpy as np

TZ = "America/Santiago"

# Nombres aceptados (case-insensitive)
CANDIDATE_FILENAMES = [
    "HISTORICO_TMO.csv",
    "TMO_HISTORICO.csv",
    "tmo_historico.csv",
    "historico_tmo.csv",
]

def _read_csv_robust(path):
    # intentos con diferentes separadores/comas
    seps = [",", ";", "|", "\t"]
    last_err = None
    for sep in seps:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] >= 3:
                return df
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise ValueError(f"No pude leer {path} con separadores estándar.")

def _pick_col(cols, candidates):
    m = {c.lower().strip(): c for c in cols}
    for c in candidates:
        key = c.lower().strip()
        if key in m:
            return m[key]
    return None

def _ensure_ts(df, fecha_cands=("fecha","date"), hora_cands=("hora","hour")):
    c_fecha = _pick_col(df.columns, fecha_cands)
    c_hora  = _pick_col(df.columns, hora_cands)
    if c_fecha is None or c_hora is None:
        raise ValueError("HISTORICO_TMO: faltan columnas de fecha/hora.")
    # normalizar hora a HH:MM
    h = df[c_hora].astype(str).str.slice(0,5)
    ts = pd.to_datetime(df[c_fecha].astype(str) + " " + h, dayfirst=True, errors="coerce")
    df = df.assign(ts=ts).dropna(subset=["ts"]).sort_values("ts")
    try:
        df["ts"] = df["ts"].dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
        df = df.dropna(subset=["ts"])
    except Exception:
        # ya tenía tz o hubo casos de DST raros; igual seguimos sin tz
        pass
    return df.set_index("ts")

def load_tmo_historico(data_dir="data"):
    """
    Carga el archivo de TMO histórico y devuelve un DataFrame indexado por ts con columnas:
      - q_llamadas_comercial
      - q_llamadas_tecnico
      - tmo_comercial
      - tmo_tecnico
      - tmo_general (si no existe, se calcula ponderado)
    Acepta varios nombres de archivo (case-insensitive) en la carpeta data.
    """
    # localizar archivo
    files = os.listdir(data_dir) if os.path.isdir(data_dir) else []
    tmo_file = None
    for f in files:
        for cand in CANDIDATE_FILENAMES:
            if f.lower() == cand.lower():
                tmo_file = os.path.join(data_dir, f)
                break
        if tmo_file:
            break
    if tmo_file is None:
        raise FileNotFoundError(
            f"No encontré archivo TMO histórico en {data_dir}. "
            f"Busca alguno de: {', '.join(CANDIDATE_FILENAMES)}"
        )

    df_raw = _read_csv_robust(tmo_file)
    # mapear columnas
    cols = [c.strip() for c in df_raw.columns]
    df_raw.columns = cols

    col_q_com = _pick_col(cols, ["q_llamadas_comercial","llamadas_comercial","q_comercial","q_com","q_llamadas_com"])
    col_q_tec = _pick_col(cols, ["q_llamadas_tecnico","llamadas_tecnico","q_tecnico","q_tec","q_llamadas_tec"])
    col_tmo_com = _pick_col(cols, ["tmo_comercial","tmo_com","aht_comercial","aht_com","tmo_com_seg"])
    col_tmo_tec = _pick_col(cols, ["tmo_tecnico","tmo_tec","aht_tecnico","aht_tec","tmo_tec_seg"])
    col_tmo_gen = _pick_col(cols, ["tmo_general","tmo","aht","tmo_seg","tmo (segundos)"])

    # mínimo indispensable: cantidades por tipo + tmo por tipo o general
    if col_q_com is None or col_q_tec is None:
        raise ValueError("HISTORICO_TMO: faltan cantidades por tipo (comercial/técnico).")

    df = df_raw.copy()
    # asegurar numéricos
    for c in [col_q_com, col_q_tec, col_tmo_com, col_tmo_tec, col_tmo_gen]:
        if c is not None and c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")

    # timestamp
    df = _ensure_ts(df)

    # renombrar estándar
    rename_map = {}
    rename_map[col_q_com] = "q_llamadas_comercial"
    rename_map[col_q_tec] = "q_llamadas_tecnico"
    if col_tmo_com: rename_map[col_tmo_com] = "tmo_comercial"
    if col_tmo_tec: rename_map[col_tmo_tec] = "tmo_tecnico"
    if col_tmo_gen: rename_map[col_tmo_gen] = "tmo_general"

    df = df.rename(columns=rename_map)

    # Calcular tmo_general si falta y tenemos por tipo
    if "tmo_general" not in df.columns:
        if "tmo_comercial" in df.columns and "tmo_tecnico" in df.columns:
            calls_tot = (df["q_llamadas_comercial"].fillna(0) + df["q_llamadas_tecnico"].fillna(0)).replace(0, np.nan)
            wsum = (df["q_llamadas_comercial"].fillna(0) * df["tmo_comercial"].fillna(0)) + \
                   (df["q_llamadas_tecnico"].fillna(0)   * df["tmo_tecnico"].fillna(0))
            df["tmo_general"] = (wsum / calls_tot).fillna(method="ffill").fillna(method="bfill")
        else:
            # último recurso: constante
            df["tmo_general"] = 180.0

    # proporciones
    total = (df["q_llamadas_comercial"].fillna(0) + df["q_llamadas_tecnico"].fillna(0)).replace(0, np.nan)
    df["proporcion_comercial"] = (df["q_llamadas_comercial"] / total).fillna(0.5).clip(0,1)
    df["proporcion_tecnica"]   = (df["q_llamadas_tecnico"]   / total).fillna(0.5).clip(0,1)

    # limpieza final
    keep = [
        "q_llamadas_comercial",
        "q_llamadas_tecnico",
        "proporcion_comercial",
        "proporcion_tecnica",
        "tmo_comercial",
        "tmo_tecnico",
        "tmo_general",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].sort_index()

