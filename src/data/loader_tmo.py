# src/data/loader_tmo.py
import os
import pandas as pd
import numpy as np

TZ = "America/Santiago"

# Aceptamos estos nombres si nos pasan un directorio en vez de archivo
CANDIDATE_FILENAMES = [
    "HISTORICO_TMO.csv",
    "TMO_HISTORICO.csv",
    "tmo_historico.csv",
    "historico_tmo.csv",
]

def _read_csv_robust(path: str) -> pd.DataFrame:
    """Intenta leer un CSV probando separadores comunes."""
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
    """Devuelve el nombre de columna real (respetando mayúsculas/minúsculas) que coincida con alguna candidata."""
    m = {c.lower().strip(): c for c in cols}
    for c in candidates:
        key = c.lower().strip()
        if key in m:
            return m[key]
    return None

def _ensure_ts(df, fecha_cands=("fecha","date"), hora_cands=("hora","hour")):
    """Crea índice 'ts' con tz si es posible."""
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
        # ya traía tz o hubo DST raro; seguimos sin tz explícita
        pass
    return df.set_index("ts")

def _find_file(path_or_dir: str) -> str:
    """Si es archivo, lo devuelve; si es carpeta, busca candidatos."""
    if os.path.isfile(path_or_dir):
        return path_or_dir
    if os.path.isdir(path_or_dir):
        for f in os.listdir(path_or_dir):
            for cand in CANDIDATE_FILENAMES:
                if f.lower() == cand.lower():
                    return os.path.join(path_or_dir, f)
    raise FileNotFoundError(
        f"No encontré archivo TMO histórico en {path_or_dir}. "
        f"Busca alguno de: {', '.join(CANDIDATE_FILENAMES)}"
    )

def load_historico_tmo(path_or_dir: str = "data") -> pd.DataFrame:
    """
    Carga el histórico de TMO (archivo o carpeta) y retorna un DataFrame indexado por ts con columnas:
      - q_llamadas_comercial
      - q_llamadas_tecnico
      - proporcion_comercial
      - proporcion_tecnica
      - tmo_comercial
      - tmo_tecnico
      - tmo_general  (si falta, se calcula ponderado por tipo)
    """
    tmo_file = _find_file(path_or_dir)
    df_raw = _read_csv_robust(tmo_file)

    # normalizar nombres
    df_raw.columns = [c.strip() for c in df_raw.columns]
    cols = df_raw.columns

    # mapeos flexibles
    col_q_com  = _pick_col(cols, ["q_llamadas_comercial","llamadas_comercial","q_comercial","q_com","q_llamadas_com"])
    col_q_tec  = _pick_col(cols, ["q_llamadas_tecnico","llamadas_tecnico","q_tecnico","q_tec","q_llamadas_tec"])
    col_tmo_com = _pick_col(cols, ["tmo_comercial","tmo_com","aht_comercial","aht_com","tmo_com_seg"])
    col_tmo_tec = _pick_col(cols, ["tmo_tecnico","tmo_tec","aht_tecnico","aht_tec","tmo_tec_seg"])
    col_tmo_gen = _pick_col(cols, ["tmo_general","tmo","aht","tmo_seg","tmo (segundos)"])

    if col_q_com is None or col_q_tec is None:
        raise ValueError("HISTORICO_TMO: faltan cantidades por tipo (comercial/técnico).")

    df = df_raw.copy()

    # a numéricos (manejo de comas)
    to_numeric_cols = [c for c in [col_q_com, col_q_tec, col_tmo_com, col_tmo_tec, col_tmo_gen] if c]
    for c in to_numeric_cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")

    # timestamp
    df = _ensure_ts(df)

    # renombrar estándar
    rename_map = {col_q_com: "q_llamadas_comercial", col_q_tec: "q_llamadas_tecnico"}
    if col_tmo_com: rename_map[col_tmo_com] = "tmo_comercial"
    if col_tmo_tec: rename_map[col_tmo_tec] = "tmo_tecnico"
    if col_tmo_gen: rename_map[col_tmo_gen] = "tmo_general"
    df = df.rename(columns=rename_map)

    # calcular tmo_general si falta y tenemos por tipo
    if "tmo_general" not in df.columns:
        if "tmo_comercial" in df.columns and "tmo_tecnico" in df.columns:
            calls_tot = (df["q_llamadas_comercial"].fillna(0) + df["q_llamadas_tecnico"].fillna(0)).replace(0, np.nan)
            wsum = (df["q_llamadas_comercial"].fillna(0) * df["tmo_comercial"].fillna(0)) + \
                   (df["q_llamadas_tecnico"].fillna(0)   * df["tmo_tecnico"].fillna(0))
            df["tmo_general"] = (wsum / calls_tot).replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")
        else:
            df["tmo_general"] = 180.0

    # proporciones
    total = (df["q_llamadas_comercial"].fillna(0) + df["q_llamadas_tecnico"].fillna(0)).replace(0, np.nan)
    df["proporcion_comercial"] = (df["q_llamadas_comercial"] / total).fillna(0.5).clip(0,1)
    df["proporcion_tecnica"]   = (df["q_llamadas_tecnico"]   / total).fillna(0.5).clip(0,1)

    # salida ordenada
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

# Alias para compatibilidad con nombre anterior
load_tmo_historico = load_historico_tmo
