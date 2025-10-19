# src/data/loader_tmo.py
import os
import numpy as np
import pandas as pd

TZ = "America/Santiago"

# ----------------- utilidades -----------------
def _smart_read_csv(path: str) -> pd.DataFrame:
    # Tu archivo suele venir con ';'
    try:
        df = pd.read_csv(path, sep=';', low_memory=False)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    # Fallback: coma
    return pd.read_csv(path, low_memory=False)

def _pick(cols, candidates):
    m = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        key = cand.lower().strip()
        if key in m:
            return m[key]
    return None

def _to_num(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

def _parse_tmo(val):
    """Acepta 'mm:ss', 'hh:mm:ss' o número en seg."""
    if pd.isna(val): return np.nan
    s = str(val).strip().replace(",", ".")
    parts = s.split(":")
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(s)
    except Exception:
        return np.nan

def _ensure_ts(df, col_fecha, col_hora):
    # Hora -> HH:MM (si venía 0:00:00)
    h = df[col_hora].astype(str).str.slice(0, 5)
    # Parse flexible: dayfirst SÍ, sin format estricto
    ts = pd.to_datetime(
        df[col_fecha].astype(str) + " " + h,
        dayfirst=True, errors="coerce"
    )
    df = df.assign(ts=ts).dropna(subset=["ts"])
    # Alinear a la hora en punto y localizar TZ
    df["ts"] = df["ts"].dt.floor("h").dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    return df.set_index("ts")

# ----------------- lector principal -----------------
def load_historico_tmo(path="data/TMO_HISTORICO.csv") -> pd.DataFrame:
    """
    Devuelve DF horario (index ts) con columnas:
      q_llamadas_general, q_llamadas_comercial, q_llamadas_tecnico,
      proporcion_comercial, proporcion_tecnica,
      tmo_comercial, tmo_tecnico, tmo_general (seg)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df0 = _smart_read_csv(path)
    df0.columns = [c.strip() for c in df0.columns]

    # columnas de fecha/hora
    c_fecha = _pick(df0.columns, ["fecha", "date", "dia", "día"])
    c_hora  = _pick(df0.columns, ["hora", "time", "hh", "h"])
    if not c_fecha or not c_hora:
        raise ValueError("TMO_HISTORICO.csv debe tener columnas de fecha y hora.")

    # cantidades por tipo y total
    c_q_gen = _pick(df0.columns, ["q_llamadas_general", "llamadas_general", "recibidos_nacional", "llamadas", "total_llamadas"])
    c_q_com = _pick(df0.columns, ["q_llamadas_comercial", "llamadas_comercial", "comercial"])
    c_q_tec = _pick(df0.columns, ["q_llamadas_tecnico", "llamadas_tecnico", "tecnico", "técnico"])

    # tmos por tipo y general
    c_tmo_com = _pick(df0.columns, ["tmo_comercial", "aht_comercial", "tmo_com", "aht_com"])
    c_tmo_tec = _pick(df0.columns, ["tmo_tecnico", "aht_tecnico", "tmo_tec", "aht_tec"])
    c_tmo_gen = _pick(df0.columns, ["tmo_general", "tmo (segundos)", "tmo_seg", "aht"])

    df = _ensure_ts(df0, c_fecha, c_hora)

    # Numeric clean
    for c in [c_q_gen, c_q_com, c_q_tec]:
        if c and c in df.columns: df[c] = df[c].apply(_to_num)
    for c in [c_tmo_com, c_tmo_tec, c_tmo_gen]:
        if c and c in df.columns: df[c] = df[c].apply(_parse_tmo)

    # Agregar por hora (si hay múltiples filas en la misma hora)
    agg = {}
    if c_q_gen: agg[c_q_gen] = "sum"
    if c_q_com: agg[c_q_com] = "sum"
    if c_q_tec: agg[c_q_tec] = "sum"
    if c_tmo_com: agg[c_tmo_com] = "mean"
    if c_tmo_tec: agg[c_tmo_tec] = "mean"
    if c_tmo_gen: agg[c_tmo_gen] = "mean"
    if agg:
        df = df.groupby("ts").agg(agg)

    # Completar totales y desgloses
    if not c_q_gen:
        if c_q_com and c_q_tec:
            df["q_llamadas_general"] = df[c_q_com].fillna(0) + df[c_q_tec].fillna(0)
            c_q_gen = "q_llamadas_general"
        else:
            raise ValueError("No encuentro total de llamadas ni desgloses en TMO_HISTORICO.csv")
    if not c_q_com:
        df["q_llamadas_comercial"] = df[c_q_gen] * 0.5; c_q_com = "q_llamadas_comercial"
    if not c_q_tec:
        df["q_llamadas_tecnico"]   = df[c_q_gen] * 0.5; c_q_tec = "q_llamadas_tecnico"

    # Proporciones
    den = df[c_q_gen].replace(0, np.nan)
    df["proporcion_comercial"] = (df[c_q_com] / den).clip(0, 1).fillna(0.5)
    df["proporcion_tecnica"]   = (df[c_q_tec] / den).clip(0, 1).fillna(0.5)

    # TMO por tipo y general
    if not c_tmo_com and c_tmo_gen:
        df["tmo_comercial"] = df[c_tmo_gen]; c_tmo_com = "tmo_comercial"
    if not c_tmo_tec and c_tmo_gen:
        df["tmo_tecnico"]   = df[c_tmo_gen]; c_tmo_tec = "tmo_tecnico"

    if c_tmo_gen:
        df["tmo_general"] = df[c_tmo_gen]
    else:
        num = df[c_q_com].fillna(0)*df[c_tmo_com].fillna(0) + df[c_q_tec].fillna(0)*df[c_tmo_tec].fillna(0)
        den = df[c_q_com].fillna(0) + df[c_q_tec].fillna(0)
        df["tmo_general"] = (num / den.replace(0, np.nan)).fillna(0)

    out = df.rename(columns={
        c_q_gen: "q_llamadas_general",
        c_q_com: "q_llamadas_comercial",
        c_q_tec: "q_llamadas_tecnico",
        c_tmo_com if c_tmo_com else "tmo_comercial": "tmo_comercial",
        c_tmo_tec if c_tmo_tec else "tmo_tecnico":   "tmo_tecnico",
    })[[
        "q_llamadas_general","q_llamadas_comercial","q_llamadas_tecnico",
        "proporcion_comercial","proporcion_tecnica",
        "tmo_comercial","tmo_tecnico","tmo_general"
    ]]

    return out
