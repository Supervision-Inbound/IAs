# src/data/loader_tmo.py
import os
import numpy as np
import pandas as pd

TZ = "America/Santiago"

# --- utilidades robustas ---
def _smart_read_csv(path: str) -> pd.DataFrame:
    # intenta csv normal, luego con ';'
    try:
        df = pd.read_csv(path, low_memory=False)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    return pd.read_csv(path, delimiter=';', low_memory=False)

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
    """acepta 'mm:ss', 'hh:mm:ss' o número en seg."""
    if pd.isna(val): return np.nan
    s = str(val).strip()
    s = s.replace(",", ".")
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
    # normaliza hora a HH:MM, arma tz local y ordena
    h = df[col_hora].astype(str).str.slice(0, 5)
    ts = pd.to_datetime(df[col_fecha].astype(str) + " " + h,
                        errors="coerce", dayfirst=True, format="%Y-%m-%d %H:%M")
    df = df.assign(ts=ts).dropna(subset=["ts"]).sort_values("ts")
    df["ts"] = df["ts"].dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    df = df.dropna(subset=["ts"])
    return df.set_index("ts")

# --- lector principal ---
def load_historico_tmo(path="data/HISTORICO_TMO.csv") -> pd.DataFrame:
    """
    Devuelve un DF indexado por ts (tz=America/Santiago) con columnas:
      - q_llamadas_general, q_llamadas_comercial, q_llamadas_tecnico
      - proporcion_comercial, proporcion_tecnica
      - tmo_comercial, tmo_tecnico, tmo_general (seg)
    Acepta nombres flexibles de columnas.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df0 = _smart_read_csv(path)
    df0.columns = [c.strip() for c in df0.columns]

    # columnas de fecha/hora
    c_fecha = _pick(df0.columns, ["fecha", "date", "dia", "día"])
    c_hora  = _pick(df0.columns, ["hora", "time", "hh", "h"])
    if not c_fecha or not c_hora:
        raise ValueError("HISTORICO_TMO.csv debe tener columnas de fecha y hora.")

    # cantidades por tipo y total
    c_q_gen = _pick(df0.columns, ["q_llamadas_general", "llamadas_general", "recibidos_nacional", "llamadas", "total_llamadas"])
    c_q_com = _pick(df0.columns, ["q_llamadas_comercial", "llamadas_comercial", "comercial"])
    c_q_tec = _pick(df0.columns, ["q_llamadas_tecnico", "llamadas_tecnico", "tecnico", "técnico"])

    # tmos por tipo
    c_tmo_com = _pick(df0.columns, ["tmo_comercial", "aht_comercial", "tmo_com", "aht_com"])
    c_tmo_tec = _pick(df0.columns, ["tmo_tecnico", "aht_tecnico", "tmo_tec", "aht_tec"])
    c_tmo_gen = _pick(df0.columns, ["tmo_general", "tmo (segundos)", "tmo_seg", "aht"])

    df = _ensure_ts(df0, c_fecha, c_hora)

    # numeric clean
    for c in [c_q_gen, c_q_com, c_q_tec]:
        if c and c in df.columns: df[c] = df[c].apply(_to_num)
    for c in [c_tmo_com, c_tmo_tec, c_tmo_gen]:
        if c and c in df.columns: df[c] = df[c].apply(_parse_tmo)

    # completa total si falta
    if not c_q_gen:
        # si hay desgloses, suma
        if c_q_com and c_q_tec:
            df["q_llamadas_general"] = df[c_q_com].fillna(0) + df[c_q_tec].fillna(0)
            c_q_gen = "q_llamadas_general"
        else:
            raise ValueError("No encuentro total de llamadas ni desgloses en HISTORICO_TMO.csv")

    # si faltan desgloses, reparte 50/50 para proporciones
    if not c_q_com:
        df["q_llamadas_comercial"] = df[c_q_gen] * 0.5
        c_q_com = "q_llamadas_comercial"
    if not c_q_tec:
        df["q_llamadas_tecnico"] = df[c_q_gen] * 0.5
        c_q_tec = "q_llamadas_tecnico"

    # proporciones
    den = df[c_q_gen].replace(0, np.nan)
    df["proporcion_comercial"] = (df[c_q_com] / den).clip(0, 1).fillna(0.5)
    df["proporcion_tecnica"]   = (df[c_q_tec] / den).clip(0, 1).fillna(0.5)

    # TMO por tipo (si no están, usa tmo_general como proxy)
    if not c_tmo_com and c_tmo_gen:
        df["tmo_comercial"] = df[c_tmo_gen]
        c_tmo_com = "tmo_comercial"
    if not c_tmo_tec and c_tmo_gen:
        df["tmo_tecnico"] = df[c_tmo_gen]
        c_tmo_tec = "tmo_tecnico"

    # TMO general ponderado
    if c_tmo_gen:
        df["tmo_general"] = df[c_tmo_gen]
    else:
        # pondera por llamadas (no proporción) para ser consistente
        num = df[c_q_com].fillna(0) * df[c_tmo_com].fillna(0) + df[c_q_tec].fillna(0) * df[c_tmo_tec].fillna(0)
        den = df[c_q_com].fillna(0) + df[c_q_tec].fillna(0)
        df["tmo_general"] = (num / den.replace(0, np.nan)).fillna(0)
    # salida compacta
    out = df[[
        c_q_gen, c_q_com, c_q_tec,
        "proporcion_comercial", "proporcion_tecnica",
        c_tmo_com if c_tmo_com else "tmo_comercial",
        c_tmo_tec if c_tmo_tec else "tmo_tecnico",
        "tmo_general"
    ]].copy()

    # nombres estándar
    out = out.rename(columns={
        c_q_gen: "q_llamadas_general",
        c_q_com: "q_llamadas_comercial",
        c_q_tec: "q_llamadas_tecnico",
        (c_tmo_com if c_tmo_com else "tmo_comercial"): "tmo_comercial",
        (c_tmo_tec if c_tmo_tec else "tmo_tecnico"): "tmo_tecnico",
    })
    return out
