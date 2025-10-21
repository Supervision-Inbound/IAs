# src/inferencia/tmo_from_historico.py
import os
import numpy as np
import pandas as pd

TIMEZONE = "America/Santiago"

# Columnas esperadas en HISTORICO_TMO.csv (robusto a faltantes)
COL_Q_COM   = "q_llamadas_comercial"
COL_Q_TEC   = "q_llamadas_tecnico"
COL_Q_TOT   = "q_llamadas_general"
COL_TMO_COM = "tmo_comercial"
COL_TMO_TEC = "tmo_tecnico"
COL_TMO     = "tmo_general"   # segundos


def _ensure_ts_idx(df: pd.DataFrame) -> pd.DataFrame:
    # Busca columna 'ts' (o variantes comunes) y la pone como índice con tz
    candidates = [c for c in df.columns if c.lower().strip() in ("ts", "fecha_hora", "datetime", "datatime")]
    if not candidates:
        # Alternativa: (fecha, hora)
        cols = {c.lower().strip(): c for c in df.columns}
        fcol = next((cols[k] for k in ("fecha","date","dia","día") if k in cols), None)
        hcol = next((cols[k] for k in ("hora","hour","h") if k in cols), None)
        if not fcol or not hcol:
            raise ValueError("No se encontró columna de timestamp (ts/fecha_hora/datetime) en HISTORICO_TMO.csv")
        hora = df[hcol].astype(str).str.strip().str.replace('.', ':', regex=False).str.slice(0, 5)
        ts = pd.to_datetime(df[fcol].astype(str) + " " + hora, errors="coerce", dayfirst=True, format="%Y-%m-%d %H:%M")
    else:
        ts = pd.to_datetime(df[candidates[0]], errors="coerce", dayfirst=True)

    df = df.loc[ts.notna()].copy()
    idx = ts[ts.notna()]
    if getattr(idx, "tz", None) is None:
        idx = idx.dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
    else:
        idx = idx.dt.tz_convert(TIMEZONE)
    df.index = idx
    df = df.dropna(subset=[df.index.name]).sort_index()
    return df


def _coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _compute_tmo_general_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    # Si no hay tmo_general, calcúlalo ponderado por cantidades de llamadas:
    # tmo_general = (tmo_comercial * q_com + tmo_tecnico * q_tec) / q_tot
    if COL_TMO not in df.columns:
        needed = [COL_TMO_COM, COL_TMO_TEC, COL_Q_COM, COL_Q_TEC, COL_Q_TOT]
        if not all(c in df.columns for c in needed):
            df[COL_TMO] = np.nan
            return df
        num = (df[COL_TMO_COM].astype(float) * df[COL_Q_COM].astype(float)
             + df[COL_TMO_TEC].astype(float) * df[COL_Q_TEC].astype(float))
        den = df[COL_Q_TOT].astype(float).replace(0, np.nan)
        df[COL_TMO] = (num / den).astype(float)
    return df


def _add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["dow"]  = d.index.dayofweek
    d["hour"] = d.index.hour
    return d


def _robust_hourly_profile(df_tmo: pd.DataFrame, lookback_days: int = 90) -> pd.DataFrame:
    """
    Perfil robusto de TMO por (dow,hour) usando MEDIANA de los últimos N días.
    """
    if df_tmo.empty:
        raise ValueError("HISTORICO_TMO.csv está vacío tras parsing.")

    last_ts = df_tmo.index.max()
    start = last_ts - pd.Timedelta(days=lookback_days)
    df_lb = df_tmo.loc[df_tmo.index >= start].copy()
    if df_lb.empty:
        df_lb = df_tmo.copy()

    df_lb = df_lb.dropna(subset=[COL_TMO])
    if df_lb.empty:
        return pd.DataFrame({"dow":[], "hour":[], "tmo_mediana":[]})

    d = _add_time_parts(df_lb[[COL_TMO]])
    prof = d.groupby(["dow","hour"])[COL_TMO].median().rename("tmo_mediana").reset_index()
    return prof


def _map_profile_to_future(profile: pd.DataFrame, future_idx: pd.DatetimeIndex) -> pd.Series:
    mm = pd.DataFrame({"ts": future_idx})
    mm["dow"] = mm["ts"].dayofweek
    mm["hour"] = mm["ts"].hour
    out = mm.merge(profile, on=["dow","hour"], how="left")
    # Fallback a mediana global si falta
    if out["tmo_mediana"].isna().any():
        global_med = out["tmo_mediana"].median(skipna=True)
        out["tmo_mediana"] = out["tmo_mediana"].fillna(global_med if np.isfinite(global_med) else 0.0)
    s = pd.Series(out["tmo_mediana"].values, index=future_idx, name="tmo_s")
    # Limpiar rango (0–4 horas)
    s = s.clip(lower=0, upper=4*3600).fillna(0.0).astype(float)
    return s


def forecast_tmo_from_historico(
    historico_path: str,
    future_idx: pd.DatetimeIndex,
    lookback_days: int = 90
) -> pd.Series:
    """
    Lee data/HISTORICO_TMO.csv y construye TMO por hora para el horizonte futuro
    usando mediana por (dow,hour) de últimos N días.
    - No depende del archivo de llamadas.
    - Si falta tmo_general, lo calcula como ponderado por cantidades comercial/técnico.
    """
    if not os.path.exists(historico_path):
        return pd.Series(0.0, index=future_idx, name="tmo_s")

    df = pd.read_csv(historico_path, low_memory=False)
    df = _ensure_ts_idx(df)
    df = _coerce_numeric(df, [COL_Q_COM, COL_Q_TEC, COL_Q_TOT, COL_TMO_COM, COL_TMO_TEC, COL_TMO])
    df = _compute_tmo_general_if_missing(df)

    profile = _robust_hourly_profile(df, lookback_days=lookback_days)
    if profile.empty:
        return pd.Series(0.0, index=future_idx, name="tmo_s")

    return _map_profile_to_future(profile, future_idx)
