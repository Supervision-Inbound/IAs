# src/inferencia/tmo_from_historico.py
import os
import numpy as np
import pandas as pd

TIMEZONE = "America/Santiago"

COL_Q_COM   = "q_llamadas_comercial"
COL_Q_TEC   = "q_llamadas_tecnico"
COL_Q_TOT   = "q_llamadas_general"
COL_TMO_COM = "tmo_comercial"
COL_TMO_TEC = "tmo_tecnico"
COL_TMO     = "tmo_general"

def _ensure_ts_idx(df: pd.DataFrame) -> pd.DataFrame:
    candidates = [c for c in df.columns if c.lower().strip() in ("ts","fecha_hora","datetime","datatime")]
    if not candidates:
        cols = {c.lower().strip(): c for c in df.columns}
        fcol = next((cols[k] for k in ("fecha","date","dia","día") if k in cols), None)
        hcol = next((cols[k] for k in ("hora","hour","h") if k in cols), None)
        if not fcol or not hcol:
            raise ValueError("HISTORICO_TMO.csv: no hay ts ni (fecha+hora)")
        # armar hora robusta
        hora_raw = df[hcol].astype(str).str.strip().str.replace(".", ":", regex=False)
        parts = hora_raw.str.extract(r'^\s*(\d{1,2})(?::\s*(\d{1,2}))?(?::\s*(\d{1,2}))?')
        hh = pd.to_numeric(parts[0], errors='coerce').clip(0,23).fillna(0).astype(int)
        mm = pd.to_numeric(parts[1], errors='coerce').clip(0,59).fillna(0).astype(int)
        ss = pd.to_numeric(parts[2], errors='coerce').clip(0,59).fillna(0).astype(int)
        hora_str = hh.map(lambda x: f"{x:02d}") + ":" + mm.map(lambda x: f"{x:02d}") + ":" + ss.map(lambda x: f"{x:02d}")
        ts = pd.to_datetime(pd.to_datetime(df[fcol].astype(str), dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
                            + " " + hora_str, errors="coerce", infer_datetime_format=True)
    else:
        ts = pd.to_datetime(df[candidates[0]], errors="coerce", dayfirst=True)

    df = df.loc[ts.notna()].copy()
    idx = ts[ts.notna()]
    if getattr(idx, "tz", None) is None:
        idx = idx.dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
    else:
        idx = idx.dt.tz_convert(TIMEZONE)
    df.index = idx
    return df.sort_index()

def _coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _compute_tmo_general_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    if COL_TMO not in df.columns:
        needed = [COL_TMO_COM, COL_TMO_TEC, COL_Q_COM, COL_Q_TEC, COL_Q_TOT]
        if not all(c in df.columns for c in needed):
            df[COL_TMO] = np.nan
            return df
        num = (df[COL_TMO_COM].astype(float) * df[COL_Q_COM].astype(float)
             + df[COL_TMO_TEC].astype(float) * df[COL_Q_TEC].astype(float))
        den = df[COL_Q_TOT].astype(float).replace(0, np.nan)
        df[COL_TMO] = (num/den).astype(float)
    return df

def _add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["dow"]  = d.index.dayofweek
    d["hour"] = d.index.hour
    return d

def _robust_hourly_profile(df_tmo: pd.DataFrame, lookback_days: int = 90) -> pd.DataFrame:
    if df_tmo.empty:
        raise ValueError("HISTORICO_TMO.csv vacío tras parsing.")
    last_ts = df_tmo.index.max()
    df_lb = df_tmo.loc[df_tmo.index >= last_ts - pd.Timedelta(days=lookback_days)].copy()
    if df_lb.empty:
        df_lb = df_tmo.copy()
    df_lb = df_lb.dropna(subset=[COL_TMO])
    if df_lb.empty:
        return pd.DataFrame({"dow":[], "hour":[], "tmo_mediana":[]})
    d = _add_time_parts(df_lb[[COL_TMO]])
    return d.groupby(["dow","hour"])[COL_TMO].median().rename("tmo_mediana").reset_index()

def _map_profile_to_future(profile: pd.DataFrame, future_idx: pd.DatetimeIndex) -> pd.Series:
    mm = pd.DataFrame({"ts": future_idx})
    mm["dow"] = mm["ts"].dayofweek
    mm["hour"] = mm["ts"].hour
    out = mm.merge(profile, on=["dow","hour"], how="left")
    if out["tmo_mediana"].isna().any():
        gm = out["tmo_mediana"].median(skipna=True)
        out["tmo_mediana"] = out["tmo_mediana"].fillna(gm if np.isfinite(gm) else 0.0)
    s = pd.Series(out["tmo_mediana"].values, index=future_idx, name="tmo_s")
    return s.clip(lower=0, upper=4*3600).fillna(0.0).astype(float)

def forecast_tmo_from_historico(historico_path: str,
                                future_idx: pd.DatetimeIndex,
                                lookback_days: int = 90) -> pd.Series:
    """
    Genera TMO horario futuro exclusivamente desde data/HISTORICO_TMO.csv,
    usando mediana por (dow,hour) de últimos N días (perfil robusto).
    """
    if not os.path.exists(historico_path):
        return pd.Series(0.0, index=future_idx, name="tmo_s")
    df = pd.read_csv(historico_path, low_memory=False)
    df = _ensure_ts_idx(df)
    df = _coerce_numeric(df, [COL_Q_COM, COL_Q_TEC, COL_Q_TOT, COL_TMO_COM, COL_TMO_TEC, COL_TMO])
    df = _compute_tmo_general_if_missing(df)
    prof = _robust_hourly_profile(df, lookback_days=lookback_days)
    if prof.empty:
        return pd.Series(0.0, index=future_idx, name="tmo_s")
    return _map_profile_to_future(prof, future_idx)

