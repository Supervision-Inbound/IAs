# src/inferencia/tmo_from_historico.py
import os
import numpy as np
import pandas as pd

TIMEZONE = "America/Santiago"

COL_Q_COM   = "q_llamadas_comercial"
COL_Q_TEC   = "q_llamadas_tecnico"
COL_Q_TOT   = "q_llamadas_general"
COL_P_COM   = "proporcion_comercial"
COL_P_TEC   = "proporcion_tecnica"
COL_TMO_COM = "tmo_comercial"
COL_TMO_TEC = "tmo_tecnico"
COL_TMO     = "tmo_general"

def _smart_read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if df.shape[1] == 1 and df.iloc[:,0].astype(str).str.contains(";").any():
        df = pd.read_csv(path, delimiter=";", low_memory=False)
    return df

def _parse_seconds(val) -> float:
    if pd.isna(val): return np.nan
    s = str(val).strip().replace(",", ".")
    if ":" in s:
        parts = s.split(":")
        try:
            if len(parts) == 3:
                h, m, s2 = [float(x) for x in parts]
                return h*3600 + m*60 + s2
            if len(parts) == 2:
                m, s2 = [float(x) for x in parts]
                return m*60 + s2
        except:
            return np.nan
    try:
        return float(s)
    except:
        return np.nan

def _ensure_ts_idx(df: pd.DataFrame) -> pd.DataFrame:
    low = {c.lower().strip(): c for c in df.columns}
    for cand in ("ts","fecha_hora","datetime","datatime"):
        if cand in low:
            ts = pd.to_datetime(df[low[cand]], errors="coerce", dayfirst=True)
            mask = ts.notna()
            df = df.loc[mask].copy()
            ts = ts[mask]
            tz = getattr(ts.dt, "tz", None)
            ts = ts.dt.tz_localize(TIMEZONE, ambiguous="infer", nonexistent="shift_forward") if tz is None else ts.dt.tz_convert(TIMEZONE)
            df.index = ts
            return df.sort_index()
    fcol = next((low[k] for k in ("fecha","date","dia","día") if k in low), None)
    hcol = next((low[k] for k in ("hora","hour","h") if k in low), None)
    if fcol is None or hcol is None:
        raise ValueError("HISTORICO_TMO.csv: no hay 'ts' ni (fecha+hora).")
    fecha = pd.to_datetime(df[fcol].astype(str), errors="coerce", dayfirst=True)
    hora_raw = df[hcol].astype(str).str.strip().str.replace(".", ":", regex=False)
    parts = hora_raw.str.extract(r'^\s*(\d{1,2})(?::\s*(\d{1,2}))?(?::\s*(\d{1,2}))?')
    hh = pd.to_numeric(parts[0], errors='coerce').clip(0,23).fillna(0).astype(int)
    mm = pd.to_numeric(parts[1], errors='coerce').clip(0,59).fillna(0).astype(int)
    ss = pd.to_numeric(parts[2], errors='coerce').clip(0,59).fillna(0).astype(int)
    hora = hh.map(lambda x: f"{x:02d}") + ":" + mm.map(lambda x: f"{x:02d}") + ":" + ss.map(lambda x: f"{x:02d}")
    ts = pd.to_datetime(fecha.dt.strftime("%Y-%m-%d") + " " + hora, errors="coerce", infer_datetime_format=True)
    mask = ts.notna()
    df = df.loc[mask].copy()
    ts = ts[mask].dt.tz_localize(TIMEZONE, ambiguous="infer", nonexistent="shift_forward")
    df.index = ts
    return df.sort_index()

def _coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _ensure_tmo_seconds(df: pd.DataFrame):
    for c in [COL_TMO, COL_TMO_COM, COL_TMO_TEC]:
        if c in df.columns and (df[c].dtype == object or df[c].astype(str).str.contains(":").any()):
            df[c] = df[c].apply(_parse_seconds)
    return df

def _compute_tmo_general(df: pd.DataFrame) -> pd.DataFrame:
    has_general = COL_TMO in df.columns
    has_q = all(c in df.columns for c in (COL_Q_COM, COL_Q_TEC, COL_Q_TOT))
    has_p = all(c in df.columns for c in (COL_P_COM, COL_P_TEC))
    has_parts = any(c in df.columns for c in (COL_TMO_COM, COL_TMO_TEC))

    if not has_general:
        df[COL_TMO] = np.nan

    df = _ensure_tmo_seconds(df)

    if has_general and df[COL_TMO].notna().any():
        df[COL_TMO] = pd.to_numeric(df[COL_TMO], errors="coerce")
    else:
        if has_q and all(c in df.columns for c in (COL_TMO_COM, COL_TMO_TEC)):
            num = (df[COL_TMO_COM].astype(float) * df[COL_Q_COM].astype(float)) \
                + (df[COL_TMO_TEC].astype(float) * df[COL_Q_TEC].astype(float))
            den = df[COL_Q_TOT].astype(float).replace(0, np.nan)
            df[COL_TMO] = (num / den).astype(float)
        if df[COL_TMO].isna().all() and has_p and all(c in df.columns for c in (COL_TMO_COM, COL_TMO_TEC)):
            w = (pd.to_numeric(df[COL_P_COM], errors="coerce").astype(float).fillna(0.0) +
                 pd.to_numeric(df[COL_P_TEC], errors="coerce").astype(float).fillna(0.0)).replace(0, np.nan)
            num = (pd.to_numeric(df[COL_TMO_COM], errors="coerce").astype(float).fillna(np.nan) *
                   pd.to_numeric(df[COL_P_COM], errors="coerce").astype(float).fillna(0.0)) + \
                  (pd.to_numeric(df[COL_TMO_TEC], errors="coerce").astype(float).fillna(np.nan) *
                   pd.to_numeric(df[COL_P_TEC], errors="coerce").astype(float).fillna(0.0))
            df[COL_TMO] = (num / w).astype(float)
        if df[COL_TMO].isna().all() and has_parts:
            tmo_med = pd.concat([df.get(COL_TMO_COM), df.get(COL_TMO_TEC)], axis=1)
            df[COL_TMO] = tmo_med.median(axis=1, skipna=True)

    return df

def _add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["dow"]  = d.index.dayofweek
    d["hour"] = d.index.hour
    return d

def _robust_hourly_profile(df_tmo: pd.DataFrame, lookback_days: int = 90) -> pd.DataFrame:
    if df_tmo.empty:
        return pd.DataFrame({"dow":[], "hour":[], "tmo_mediana":[]})
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
    Genera TMO horario futuro desde data/HISTORICO_TMO.csv,
    robusto a delimitadores, strings de tiempo y faltantes.
    """
    if not os.path.exists(historico_path):
        return pd.Series(0.0, index=future_idx, name="tmo_s")

    df = _smart_read_csv(historico_path)
    df = _ensure_ts_idx(df)
    df = _coerce_numeric(df, [COL_Q_COM, COL_Q_TEC, COL_Q_TOT, COL_P_COM, COL_P_TEC])
    df = _compute_tmo_general(df)

    prof = _robust_hourly_profile(df, lookback_days=lookback_days)
    if prof.empty:
        gm = pd.to_numeric(df[COL_TMO], errors="coerce").median(skipna=True)
        const = float(gm) if np.isfinite(gm) else 0.0
        return pd.Series(const, index=future_idx, name="tmo_s")

    return _map_profile_to_future(prof, future_idx)

