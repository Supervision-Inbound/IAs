# src/inferencia/features.py
import pandas as pd
import numpy as np

TIMEZONE = "America/Santiago"

def _col(name: str) -> str:
    return name

def ensure_ts(d: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve DF indexado por ts (tz=America/Santiago), ordenado.
    Acepta formatos de hora heterogéneos: "8", "8:0", "08:00", "8.00", "08:00:00", etc.
    Soporta (ts) o (fecha + hora). No descarta filas por hora "imperfecta".
    """
    d = d.copy()
    # Normaliza nombres para detectar columnas conocidas
    lowmap = {c.lower().strip(): c for c in d.columns}
    # 1) Caso: columna ts directa
    for cand in ("ts", "fecha_hora", "datetime", "datatime"):
        if cand in lowmap:
            ts_col = lowmap[cand]
            ts = pd.to_datetime(d[ts_col], errors="coerce", dayfirst=True)
            d = d.loc[ts.notna()].copy()
            ts = ts[ts.notna()]
            if getattr(ts, "tz", None) is None:
                ts = ts.dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
            else:
                ts = ts.dt.tz_convert(TIMEZONE)
            d.index = ts
            return d.sort_index()

    # 2) Caso: fecha + hora
    fcol = next((lowmap[k] for k in ("fecha", "date", "dia", "día") if k in lowmap), None)
    hcol = next((lowmap[k] for k in ("hora", "hour", "h") if k in lowmap), None)
    if fcol is None or hcol is None:
        raise ValueError("No se encontró ni 'ts' ni (fecha+hora) para construir el índice temporal.")

    # Fecha
    fecha_dt = pd.to_datetime(d[fcol].astype(str), errors="coerce", dayfirst=True)

    # Hora robusta
    hora_raw = d[hcol].astype(str).str.strip()
    tmp = hora_raw.str.replace(".", ":", regex=False)        # 8.0 -> 8:0
    parts = tmp.str.extract(r'^\s*(\d{1,2})(?::\s*(\d{1,2}))?(?::\s*(\d{1,2}))?')
    hh = pd.to_numeric(parts[0], errors='coerce').clip(lower=0, upper=23).fillna(0).astype(int)
    mm = pd.to_numeric(parts[1], errors='coerce').clip(lower=0, upper=59).fillna(0).astype(int)
    # segundos (opcional)
    ss = pd.to_numeric(parts[2], errors='coerce').clip(lower=0, upper=59).fillna(0).astype(int)

    hora_str = hh.map(lambda x: f"{x:02d}") + ":" + mm.map(lambda x: f"{x:02d}") + ":" + ss.map(lambda x: f"{x:02d}")
    ts = pd.to_datetime(fecha_dt.dt.strftime("%Y-%m-%d") + " " + hora_str,
                        errors="coerce", infer_datetime_format=True)
    ts = ts.dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")

    d = d.loc[ts.notna()].copy()
    d.index = ts[ts.notna()]
    return d.sort_index()

def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dow"] = out.index.dayofweek
    out["month"] = out.index.month
    out["hour"] = out.index.hour
    out["day"] = out.index.day
    out["sin_hour"] = np.sin(2*np.pi*out["hour"]/24)
    out["cos_hour"] = np.cos(2*np.pi*out["hour"]/24)
    out["sin_dow"] = np.sin(2*np.pi*out["dow"]/7)
    out["cos_dow"] = np.cos(2*np.pi*out["dow"]/7)
    return out

def add_lags_mas(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    for lag in [24, 48, 72, 168]:
        out[f"lag_{lag}"] = out[col].shift(lag)
    for w in [24, 72, 168]:
        out[f"ma_{w}"] = out[col].rolling(w, min_periods=1).mean()
    return out

def dummies_and_reindex(df: pd.DataFrame, cols_expected: list) -> pd.DataFrame:
    x = pd.get_dummies(df.copy(), columns=[c for c in ["dow","month","hour"] if c in df.columns])
    x = x.reindex(columns=cols_expected, fill_value=0)
    # Forzar numérico
    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
    return x
