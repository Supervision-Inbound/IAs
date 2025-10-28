# src/inferencia/features.py
import pandas as pd
import numpy as np
import re

TIMEZONE = "America/Santiago"

# ---------------- Utils para normalizar nombres ----------------
def _norm_col(s: str) -> str:
    s = re.sub(r"[\s\-\./:_]+", "_", s.strip().lower())
    s = re.sub(r"_+", "_", s)
    return s

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        if cand in norm:
            return norm[cand]
    return None

def _find_first_with(df: pd.DataFrame, substrings: list[str]) -> str | None:
    for c in df.columns:
        n = _norm_col(c)
        if any(sub in n for sub in substrings):
            return c
    return None

# ---------------- Parse de fechas/horas ----------------
def _parse_epoch_series(s: pd.Series):
    """Intenta parsear epoch (s o ms). Devuelve DatetimeIndex o None si no aplica."""
    if not pd.api.types.is_numeric_dtype(s):
        # Si son strings numéricas, intenta convertir
        try:
            s_num = pd.to_numeric(s, errors="coerce")
        except Exception:
            return None
    else:
        s_num = s

    if s_num.notna().sum() == 0:
        return None

    # Heurística: ms si valores grandes
    q50 = float(s_num.dropna().quantile(0.5))
    try:
        if q50 > 1e12:   # milisegundos
            dt = pd.to_datetime(s_num, unit="ms", utc=True, errors="coerce")
        elif q50 > 1e9:  # segundos (a veces caen aquí también)
            dt = pd.to_datetime(s_num, unit="s", utc=True, errors="coerce")
        else:
            return None
        return dt
    except Exception:
        return None

def _parse_datetime_flex(s: pd.Series):
    """
    Parser flexible:
    1) epoch (s/ms)
    2) to_datetime utc=True con dayfirst=True
    3) to_datetime utc=True con dayfirst=False
    """
    # 1) epoch
    dt = _parse_epoch_series(s)
    if dt is not None and dt.notna().sum() > 0:
        return dt

    # 2) dayfirst=True
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True, utc=True)
    if dt.notna().sum() > len(s) * 0.5:
        return dt

    # 3) dayfirst=False
    dt2 = pd.to_datetime(s, errors="coerce", dayfirst=False, utc=True)
    # Escoge el que mejor parseó
    return dt if dt.notna().sum() >= dt2.notna().sum() else dt2

def _coerce_ts_series(s: pd.Series) -> pd.Series:
    """
    Convierte una serie a datetime con tz consistente (UTC -> TIMEZONE).
    Admite múltiples formatos (incluido epoch).
    """
    dt = _parse_datetime_flex(s.astype(str))
    if dt is None:
        dt = pd.Series(pd.NaT, index=s.index)
    # Asegura tz
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize("UTC")
    return dt.dt.tz_convert(TIMEZONE)

# ---------------- TS helpers ----------------
def ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garantiza un índice datetime llamado 'ts'.

    Detecta, en este orden:
      A) Columna combinada: ['ts','timestamp','fecha_hora','fechahora','datetime','date_time','fecha_y_hora']
      B) Par columnas: una de 'fecha*' + una de 'hora*'
      C) Índice ya datetime
      D) Cualquier columna con >80% parseable a datetime (rescate)

    Todo se normaliza a UTC y luego America/Santiago.
    """
    d = df.copy()
    if d.empty:
        raise ValueError("DataFrame vacío: no se puede construir 'ts'.")

    # --- A) Columna combinada
    combined_candidates = [
        "ts", "timestamp", "fecha_hora", "fechahora", "datetime",
        "date_time", "fecha_y_hora", "fecha__hora"
    ]
    # Mapa normalizado -> original
    norm_map = {_norm_col(c): c for c in d.columns}
    comb_col = _find_col(d, combined_candidates)
    if comb_col is None:
        # busca nombres "parecidos" (ej. 'fecha y hora', 'Fecha-Hora')
        for c in d.columns:
            n = _norm_col(c)
            if n in ("fecha_hora", "fecha_y_hora", "fecha_hora_local", "fecha_hora_utc", "fechahora"):
                comb_col = c
                break

    if comb_col is not None:
        ts = _coerce_ts_series(d[comb_col])
        d["ts"] = ts
        d = d.dropna(subset=["ts"]).sort_values("ts").set_index("ts")
        return d

    # --- B) Par fecha + hora (flex)
    # Busca columna con 'fecha' y columna con 'hora'
    fecha_col = _find_first_with(d, ["fecha"])
    hora_col  = _find_first_with(d, ["hora", "time", "hr"])
    if fecha_col is not None and hora_col is not None:
        mix = d[fecha_col].astype(str).str.strip() + " " + d[hora_col].astype(str).str.strip()
        ts = _coerce_ts_series(mix)
        d["ts"] = ts
        d = d.dropna(subset=["ts"]).sort_values("ts").set_index("ts")
        return d

    # --- C) Índice ya datetime
    if isinstance(d.index, pd.DatetimeIndex):
        idx = pd.to_datetime(d.index, errors="coerce", utc=True)
        # Si venía tz-naive, localiza a UTC primero
        if getattr(idx, "tz", None) is None:
            idx = idx.tz_localize("UTC")
        idx = idx.tz_convert(TIMEZONE)
        d.index = idx
        d = d.sort_index()
        d.index.name = "ts"
        return d

    # --- D) Rescate: intenta cualquier columna con >80% parseable
    best_col = None
    best_ok = -1
    for c in d.columns:
        dt_try = _parse_datetime_flex(d[c].astype(str))
        ok = int(dt_try.notna().sum())
        if ok > best_ok:
            best_ok = ok
            best_col = c
            best_dt = dt_try
    if best_ok >= int(len(d) * 0.8):
        d["ts"] = best_dt.dt.tz_convert(TIMEZONE) if getattr(best_dt.dt, "tz", None) else best_dt.dt.tz_localize("UTC").dt.tz_convert(TIMEZONE)
        d = d.dropna(subset=["ts"]).sort_values("ts").set_index("ts")
        return d

    # Si llegamos aquí, no pudimos construir TS
    cols_disp = ", ".join(list(d.columns)[:15])
    raise ValueError(
        "No se pudo construir 'ts'. Aporta una columna de fecha/hora. "
        f"Columnas disponibles (primeras 15): {cols_disp}"
    )

# ---------------- Time parts / dummies ----------------
def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega columnas: dow, month, hour, day, sin/cos hora y dow, y flag es_dia_de_pago si no existe.
    """
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        raise ValueError("add_time_parts requiere un índice datetime o una columna 'ts' ya indexada.")

    d["dow"]   = d.index.weekday
    d["month"] = d.index.month
    d["hour"]  = d.index.hour
    d["day"]   = d.index.day
    d["sin_hour"] = np.sin(2*np.pi*d["hour"]/24.0)
    d["cos_hour"] = np.cos(2*np.pi*d["hour"]/24.0)
    d["sin_dow"]  = np.sin(2*np.pi*d["dow"]/7.0)
    d["cos_dow"]  = np.cos(2*np.pi*d["dow"]/7.0)
    if "es_dia_de_pago" not in d.columns:
        d["es_dia_de_pago"] = d["day"].isin([1,2,15,16,29,30,31]).astype(int)
    return d

def dummies_and_reindex(df_last_row: pd.DataFrame, training_cols: list) -> pd.DataFrame:
    """
    Crea dummies solo de ['dow','month','hour'] si alguna de esas aparece en training_cols,
    y devuelve un DataFrame UNA FILA con exactamente training_cols en el mismo orden.
    """
    d = df_last_row.copy()
    base_cols = ["dow", "month", "hour"]
    need_dum = any([(c.startswith("dow_") or c.startswith("month_") or c.startswith("hour_")) for c in training_cols])

    if need_dum:
        # Si faltan columnas base, las construyo a partir del índice
        idx_stub = pd.DataFrame(index=d.index)
        idx_stub = add_time_parts(idx_stub)
        for c in base_cols:
            if c not in d.columns:
                d[c] = idx_stub[c]
        d = pd.get_dummies(d, columns=[c for c in base_cols if c in d.columns], drop_first=False)

    # numérico y saneo
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.ffill().fillna(0)

    # reindex exacto
    out = d.reindex(columns=training_cols, fill_value=0).tail(1)
    return out

# ---------------- (Compat) lags/MA genéricos ----------------
def add_lags_mas(s: pd.Series, lags: list[int] = None, mas: list[int] = None, prefix: str = "") -> pd.DataFrame:
    """
    Pequeña utilidad para generar lags y medias móviles de una serie.
    Devuelve DataFrame con columnas lag_{k} y ma_{w} (con prefijo opcional).
    """
    lags = lags or []
    mas = mas or []
    d = pd.DataFrame(index=s.index)
    for k in lags:
        d[f"{prefix}lag_{k}"] = s.shift(k)
    for w in mas:
        d[f"{prefix}ma_{w}"] = s.rolling(w, min_periods=1).mean()
    return d

