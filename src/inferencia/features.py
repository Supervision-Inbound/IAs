# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import numpy as np

TIMEZONE = "America/Santiago"

def ensure_ts(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura columna/índice temporal 'ts':
      - Si vienen 'fecha' y 'hora', arma 'ts'
      - Si viene 'ts' columna, la usa
      - Devuelve con 'ts' como ÚNICO índice (DatetimeIndex tz=America/Santiago), ordenado asc.
    """
    d = df_in.copy()
    cols = {c.lower().strip(): c for c in d.columns}

    if "ts" in cols:
        ts = pd.to_datetime(d[cols["ts"]], errors="coerce", dayfirst=True)
    else:
        # buscar 'fecha' y 'hora'
        fecha = next((cols[c] for c in cols if "fecha" in c), None)
        hora = next((cols[c] for c in cols if "hora" in c), None)
        if not fecha or not hora:
            raise ValueError("No se encontraron columnas de 'fecha' y 'hora', ni 'ts'.")
        ts = pd.to_datetime(
            d[fecha].astype(str).str.strip() + " " + d[hora].astype(str).str.strip(),
            errors="coerce",
            dayfirst=True,
        )

    d["ts"] = ts
    d = d.dropna(subset=["ts"])

    # set as index, tz-localize
    d = d.set_index("ts")
    d.index = pd.DatetimeIndex(d.index)
    try:
        d.index = d.index.tz_localize(TIMEZONE, nonexistent="shift_forward", ambiguous="NaT")
    except TypeError:
        # compat
        d.index = d.index.tz_localize(TIMEZONE)
    d = d.sort_index()

    # IMPORTANTÍSIMO: no dejamos 'ts' como columna
    if "ts" in d.columns:
        d = d.drop(columns=["ts"])

    return d


def add_time_parts(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega partes de tiempo (dow, month, hour, day) esperando 'ts' como índice.
    """
    d = df_in.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        raise ValueError("add_time_parts requiere DatetimeIndex en el índice (ts).")
    d["dow"] = d.index.dayofweek
    d["month"] = d.index.month
    d["hour"] = d.index.hour
    d["day"] = d.index.day
    return d

