# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

def write_json_pretty(path: str | Path, df_or_records):
    """
    Escribe JSON indentado (UTF-8, ascii off).
    Acepta DataFrame (se serializa como records) o lista de dicts.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(df_or_records, pd.DataFrame):
        records = df_or_records.to_dict(orient="records")
    else:
        records = df_or_records

    with open(p, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def write_daily_agg(path: str | Path, df_daily: pd.DataFrame):
    """
    Guarda daily agg (DataFrame con índice date) como JSON bonito.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = df_daily.reset_index()
    # normalizar fecha a YYYY-MM-DD
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
