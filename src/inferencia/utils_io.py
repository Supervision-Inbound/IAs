# src/inferencia/utils_io.py
import json, os
import pandas as pd

def write_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def write_hourly_json(path: str, df_hourly: pd.DataFrame, calls_col: str, tmo_col: str, agentes_col: str):
    out = (df_hourly.reset_index()
                   .rename(columns={"index":"ts", calls_col:"llamadas_hora", tmo_col:"tmo_hora", agentes_col:"agentes_requeridos"}))
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
    out = out.dropna(subset=["ts"])
    out["ts"] = out["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out["llamadas_hora"] = out["llamadas_hora"].astype(int)
    out["tmo_hora"] = out["tmo_hora"].astype(float)
    out["agentes_requeridos"] = out["agentes_requeridos"].astype(int)
    write_json(path, out.to_dict(orient="records"))

def write_daily_json(path: str, df_hourly: pd.DataFrame, calls_col: str, tmo_col: str):
    """
    Diarios:
      - llamadas_diarias = suma de llamadas
      - tmo_diario = promedio ponderado por llamadas (Σ tmo_hora*llamadas_hora / Σ llamadas_hora)
                     si Σ llamadas_hora == 0 -> promedio simple del día
    """
    tmp = (df_hourly.reset_index().rename(columns={"index":"ts"}))
    tmp["ts"] = pd.to_datetime(tmp["ts"], errors="coerce")
    tmp = tmp.dropna(subset=["ts"])
    tmp["fecha"] = tmp["ts"].dt.date.astype(str)

    grp = tmp.groupby("fecha", as_index=False)
    llamadas = grp.agg(llamadas_diarias=(calls_col, "sum"))

    def _tmo_pond(d: pd.DataFrame) -> float:
        v = d[tmo_col].astype(float)
        w = d[calls_col].astype(float)
        s = w.sum()
        if s > 0:
            return float((v*w).sum() / s)
        return float(v.mean())

    weighted = grp.apply(lambda d: pd.Series({"tmo_diario": _tmo_pond(d)})).reset_index()

    daily = llamadas.merge(weighted, on="fecha", how="left").fillna({"tmo_diario": 0})
    daily["llamadas_diarias"] = daily["llamadas_diarias"].astype(int)
    daily["tmo_diario"] = daily["tmo_diario"].astype(float)
    write_json(path, daily.to_dict(orient="records"))


