# src/inferencia/utils_io.py
import json, os
import pandas as pd

def load_holidays(path: str) -> set:
    """
    Carga un set de fechas de feriados desde un CSV.
    Espera una columna 'fecha' (insensible a mayúsculas) 
    con formato 'dd-mm-YYYY' o 'YYYY-mm-dd'.
    Intenta detectar separador (',' o ';').
    """
    if not os.path.exists(path):
        print(f"ADVERTENCIA: Archivo de feriados no encontrado en {path}. Usando set vacío.")
        return set()
    
    try:
        # Intentar con ;
        df_h = pd.read_csv(path, delimiter=';', low_memory=False)
        # --- CORRECCIÓN: Normalizar nombres de columnas ---
        df_h.columns = [str(c).strip().lower() for c in df_h.columns]
        
        if 'fecha' not in df_h.columns:
             # Fallback a ,
             df_h = pd.read_csv(path, delimiter=',', low_memory=False)
             # --- CORRECCIÓN: Normalizar nombres de columnas ---
             df_h.columns = [str(c).strip().lower() for c in df_h.columns]
        
        if 'fecha' not in df_h.columns:
             print(f"ADVERTENCIA: No se encontró la columna 'fecha' en {path}. Usando set vacío.")
             return set()
             
        # Intentar parsear con dayfirst=True
        dates = pd.to_datetime(df_h['fecha'], dayfirst=True, errors='coerce').dt.date
        # Si fallan muchos (ej. formato YYYY-MM-DD), intentar sin dayfirst
        if dates.isna().sum() > len(df_h) / 2:
            dates = pd.to_datetime(df_h['fecha'], dayfirst=False, errors='coerce').dt.date
            
        print(f"Cargados {len(dates.dropna())} feriados desde {path}")
        return set(dates.dropna())
        
    except Exception as e:
        print(f"ADVERTENCIA: No se pudieron cargar feriados de {path}: {e}. Usando set vacío.")
        return set()

def write_json(path: str, data):
    """Escribe data (dict, list) a un archivo JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def write_hourly_json(path: str, df_hourly: pd.DataFrame, calls_col: str, tmo_col: str, agentes_col: str):
    """Guarda el pronóstico por hora en JSON."""
    out = (df_hourly.reset_index()
                   .rename(columns={"index":"ts", calls_col:"llamadas_hora", tmo_col:"tmo_hora", agentes_col:"agentes_requeridos"}))
    
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
    out = out.dropna(subset=["ts"])
    out["ts"] = out["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    out["llamadas_hora"] = out["llamadas_hora"].astype(int)
    out["tmo_hora"] = out["tmo_hora"].astype(float).round(2) # Redondear TMO
    out["agentes_requeridos"] = out["agentes_requeridos"].astype(int)

    write_json(path, out.to_dict(orient="records"))

def write_daily_json(path: str, df_hourly: pd.DataFrame, calls_col: str, tmo_col: str):
    """Guarda un resumen por día en JSON."""
    tmp = (df_hourly.reset_index()
                     .rename(columns={"index": "ts"}))

    tmp["ts"] = pd.to_datetime(tmp["ts"], errors="coerce")
    tmp = tmp.dropna(subset=["ts"])
    tmp["fecha"] = tmp["ts"].dt.date.astype(str)

    # Agregar por día: suma de llamadas, promedio ponderado de TMO
    def weighted_avg(g):
        d = g[calls_col]
        w = g[tmo_col]
        # Evitar división por cero si un día no tiene llamadas
        if d.sum() == 0:
            return 0
        return (d * w).sum() / d.sum()

    daily_calls = tmp.groupby("fecha")[calls_col].sum()
    daily_tmo = tmp.groupby("fecha").apply(weighted_avg)
    
    daily = pd.DataFrame({
        "llamadas_dia": daily_calls,
        "tmo_prom_dia": daily_tmo
    }).reset_index()

    daily["llamadas_dia"] = daily["llamadas_dia"].astype(int)
    daily["tmo_prom_dia"] = daily["tmo_prom_dia"].astype(float).round(2)

    write_json(path, daily.to_dict(orient="records"))
