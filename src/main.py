# src/main.py (¡Actualizado para v10.3 - Fix Normalización!)
import argparse
import os
import numpy as np
import pandas as pd
import re
import unicodedata

from src.inferencia.inferencia_core import forecast_120d
# Ajustamos la importación de ensure_ts, ya que la definiremos localmente
# para asegurar que sea la misma que en el entrenamiento.
# from src.inferencia.features import ensure_ts 

DATA_FILE = "data/historical_data.csv"
HOLIDAYS_FILE = "data/Feriados_Chilev2.csv"

# ======= Claves de negocio (AJUSTADAS A v10.2) =======
TARGET_CALLS_NEW = "recibidos"
TARGET_TMO_NEW   = "tmo (segundos)"
TZ = "America/Santiago"

def smart_read_historical(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    return pd.read_csv(path, delimiter=';', low_memory=False)

def parse_tmo_to_seconds(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().replace(",", ".")
    if s.replace(".", "", 1).isdigit():
        try: return float(s)
        except: return np.nan
    parts = s.split(":")
    try:
        if len(parts) == 3: return float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
        if len(parts) == 2: return float(parts[0])*60 + float(parts[1])
        return float(s)
    except:
        return np.nan

def load_holidays(csv_path: str) -> set:
    if not os.path.exists(csv_path): return set()
    fer = pd.read_csv(csv_path)
    cols_map = {c.lower().strip(): c for c in fer.columns}
    fecha_col = None
    for cand in ["fecha", "date", "dia", "día"]:
        if cand in cols_map:
            fecha_col = cols_map[cand]; break
    if not fecha_col: return set()
    fechas = pd.to_datetime(fer[fecha_col].astype(str), dayfirst=True, errors="coerce").dropna().dt.date
    return set(fechas)

def mark_holidays_index(dt_index, holidays_set: set) -> pd.Series:
    tz = getattr(dt_index, "tz", None)
    idx_dates = dt_index.tz_convert(TZ).date if tz is not None else dt_index.date
    return pd.Series([d in holidays_set for d in idx_dates], index=dt_index, dtype=int, name="feriados")

def add_es_dia_de_pago(df_idx: pd.DataFrame) -> pd.Series:
    return pd.Series(0, index=df_idx.index, name="es_dia_de_pago")

# --- AJUSTE v10.3: Funciones de normalización traídas aquí ---
# Para asegurar que el 'main' normalice igual que el 'entrenamiento'

def _norm(s: str) -> str:
    """Normaliza un string a minúsculas, sin acentos y con guiones bajos."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s) # Usamos guion bajo para consistencia
    s = re.sub(r"[^a-z0-9_()]+", "", s) # Limpia caracteres especiales
    return s

def ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura un índice 'ts' y normaliza columnas a minúsculas/guion bajo.
    Esta es la función CRÍTICA que alinea los datos.
    """
    d = df.copy()
    
    # 1. Normalizar todas las columnas primero
    # (Ej: "TMO (segundos)" -> "tmo_(segundos)")
    d.columns = [_norm(c) for c in d.columns]

    # 2. Encontrar fecha y hora (ya normalizadas)
    fecha_col = next((c for c in d.columns if "fecha" in c), None)
    hora_col  = next((c for c in d.columns if "hora"  in c), None)
    
    if not fecha_col or not hora_col:
        raise ValueError("No se pudieron encontrar columnas 'fecha' y 'hora' en historical_data.csv")

    # 3. Crear 'ts'
    d["ts"] = pd.to_datetime(d[fecha_col].astype(str) + " " + d[hora_col].astype(str),
                             errors="coerce", dayfirst=True)
    
    d = d.dropna(subset=["ts"]).sort_values("ts")
    
    # 4. Establecer 'ts' como índice
    d = d.set_index("ts").tz_localize("UTC").tz_convert(TZ)
    
    return d

def main(horizonte_dias: int):
    os.makedirs("public", exist_ok=True)

    # 1) Leer histórico único (volumen + TMO)
    dfh = smart_read_historical(DATA_FILE)
    
    # --- AJUSTE v10.3: Normalizar columnas ANTES de buscar ---
    # Esto es lo que faltaba. Ahora "Recibidos" se volverá "recibidos"
    dfh.columns = [c.strip().lower() for c in dfh.columns]
    # --- FIN AJUSTE v10.3 ---

    # 2) Normalizar columna de volumen al nombre EXACTO (v10.2)
    if TARGET_CALLS_NEW not in dfh.columns:
        # La lista de candidatos ahora sí coincidirá
        for cand in ["recibidos_nacional", "recibidas", "recibidos", "contestados", "total_llamadas", "llamadas"]:
            if cand in dfh.columns:
                dfh = dfh.rename(columns={cand: TARGET_CALLS_NEW})
                break
        # Si AÚN no se encuentra, lanzar un error claro
        if TARGET_CALLS_NEW not in dfh.columns:
            raise ValueError(f"No se encontró la columna '{TARGET_CALLS_NEW}' (ni candidatos) en {DATA_FILE}")

    # 3) Normalizar TMO a segundos -> 'tmo (segundos)'
    if TARGET_TMO_NEW not in dfh.columns:
        tmo_source = None
        for cand in ["tmo_general", "tmo (s)", "tmo (segundos)", "tmo_seg", "tmo", "aht"]:
            if cand in dfh.columns:
                tmo_source = cand; break
        if tmo_source:
            if tmo_source != TARGET_TMO_NEW:
                 dfh[TARGET_TMO_NEW] = dfh[tmo_source].apply(parse_tmo_to_seconds)
            else:
                 dfh[TARGET_TMO_NEW] = dfh[tmo_source].apply(parse_tmo_to_seconds)
        # Si AÚN no se encuentra
        if TARGET_TMO_NEW not in dfh.columns:
             raise ValueError(f"No se encontró la columna '{TARGET_TMO_NEW}' (ni candidatos) en {DATA_FILE}")

    # 4) Asegurar índice temporal (usamos la nueva función 'ensure_ts' local)
    # Esta función es diferente de la v8, normaliza todas las columnas
    # La redefinimos localmente para no depender de .features
    
    # 4.A) Encontrar 'fecha' y 'hora' antes de que ensure_ts las normalice
    fecha_col_name = next((c for c in dfh.columns if "fecha" in c), None)
    hora_col_name = next((c for c in dfh.columns if "hora" in c), None)
    if not fecha_col_name or not hora_col_name:
        raise ValueError(f"Faltan 'fecha' u 'hora' en {DATA_FILE}")

    d = dfh.copy()
    d["ts"] = pd.to_datetime(d[fecha_col_name].astype(str) + " " + d[hora_col_name].astype(str),
                             errors="coerce", dayfirst=True)
    d = d.dropna(subset=["ts"]).sort_values("ts")
    d = d.set_index("ts").tz_localize("UTC").tz_convert(TZ)
    dfh = d # Reemplazamos dfh con la versión indexada por tiempo


    # 5) Derivar calendario (feriados) y 'es_dia_de_pago' (forzado a 0)
    holidays_set = load_holidays(HOLIDAYS_FILE)
    if "feriados" not in dfh.columns:
        dfh["feriados"] = mark_holidays_index(dfh.index, holidays_set).values
    dfh["feriados"] = pd.to_numeric(dfh["feriados"], errors="coerce").fillna(0).astype(int)
    if "es_dia_de_pago" not in dfh.columns:
        dfh["es_dia_de_pago"] = add_es_dia_de_pago(dfh).values
    else:
        dfh["es_dia_de_pago"] = 0

    # 6) ffill columnas clave (¡Limpio!)
    for c in [TARGET_CALLS_NEW, TARGET_TMO_NEW, "feriados", "es_dia_de_pago"]:
        if c in dfh.columns:
            dfh[c] = dfh[c].ffill()
        else:
            # Si una columna clave (ej. feriados) no existe, crearla
            if c not in [TARGET_CALLS_NEW, TARGET_TMO_NEW]:
                dfh[c] = 0

    # 7) Forecast
    df_hourly = forecast_120d(
        dfh,
        horizon_days=horizonte_dias,
        holidays_set=holidays_set
    )

    # 8) Alertas clima
    from src.inferencia.alertas_clima import generar_alertas
    generar_alertas(df_hourly[["calls"]])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizonte", type=int, default=120)
    args = ap.parse_args()
    main(args.horizonte)
