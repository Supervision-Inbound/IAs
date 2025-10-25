# src/data/loader_tmo.py
import pandas as pd

# Este loader asume el esquema del entrenamiento:
# CSV delimitado por ';' con columnas: FECHA, HORA, TMO_COMERCIAL, Q_LLAMADAS_COMERCIAL,
#                                      TMO_TECNICO, Q_LLAMADAS_TECNICO, TMO_GENERAL,
#                                      Q_LLAMADAS_GENERAL
# Devuelve un DataFrame indexado por 'ts' en America/Santiago y nombres normalizados.

TIMEZONE = "America/Santiago"

def load_historico_tmo(path_csv: str) -> pd.DataFrame:
    # intenta ; luego ,
    try:
        df = pd.read_csv(path_csv, delimiter=';', low_memory=False)
        if df.shape[1] == 1:
            df = pd.read_csv(path_csv, delimiter=',', low_memory=False)
    except Exception:
        df = pd.read_csv(path_csv, delimiter=',', low_memory=False)

    # normaliza nombres a minúsculas con _
    cols_norm = {c: c.lower().strip().replace("  "," ").replace(" ","_") for c in df.columns}
    df = df.rename(columns=cols_norm)

    # alias razonables
    # fecha + hora (pueden venir como strings tipo 2024-01-01 y 08:00)
    fecha_col = next((c for c in df.columns if "fecha" in c), None)
    hora_col  = next((c for c in df.columns if "hora"  in c), None)
    if not fecha_col or not hora_col:
        raise ValueError("El archivo TMO requiere columnas de fecha y hora.")

    df["ts"] = pd.to_datetime(df[fecha_col].astype(str) + " " + df[hora_col].astype(str),
                              errors="coerce", dayfirst=True)
    df = df.dropna(subset=["ts"]).sort_values("ts")
    # fija tz
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
    else:
        df["ts"] = df["ts"].dt.tz_convert(TIMEZONE)

    # mapeo de columnas de interés
    rename_map = {
        "tmo_comercial": "tmo_comercial",
        "q_llamadas_comercial": "q_llamadas_comercial",
        "tmo_tecnico": "tmo_tecnico",
        "q_llamadas_tecnico": "q_llamadas_tecnico",
        "tmo_general": "tmo_general",
        "q_llamadas_general": "q_llamadas_general",
    }
    # Asegura que existan (si vienen con acentos o variaciones)
    for k in list(rename_map.keys()):
        if k not in df.columns:
            # buscar alternativa con acentos
            alt = next((c for c in df.columns if c.replace("é","e").replace("á","a") == k), None)
            if alt:
                df = df.rename(columns={alt: k})

    keep_cols = ["ts"] + [c for c in rename_map.keys() if c in df.columns]
    df = df[keep_cols].set_index("ts")

    # calcula proporciones si es posible
    if "q_llamadas_general" in df.columns:
        qg = df["q_llamadas_general"].astype(float).replace(0, pd.NA)
        if "q_llamadas_comercial" in df.columns:
            df["proporcion_comercial"] = (df["q_llamadas_comercial"].astype(float) / qg).fillna(0.0)
        if "q_llamadas_tecnico" in df.columns:
            df["proporcion_tecnica"] = (df["q_llamadas_tecnico"].astype(float) / qg).fillna(0.0)

    return df
