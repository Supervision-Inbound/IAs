# src/data/loader_tmo.py
import pandas as pd

TIMEZONE = "America/Santiago"


def _read_csv_smart(path_csv: str) -> pd.DataFrame:
    """Lee CSV probando ';' y luego ',' si es necesario."""
    try:
        df = pd.read_csv(path_csv, delimiter=';', low_memory=False)
        # Si quedó todo en una sola columna, probar con coma
        if df.shape[1] == 1:
            df = pd.read_csv(path_csv, delimiter=',', low_memory=False)
        return df
    except Exception:
        # Fallback simple
        return pd.read_csv(path_csv, delimiter=',', low_memory=False)


def load_historico_tmo(path_csv: str) -> pd.DataFrame:
    """
    Carga el histórico puro de TMO (el mismo con el que se entrenó el modelo autoregresivo).
    - Acepta separador ';' o ','.
    - Asegura columna 'ts' (a partir de 'fecha' + 'hora').
    - Normaliza nombres y deriva 'tmo_general' si no está (ponderación por llamadas).
    - Calcula proporciones 'proporcion_comercial'/'proporcion_tecnica' si existe q_llamadas_general.
    """
    df = _read_csv_smart(path_csv)

    # normalización básica de nombres
    cols_norm = {
        c: c.lower().strip()
            .replace("  ", " ")
            .replace(" ", "_")
            .replace("é", "e")
            .replace("á", "a")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
            .replace("ñ", "n")
        for c in df.columns
    }
    df = df.rename(columns=cols_norm)

    # detectar fecha y hora
    fecha_col = next((c for c in df.columns if "fecha" in c), None)
    hora_col  = next((c for c in df.columns if "hora"  in c), None)
    if not fecha_col or not hora_col:
        raise ValueError("El archivo TMO requiere columnas separadas de 'fecha' y 'hora'.")

    # construir ts
    df["ts"] = pd.to_datetime(
        df[fecha_col].astype(str) + " " + df[hora_col].astype(str),
        errors="coerce",
        dayfirst=True
    )
    df = df.dropna(subset=["ts"]).sort_values("ts")

    # localización de zona horaria
    if getattr(df["ts"].dt, "tz", None) is None:
        df["ts"] = df["ts"].dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
    else:
        df["ts"] = df["ts"].dt.tz_convert(TIMEZONE)

    df = df.set_index("ts")

    # asegurar nombres claves (si vinieron con variaciones)
    rename_candidates = {
        "tmo_comercial": "tmo_comercial",
        "q_llamadas_comercial": "q_llamadas_comercial",
        "tmo_tecnico": "tmo_tecnico",
        "q_llamadas_tecnico": "q_llamadas_tecnico",
        "tmo_general": "tmo_general",
        "q_llamadas_general": "q_llamadas_general",
    }
    for tgt in list(rename_candidates.keys()):
        if tgt not in df.columns:
            # intentar encontrar columnas equivalentes ya normalizadas
            alt = next((c for c in df.columns if c == tgt), None)
            if alt:
                df = df.rename(columns={alt: tgt})

    # convertir numéricos relevantes
    for c in ["tmo_comercial", "tmo_tecnico", "q_llamadas_comercial", "q_llamadas_tecnico", "q_llamadas_general", "tmo_general"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # derivar tmo_general si no existe: ponderación por llamadas contestadas
    if "tmo_general" not in df.columns:
        qc = df.get("q_llamadas_comercial")
        qt = df.get("q_llamadas_tecnico")
        tc = df.get("tmo_comercial")
        tt = df.get("tmo_tecnico")
        if qc is not None and qt is not None and tc is not None and tt is not None:
            den = (qc.fillna(0) + qt.fillna(0)).replace(0, pd.NA)
            num = (tc.fillna(0) * qc.fillna(0)) + (tt.fillna(0) * qt.fillna(0))
            df["tmo_general"] = (num / den).astype("float64")

    # proporciones opcionales (features útiles)
    if "q_llamadas_general" in df.columns:
        qg = df["q_llamadas_general"].replace(0, pd.NA)
        if "q_llamadas_comercial" in df.columns:
            df["proporcion_comercial"] = (df["q_llamadas_comercial"] / qg).fillna(0.0)
        if "q_llamadas_tecnico" in df.columns:
            df["proporcion_tecnica"] = (df["q_llamadas_tecnico"] / qg).fillna(0.0)

    return df

