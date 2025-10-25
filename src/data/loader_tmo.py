# src/data/loader_tmo.py
import pandas as pd

TIMEZONE = "America/Santiago"


def _read_csv_smart(path_csv: str) -> pd.DataFrame:
    """Lee CSV probando ';' y luego ',' si es necesario."""
    try:
        df = pd.read_csv(path_csv, delimiter=';', low_memory=False)
        if df.shape[1] == 1:
            df = pd.read_csv(path_csv, delimiter=',', low_memory=False)
        return df
    except Exception:
        return pd.read_csv(path_csv, delimiter=',', low_memory=False)


def load_historico_tmo(path_csv: str) -> pd.DataFrame:
    df = _read_csv_smart(path_csv)

    cols_norm = {
        c: c.lower().strip()
            .replace("  ", " ")
            .replace(" ", "_")
            .replace("é", "e").replace("á", "a").replace("í", "i").replace("ó", "o").replace("ú", "u")
            .replace("ñ", "n")
        for c in df.columns
    }
    df = df.rename(columns=cols_norm)

    fecha_col = next((c for c in df.columns if "fecha" in c), None)
    hora_col  = next((c for c in df.columns if "hora"  in c), None)
    if not fecha_col or not hora_col:
        raise ValueError("El archivo TMO requiere columnas separadas de 'fecha' y 'hora'.")

    df["ts"] = pd.to_datetime(
        df[fecha_col].astype(str) + " " + df[hora_col].astype(str),
        errors="coerce",
        dayfirst=True
    )
    df = df.dropna(subset=["ts"]).sort_values("ts")

    if getattr(df["ts"].dt, "tz", None) is None:
        df["ts"] = df["ts"].dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
    else:
        df["ts"] = df["ts"].dt.tz_convert(TIMEZONE)

    df = df.set_index("ts")

    for c in ["tmo_comercial", "tmo_tecnico", "q_llamadas_comercial", "q_llamadas_tecnico", "q_llamadas_general", "tmo_general"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "tmo_general" not in df.columns:
        qc = df.get("q_llamadas_comercial")
        qt = df.get("q_llamadas_tecnico")
        tc = df.get("tmo_comercial")
        tt = df.get("tmo_tecnico")
        if qc is not None and qt is not None and tc is not None and tt is not None:
            den = (qc.fillna(0) + qt.fillna(0)).replace(0, pd.NA)
            num = (tc.fillna(0) * qc.fillna(0)) + (tt.fillna(0) * qt.fillna(0))
            df["tmo_general"] = (num / den).astype("float64")

    if "q_llamadas_general" in df.columns:
        qg = df["q_llamadas_general"].replace(0, pd.NA)
        if "q_llamadas_comercial" in df.columns:
            df["proporcion_comercial"] = (df["q_llamadas_comercial"] / qg).fillna(0.0)
        if "q_llamadas_tecnico" in df.columns:
            df["proporcion_tecnica"] = (df["q_llamadas_tecnico"] / qg).fillna(0.0)

    return df
