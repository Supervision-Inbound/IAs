# src/download_release.py
import os
import sys
import json
import argparse
import requests
from pathlib import Path

def download_latest_assets(owner: str, repo: str, out_dir: str = "models", token: str | None = os.getenv("GITHUB_TOKEN")):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    if token:
        sess.headers["Authorization"] = f"Bearer {token}"
    sess.headers["Accept"] = "application/vnd.github+json"

    rel_url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
    print(f"Buscando release 'latest' en: {rel_url}")
    
    try:
        rel = sess.get(rel_url, timeout=10).json()
    except requests.exceptions.RequestException as e:
        print(f"Error de red al consultar la API de GitHub: {e}")
        sys.exit(1)

    if "assets" not in rel or not rel["assets"]:
        print(f"Respuesta de la API (no se encontraron 'assets'): {rel.get('message', 'Sin mensaje')}")
        raise RuntimeError(f"No se encontraron 'assets' en el último release o hubo un error de API.")

    # --- LISTA DE ARTEFACTOS ACTUALIZADA ---
    target_names = {
        # Planner (Llamadas) - Sin cambios
        "modelo_planner.keras", 
        "scaler_planner.pkl", 
        "training_columns_planner.json",
        
        # TMO (Nueva lógica v7-residual)
        "modelo_tmo.keras", 
        "scaler_tmo.pkl", 
        "training_columns_tmo.json",
        "tmo_baseline_dow_hour.csv",  # <-- ARTEFACTO NUEVO
        "tmo_residual_meta.json",     # <-- ARTEFACTO NUEVO

        # Riesgos y Clima (Sin cambios)
        "modelo_riesgos.keras", 
        "scaler_riesgos.pkl", 
        "training_columns_riesgos.json",
        "baselines_clima.pkl"
    }
    # ----------------------------------------

    downloaded_count = 0
    assets_found_in_release = {a.get("name", "") for a in rel["assets"]}
    
    print(f"Encontrados {len(assets_found_in_release)} assets en el release.")

    for name in target_names:
        if name not in assets_found_in_release:
            print(f"ADVERTENCIA: El asset '{name}' no se encontró en el release de GitHub.")
            continue

        asset_url = next((a["browser_download_url"] for a in rel["assets"] if a.get("name") == name), None)
        
        if not asset_url:
            print(f"ERROR: No se pudo obtener la URL para '{name}'.")
            continue
            
        print(f"↓ Descargando {name}...")
        try:
            r = sess.get(asset_url, timeout=30)
            r.raise_for_status()
            with open(os.path.join(out_dir, name), "wb") as f:
                f.write(r.content)
            downloaded_count += 1
        except requests.exceptions.RequestException as e:
            print(f"ERROR descargando {name}: {e}")
    
    print("-" * 30)
    print(f"✔ {downloaded_count} de {len(target_names)} assets requeridos fueron descargados en '{out_dir}'.")
    
    if downloaded_count < len(target_names):
        print("ADVERTENCIA: Faltaron assets. Revisa la lista de advertencias de arriba.")
    else:
        print("¡Descarga de artefactos completada!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Descargar últimos artefactos de modelo desde un release de GitHub.",
        epilog="Ejemplo: python -m src.download_release --owner MiUsuario --repo MiRepo"
    )
    
    # Acepta --owner y --repo como argumentos nombrados
    parser.add_argument(
        "--owner", 
        dest="repo_owner",
        type=str, 
        required=True,
        help="Owner (usuario u organización) del repositorio (e.g., 'Supervision-Inbound')"
    )
    parser.add_argument(
        "--repo", 
        dest="repo_name",
        type=str, 
        required=True,
        help="Nombre del repositorio (e.g., 'IAs')"
    )
    parser.add_argument(
        "--out_dir", 
        type=str, 
        default="models", 
        help="Directorio de salida (default: 'models')"
    )
    args = parser.parse_args()
    
    try:
        download_latest_assets(args.repo_owner, args.repo_name, args.out_dir)
    except Exception as e:
        print(f"\nError fatal durante la descarga: {e}")
        sys.exit(1)
