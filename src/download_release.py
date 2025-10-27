# src/download_release.py
import os, sys, json, argparse, requests
from pathlib import Path

def download_latest_assets(owner: str, repo: str, out_dir: str = "models", token: str | None = os.getenv("GITHUB_TOKEN")):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    if token:
        sess.headers["Authorization"] = f"Bearer {token}"
    sess.headers["Accept"] = "application/vnd.github+json"

    rel_url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
    print(f"Fetching release info from: {rel_url}")
    rel = sess.get(rel_url).json()
    
    if "assets" not in rel or not rel["assets"]:
        print(f"Respuesta API: {rel}")
        raise RuntimeError(f"No assets in latest release or API error.")

    target_names = {
        # Planner (Llamadas) - Sin cambios
        "modelo_planner.keras", "scaler_planner.pkl", "training_columns_planner.json",
        
        # TMO (Nueva lógica v7-residual)
        "modelo_tmo.keras", "scaler_tmo.pkl", "training_columns_tmo.json",
        "tmo_baseline_dow_hour.csv",  # <-- NUEVO ARTEFACTO
        "tmo_residual_meta.json",     # <-- NUEVO ARTEFACTO

        # Riesgos y Clima (Sin cambios)
        "modelo_riesgos.keras", "scaler_riesgos.pkl", "training_columns_riesgos.json",
        "baselines_clima.pkl"
    }

    downloaded_count = 0
    for a in rel["assets"]:
        name = a.get("name", "")
        if name in target_names:
            url = a["browser_download_url"]
            print(f"↓ Descargando {name}")
            try:
                r = sess.get(url)
                r.raise_for_status()
                with open(os.path.join(out_dir, name), "wb") as f:
                    f.write(r.content)
                downloaded_count += 1
            except requests.exceptions.RequestException as e:
                print(f"ERROR descargando {name}: {e}")
    
    print(f"✔ {downloaded_count} assets descargados en {out_dir}")
    if downloaded_count < len(target_names):
        print(f"ADVERTENCIA: Faltaron {len(target_names) - downloaded_count} artefactos. Revisa el release.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download latest model assets from GitHub release.")
    parser.add_argument("repo_owner", type=str, help="Owner del repo (e.g., 'usuario')")
    parser.add_argument("repo_name", type=str, help="Nombre del repo (e.g., 'mi-repo')")
    parser.add_argument("--out_dir", type=str, default="models", help="Directorio de salida (default: 'models')")
    args = parser.parse_args()
    
    # GITHUB_TOKEN se lee automáticamente desde variables de entorno
    download_latest_assets(args.repo_owner, args.repo_name, args.out_dir)

