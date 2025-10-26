# src/download_release.py
import os, sys, json, argparse, requests
from pathlib import Path

def download_latest_assets(owner: str, repo: str, out_dir: str = "models", token: str | None = os.getenv("GITHUB_TOKEN")):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    if token:
        sess.headers["Authorization"] = f"Bearer {token}"
    sess.headers["Accept"] = "application/vnd.github+json"

    rel = sess.get(f"https://api.github.com/repos/{owner}/{repo}/releases/latest").json()
    if "assets" not in rel:
        raise RuntimeError(f"No assets in latest release: {rel}")

    target_names = {
        "modelo_planner.keras",
        "scaler_planner.pkl",
        "training_columns_planner.json",

        # --- TMO NUEVO (Basado en features de llamadas) ---
        "modelo_tmo.keras",             # Asumimos que se sobreescribe
        "scaler_tmo.pkl",               # Asumimos que se sobreescribe
        "training_columns_tmo.json",    # Asumimos que se sobreescribe
        "tmo_baseline_dow_hour.csv",    # <--- MODIFICADO: Nuevo artefacto
        "tmo_residual_meta.json",       # <--- MODIFICADO: Nuevo artefacto

        "modelo_riesgos.keras",
        "scaler_riesgos.pkl",
        "training_columns_riesgos.json",
        "baselines_clima.pkl"
    }

    print("Buscando assets para descargar...")
    assets_found = 0
    for a in rel["assets"]:
        name = a.get("name","")
        if name in target_names:
            url = a["browser_download_url"]
            print(f"↓ {name}")
            r = sess.get(url)
            r.raise_for_status()
            with open(os.path.join(out_dir, name), "wb") as f:
                f.write(r.content)
            assets_found += 1
    
    if assets_found > 0:
        print(f"✔ {assets_found} assets descargados en", out_dir)
    else:
        print("X No se encontraron assets nuevos en el release.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # <--- MODIFICADO: Cambiado para soportar el formato 'owner/repo'
    parser.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY", "org/repo"))
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()
    
    if '/' not in args.repo:
        print("ERROR: El argumento --repo debe ser 'owner/repo'.", file=sys.stderr)
        sys.exit(1)
        
    owner, repo_name = args.repo.split("/", 1)
    download_latest_assets(owner, repo_name, args.out_dir)
