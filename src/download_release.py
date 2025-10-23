# src/download_release.py
import os, sys, json, argparse, requests
from pathlib import Path

def download_latest_assets(owner: str, repo: str, out_dir: str = "models", token: str | None = os.getenv("GITHUB_TOKEN")):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    if token:
        sess.headers["Authorization"] = f"Bearer {token}"
    sess.headers["Accept"] = "application/vnd.github+json"

    # --- ¡LÍNEA CLAVE! ---
    rel = sess.get(f"https://api.github.com/repos/{owner}/{repo}/releases/latest").json()
    # --- FIN LÍNEA CLAVE ---

    if "assets" not in rel:
        raise RuntimeError(f"No assets in latest release: {rel}")

    target_names = {
        "modelo_planner.keras","scaler_planner.pkl","training_columns_planner.json",
        "modelo_tmo.keras","scaler_tmo.pkl","training_columns_tmo.json",
        "modelo_riesgos.keras","scaler_riesgos.pkl","training_columns_riesgos.json",
        "baselines_clima.pkl"
    }

    print(f"INFO: Descargando assets desde la release '{rel.get('name', 'N/A')}' (ID: {rel.get('id', 'N/A')}, Tag: {rel.get('tag_name', 'N/A')})") # Añadido para depuración

    downloaded_count = 0
    for a in rel["assets"]:
        name = a.get("name","")
        if name in target_names:
            url = a["browser_download_url"]
            print(f"↓ Descargando {name}...")
            try:
                r = sess.get(url, timeout=60) # Añadir timeout
                r.raise_for_status()
                filepath = os.path.join(out_dir, name)
                with open(filepath, "wb") as f:
                    f.write(r.content)
                print(f"  ✔ Guardado en {filepath}")
                downloaded_count += 1
            except requests.exceptions.RequestException as e:
                 print(f"  ERROR: Falló la descarga de {name}: {e}")
            except IOError as e:
                 print(f"  ERROR: Falló al guardar {name}: {e}")

    if downloaded_count == 0:
         print(f"WARN: No se descargó ningún asset esperado de la release 'latest'. Verifica los nombres en target_names y los assets de la release.")
    else:
        print(f"✔ {downloaded_count} Assets descargados en {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default="models")
    args = ap.parse_args()
    try:
        download_latest_assets(args.owner, args.repo, args.out)
    except Exception as e:
        print(f"ERROR: Falló download_latest_assets: {e}")
        sys.exit(1) # Salir con código de error

