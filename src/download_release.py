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
        "modelo_planner.keras","scaler_planner.pkl","training_columns_planner.json",
        "modelo_tmo.keras","scaler_tmo.pkl","training_columns_tmo.json",
        "modelo_riesgos.keras","scaler_riesgos.pkl","training_columns_riesgos.json",
        "baselines_clima.pkl"
    }

    for a in rel["assets"]:
        name = a.get("name","")
        if name in target_names:
            url = a["browser_download_url"]
            print(f"↓ {name}")
            r = sess.get(url)
            r.raise_for_status()
            with open(os.path.join(out_dir, name), "wb") as f:
                f.write(r.content)
    print("✔ Assets descargados en", out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default="models")
    args = ap.parse_args()
    download_latest_assets(args.owner, args.repo, args.out)

