# src/download_release.py
import os, sys, json, argparse, hashlib, requests
from pathlib import Path

TARGET_NAMES = {
    # Planner (llamadas)
    "modelo_planner.keras","scaler_planner.pkl","training_columns_planner.json",
    # TMO (no se toca, pero lo dejamos por completitud)
    "modelo_tmo.keras","scaler_tmo.pkl","training_columns_tmo.json",
    "tmo_baseline_dow_hour.csv","tmo_residual_meta.json",
    # Otros opcionales
    "modelo_riesgos.keras","scaler_riesgos.pkl","training_columns_riesgos.json",
    "baselines_clima.pkl",
    # <--- ¡AÑADIDO! Este es el nuevo artefacto que faltaba
    "pre_holiday_factors.json" 
}

def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()[:16]

def _gh_get(sess: requests.Session, url: str):
    r = sess.get(url); r.raise_for_status(); return r.json()

def download_assets(owner: str, repo: str, out_dir: str = "models",
                    token: str | None = os.getenv("GITHUB_TOKEN"),
                    tag: str | None = None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    sess = requests.Session()
    if token:
        sess.headers["Authorization"] = f"Bearer {token}"
    sess.headers["Accept"] = "application/vnd.github+json"

    if tag:
        rel = _gh_get(sess, f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}")
        print(f"▶ Release por TAG: {tag}")
    else:
        rel = _gh_get(sess, f"https://api.github.com/repos/{owner}/{repo}/releases/latest")
        print("▶ Release: latest")

    assets = rel.get("assets", [])
    if not assets:
        raise RuntimeError(f"No assets en release: {json.dumps(rel)[:300]}...")

    loaded = []
    for a in assets:
        name = a.get("name", "")
        if name in TARGET_NAMES:
            url = a.get("browser_download_url")
            print(f"↓ {name}")
            rb = sess.get(url).content
            sha = _sha256_bytes(rb)
            with open(out / name, "wb") as f:
                f.write(rb)
            print(f"  ↳ guardado en {out/name} | sha256[:16]={sha}")
            loaded.append(name)

    # seguridad mínima: avisar ausentes
    missing = sorted(list(TARGET_NAMES - set(loaded)))
    if missing:
        print("⚠ Faltaron assets:", ", ".join(missing))
    else:
        print("✔ Todos los assets objetivo descargados.")

    # log útil para depurar qué se usará en inferencia
    print("\n[USO EN INFERENCIA] modelos en", str(out.resolve()))
    for p in sorted(out.glob("*")):
        print(" -", p.name)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default="models")
    ap.add_argument("--tag", default=None, help="Tag específico del release (si se omite usa 'latest').")
    args = ap.parse_args()
    download_assets(args.owner, args.repo, args.out, tag=args.tag)

