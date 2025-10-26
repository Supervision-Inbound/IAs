# src/download_release.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple
import fnmatch
import requests

DEFAULT_OWNER = "Supervision-Inbound"
DEFAULT_REPO = "IAs"
DEFAULT_OUTDIR = "models"

GITHUB_API = "https://api.github.com"


def _token() -> Optional[str]:
    return os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")


def _api_get(url: str, params: Optional[dict] = None, accept: str = "application/vnd.github+json") -> requests.Response:
    headers = {"Accept": accept}
    tok = _token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    r = requests.get(url, headers=headers, params=params, timeout=60)
    if r.status_code == 403 and "rate limit" in r.text.lower():
        reset = r.headers.get("X-RateLimit-Reset")
        if reset:
            wait_s = max(0, int(reset) - int(time.time()) + 2)
            time.sleep(min(wait_s, 60))
            r = requests.get(url, headers=headers, params=params, timeout=60)
    r.raise_for_status()
    return r


def _pick_release(owner: str, repo: str, tag: Optional[str]) -> dict:
    if tag:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/releases/tags/{tag}"
    else:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/releases/latest"
    return _api_get(url).json()


def _match(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(name, p.strip()) for p in patterns)


def _download_asset(asset: dict, out_dir: Path) -> Path:
    url = asset.get("browser_download_url")
    name = asset.get("name") or "asset.bin"
    out = out_dir / name

    headers = {}
    tok = _token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"

    with requests.get(url, headers=headers, stream=True, timeout=300) as r:
        r.raise_for_status()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return out


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descarga assets del último release (o por tag) de GitHub.")
    # Soporta ambas formas:
    #   a) --repo "owner/repo"
    #   b) --owner X --repo Y
    parser.add_argument("--owner", default=None, help="Dueño/organización del repo (opcional si usa --repo 'owner/name').")
    parser.add_argument("--repo", required=True, help="Nombre del repo o 'owner/repo'.")
    parser.add_argument("--tag", default=None, help="Tag específico del release. Si no, usa 'latest'.")
    parser.add_argument("--out_dir", default=DEFAULT_OUTDIR, help=f"Carpeta de salida (default: {DEFAULT_OUTDIR})")
    parser.add_argument(
        "--patterns",
        default="modelo_*.keras,scaler_*.pkl,training_columns_*.json,tmo_*.csv,*_meta.json",
        help="Patrones de archivos a bajar, separados por coma.",
    )
    return parser.parse_args(argv)


def _resolve_owner_repo(owner_arg: Optional[str], repo_arg: str) -> Tuple[str, str]:
    if "/" in repo_arg:
        owner, repo = repo_arg.split("/", 1)
        return owner, repo
    owner = owner_arg or DEFAULT_OWNER
    repo = repo_arg or DEFAULT_REPO
    return owner, repo


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    owner, repo = _resolve_owner_repo(args.owner, args.repo)
    out_dir = Path(args.out_dir)
    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]

    print(f"[download] Repo: {owner}/{repo} | tag: {args.tag or 'latest'}")
    print(f"[download] Out dir: {out_dir}")
    print(f"[download] Patterns: {patterns}")
    if _token():
        print("[download] Usando token GitHub de entorno (GH_TOKEN/GITHUB_TOKEN).")
    else:
        print("[download] Sin token; podrías topar rate limit público.")

    rel = _pick_release(owner, repo, args.tag)
    name = rel.get("name") or rel.get("tag_name") or "<sin nombre>"
    print(f"[download] Release: {name}")

    assets = rel.get("assets") or []
    if not assets:
        print("[download] No hay assets en el release.")
        return 0

    selected = [a for a in assets if _match(a.get("name", ""), patterns)]
    if not selected:
        print("[download] No hay assets que cumplan los patrones.")
        print("          Disponibles:")
        for a in assets:
            print(f"           - {a.get('name')}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for a in selected:
        name = a.get("name")
        print(f"[download] Bajando: {name} ...")
        path = _download_asset(a, out_dir)
        print(f"[download]   -> {path}")

    print("[download] Listo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

