#!/usr/bin/env python3
"""Step 01 - fetch the real EyePACS diabetic-retinopathy data from Kaggle.

Two real EyePACS-2015 mirrors are supported (both are the Kaggle *Diabetic
Retinopathy Detection* competition set, i.e. EyePACS, 35,126 training fundus
photographs graded 0-4 by licensed clinicians):

  eyepacs-224   sovitrath/diabetic-retinopathy-2015-data-colored-resized  (2.0 GB)
                images foldered by grade name - fast, good default
  eyepacs-hi    tanlikesmath/diabetic-retinopathy-resized                 (7.8 GB)
                ~1024 px images + trainLabels.csv - better for microaneurysms

The official Kaggle CLI stalls behind some proxies, so we stream the signed
redirect ourselves with resume support.

    python scripts/01_download_data.py --dataset eyepacs-224
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dr.config import RAW_DIR  # noqa: E402

DATASETS = {
    "eyepacs-224": {
        "slug": "sovitrath/diabetic-retinopathy-2015-data-colored-resized",
        "zip": "eyepacs_2015_colored_resized.zip",
        "approx_gb": 2.08,
    },
    "eyepacs-hi": {
        "slug": "tanlikesmath/diabetic-retinopathy-resized",
        "zip": "eyepacs_2015_resized_hi.zip",
        "approx_gb": 7.79,
    },
    # External validation cohort (no Kaggle credentials needed). 3,000 colour
    # fundus photos, ALFEDIAM grades 0-4, 1960-3600 px, CC BY 4.0.
    # https://zenodo.org/records/23101200  (checksums from the Zenodo record)
    "fdr26": {
        "zenodo": "https://zenodo.org/api/records/23101200/files/{name}/content",
        "files": {
            "Fundus_FDR26.zip": "be97fb330cb06cadf3926de2ff8be7c9",
            "metadata_classification_DR.xlsx": "128b725a17177d38e419993c423ad4b9",
        },
        "zip": "Fundus_FDR26.zip",
        "approx_gb": 1.68,
    },
}


def _md5(path: Path) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_zenodo(spec: dict, dest_dir: Path) -> None:
    """Download Zenodo files with resume and verify each MD5 before use."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name, md5 in spec["files"].items():
        dest = dest_dir / name
        if dest.exists() and _md5(dest) == md5:
            # already complete: `curl --continue-at -` on a finished file answers 416 and --fail turns that
            # into an error, so an interrupted-then-restarted run could never get past this point
            print(f"[zenodo] {name} already present and verified ({md5})")
            continue
        url = spec["zenodo"].format(name=name)
        rc = subprocess.call(["curl", "-L", "--fail", "--retry", "8", "--retry-delay", "5",
                              "--retry-all-errors", "--continue-at", "-", "-o", str(dest), url])
        if rc != 0:
            sys.exit(f"curl failed ({rc}) for {name}")
        got = _md5(dest)
        if got != md5:
            sys.exit(f"checksum mismatch for {name}: got {got}, expected {md5}")
        print(f"[zenodo] {name} verified ({md5})")


def kaggle_credentials() -> tuple[str, str]:
    env_u, env_k = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if env_u and env_k:
        return env_u, env_k
    path = Path.home() / ".kaggle" / "kaggle.json"
    if not path.exists():
        sys.exit("No Kaggle credentials: set KAGGLE_USERNAME/KAGGLE_KEY or create "
                 "~/.kaggle/kaggle.json (Kaggle > Account > Create New API Token).")
    d = json.loads(path.read_text())
    return d["username"], d["key"]


def stream_download(slug: str, dest: Path) -> None:
    """curl with --continue-at so an interrupted pull resumes instead of restarting."""
    user, key = kaggle_credentials()
    url = f"https://www.kaggle.com/api/v1/datasets/download/{slug}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-L", "--fail", "--retry", "8", "--retry-delay", "5",
           "--retry-all-errors", "--continue-at", "-",
           "-u", f"{user}:{key}", "-o", str(dest), url]
    print(f"[download] {slug}\n[download] -> {dest}")
    t0 = time.time()
    rc = subprocess.call(cmd)
    if rc != 0 and not dest.exists():
        sys.exit(f"curl failed with code {rc}")
    mb = dest.stat().st_size / 1e6
    print(f"[download] {mb:.0f} MB in {time.time() - t0:.0f}s")


def safe_extract(zip_path: Path, out_dir: Path) -> None:
    """Extract with path-traversal protection (never trust archive members)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        root = out_dir.resolve()
        for m in members:
            target = (root / m).resolve()
            if not str(target).startswith(str(root)):
                sys.exit(f"unsafe path in archive: {m}")
        print(f"[extract] {len(members)} entries -> {out_dir}")
        for i, m in enumerate(members):
            zf.extract(m, out_dir)
            if i % 2000 == 0:
                print(f"[extract] {i}/{len(members)}", flush=True)
    print("[extract] done")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="eyepacs-224")
    ap.add_argument("--keep-zip", action="store_true")
    ap.add_argument("--out", type=Path, default=RAW_DIR)
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    zip_path = args.out / spec["zip"]
    extract_dir = args.out / args.dataset

    if args.dataset == "fdr26":
        if extract_dir.exists() and any(extract_dir.rglob("*.jpg")):
            print(f"[skip] already extracted at {extract_dir}")
            return
        fetch_zenodo(spec, extract_dir)           # zip + xlsx land in extract_dir
        safe_extract(extract_dir / spec["zip"], extract_dir)
        if not args.keep_zip:
            (extract_dir / spec["zip"]).unlink(missing_ok=True)
        print(f"[next] python scripts/02_preprocess.py --root {extract_dir} --fdr26 "
              f"--out data/cache_fdr26 --quality-reference data/cache/quality_reference.json")
        return

    probe = args.out.resolve()
    while not probe.exists():                 # nearest existing ancestor = the volume that will be used
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / 1e9
    need = spec["approx_gb"] * 2.2
    if free_gb < need:
        sys.exit(f"Need ~{need:.1f} GB free, only {free_gb:.1f} GB available.")

    if extract_dir.exists() and any(extract_dir.rglob("*.png")) or \
       extract_dir.exists() and any(extract_dir.rglob("*.jpeg")):
        print(f"[skip] already extracted at {extract_dir}")
        return

    if not zip_path.exists() or zip_path.stat().st_size < spec["approx_gb"] * 9e8:
        stream_download(spec["slug"], zip_path)
    else:
        print(f"[skip] zip already present: {zip_path}")

    safe_extract(zip_path, extract_dir)
    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)
        print("[cleanup] removed zip")


if __name__ == "__main__":
    main()
