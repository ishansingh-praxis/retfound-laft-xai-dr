#!/usr/bin/env python3
"""Step 11 - fetch the real pixel-level lesion annotations.

    IDRiD  aaryapatel98/indian-diabetic-retinopathy-image-dataset   0.43 GB
           81 images, MA / HE / EX / SE masks at 4288x2848
    DDR    sunfish141/ddr-segmentation                              0.64 GB
           757 images, same four classes, mixed resolution
    FGADR  not on Kaggle - needs a signed request form from the authors.
           Drop it at data/raw/lesion/fgadr/ and the loader picks it up.

Both are Kaggle mirrors of the published academic releases, so this needs the
same ~/.kaggle/kaggle.json credentials as step 01.

The Kaggle signed redirect goes dead on a slow link, so a *stalled* transfer has
to be aborted and retried rather than waited on - hence --speed-limit /
--speed-time plus --continue-at, which resumes instead of restarting.

    python scripts/11_download_lesions.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dr.config import RAW_DIR  # noqa: E402
from dr.kaggle_auth import curl_config, describe, require  # noqa: E402

DATASETS = {
    "idrid": {"slug": "aaryapatel98/indian-diabetic-retinopathy-image-dataset",
              "approx_gb": 0.43},
    "ddr": {"slug": "sunfish141/ddr-segmentation", "approx_gb": 0.64},
}


def safe_extract(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = out_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for m in zf.namelist():
            if not str((root / m).resolve()).startswith(str(root)):
                sys.exit(f"unsafe path in archive: {m}")
        zf.extractall(out_dir)


def fetch(name: str, spec: dict, out_root: Path, attempts: int = 8) -> bool:
    dest = out_root / name
    if dest.exists() and any(dest.rglob("*.tif")):
        print(f"[skip] {name} already extracted at {dest}")
        return True
    creds = require()
    print(f"[{name}] Kaggle credentials: {describe(creds)}")
    zip_path = out_root / f"{name}.zip"
    url = f"https://www.kaggle.com/api/v1/datasets/download/{spec['slug']}"
    target = spec["approx_gb"] * 9e8

    for a in range(1, attempts + 1):
        have = zip_path.stat().st_size if zip_path.exists() else 0
        if have >= target:
            break
        print(f"[{name}] attempt {a}, have {have/1e6:.0f} MB")
        subprocess.run([
            "curl", "-L", "--fail", "--continue-at", "-",
            "--speed-limit", "20000", "--speed-time", "30",
            "--connect-timeout", "20", "--max-time", "1800",
            "-K", "-", "-o", str(zip_path), url],          # credentials on stdin, never in argv
            input=curl_config(creds), text=True)
    if not zip_path.exists() or zip_path.stat().st_size < target:
        print(f"[{name}] download incomplete, skipping")
        return False

    print(f"[{name}] extracting -> {dest}")
    safe_extract(zip_path, dest)
    zip_path.unlink(missing_ok=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RAW_DIR / "lesion")
    ap.add_argument("--only", nargs="*", choices=list(DATASETS), default=["ddr"],
                    help="default: DDR only (the revised plan uses one lesion "
                         "source). Pass `--only idrid ddr` for both.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for name, spec in DATASETS.items():
        if args.only and name not in args.only:
            continue
        fetch(name, spec, args.out)

    from dr.data.lesion_datasets import discover_all
    _, summary = discover_all(args.out)
    print("\n[lesion] annotations available:")
    for k, v in summary.items():
        print(f"         {k:6s}: {v['images']:5d} images  {v['masks_per_channel']}")
    total = sum(v["images"] for v in summary.values())
    if total == 0:
        sys.exit("no lesion annotations recovered")
    print(f"\n[next] python scripts/10_lesion_pretrain.py")


if __name__ == "__main__":
    main()
