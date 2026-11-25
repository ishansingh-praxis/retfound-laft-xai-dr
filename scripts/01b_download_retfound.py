#!/usr/bin/env python3
"""Step 01b - fetch the real RETFound CFP weights (gated on HuggingFace).

    HF_TOKEN=hf_xxx python scripts/01b_download_retfound.py

The repository `YukunZhou/RETFound_mae_natureCFP` is gated: create a HuggingFace account,
open the repo page once and request access, then create a read token and export it as
`HF_TOKEN`. The file is copied to `data/checkpoints/RETFound_MAE/pytorch_model.bin`, which is
where `BackboneCfg.retfound_ckpt` looks. Training later refuses to run if fewer than 95% of the
backbone tensors match, so a wrong file cannot silently train as "RETFound".
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO, FILE = "YukunZhou/RETFound_mae_natureCFP", "RETFound_mae_natureCFP.pth"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "data" / "checkpoints"
                    / "RETFound_MAE" / "pytorch_model.bin")
    a = ap.parse_args()
    if a.out.exists() and a.out.stat().st_size > 5e8:
        print(f"[retfound] already present: {a.out} ({a.out.stat().st_size / 1e6:.0f} MB)")
        return
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set. RETFound is gated: request access at "
                 f"https://huggingface.co/{REPO}, create a read token, then `export HF_TOKEN=...`.")
    from huggingface_hub import hf_hub_download
    try:
        src = hf_hub_download(repo_id=REPO, filename=FILE, token=token)
    except Exception as e:                                       # noqa: BLE001
        sys.exit(f"download failed: {type(e).__name__}: {e}\n"
                 f"If this is 401/403, your account has not been granted access to {REPO} yet.")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, a.out)
    print(f"[retfound] {a.out} ({a.out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
