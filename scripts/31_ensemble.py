#!/usr/bin/env python3
"""Average the class probabilities of several runs (seeds / backbones / resolutions).

    python scripts/31_ensemble.py --inputs seed1337/predictions_test_tta.npz \
        seed2024/predictions_test_tta.npz seed7/predictions_test_tta.npz \
        --out ensemble/predictions_test_tta.npz

Inputs are predictions_*.npz files written by 04_evaluate.py (keys: y, pred, probs,
patient_id, image_id). Only images present in EVERY input are kept, in a common order, and
their labels must agree. The ensemble's `probs` is the mean of the members' `probs`; `pred`
is the CORAL rank decode of that mean, matching 04_evaluate's default. Averaging probabilities
(not logits) is deliberate: the stored probs are already calibrated and TTA-averaged.

The output has the same keys as a single run, so scripts/26_final_evaluation.py and the
validation-fitted decoders (QWK cut-points, logit adjustment, eye fusion, screening
thresholds) work on it unchanged. Fit those on the ensemble's own validation file, never on
a single member's, because averaging changes both the score distribution and the calibration.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dr import stats as S  # noqa: E402
from dr.modules.a6_ordinal import coral_decode_numpy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if len(args.inputs) < 2:
        ap.error("need at least two prediction files to ensemble")

    runs = [S.load_predictions(p) for p in args.inputs]
    common = runs[0]["image_id"]
    for r in runs[1:]:
        common = np.intersect1d(common, r["image_id"])
    if len(common) == 0:
        sys.exit("no image appears in every input - are these the same split?")
    dropped = [len(r["image_id"]) - len(common) for r in runs]
    if any(dropped):
        print(f"[ensemble] keeping {len(common)} images common to all inputs "
              f"(dropped per input: {dropped})")

    def take(r: dict) -> dict:
        order = {k: i for i, k in enumerate(r["image_id"])}
        ix = np.array([order[k] for k in common])
        return {k: v[ix] for k, v in r.items()
                if isinstance(v, np.ndarray) and len(v) == len(r["image_id"])}

    runs = [take(r) for r in runs]
    y = runs[0]["y"]
    for p, r in zip(args.inputs[1:], runs[1:]):
        if not np.array_equal(r["y"], y):
            sys.exit(f"labels differ between {args.inputs[0]} and {p}: not the same images/split")
    probs = np.mean([r["probs"].astype(np.float64) for r in runs], axis=0)
    probs /= probs.sum(1, keepdims=True)
    pred = np.asarray(coral_decode_numpy(probs), dtype=np.int64)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, y=y.astype(np.int64), pred=pred, probs=probs.astype(np.float32),
                        patient_id=runs[0]["patient_id"], image_id=runs[0]["image_id"])
    qwk_members = [S.qwk(y, r["pred"]) for r in runs]
    print(f"[ensemble] {len(runs)} members, {len(y)} images -> {args.out}")
    print(f"[ensemble] QWK members {np.round(qwk_members, 4).tolist()}  "
          f"ensemble {S.qwk(y, pred):.4f} (raw CORAL decode; final cut-points are fitted later)")


if __name__ == "__main__":
    main()
