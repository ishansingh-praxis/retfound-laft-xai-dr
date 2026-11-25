#!/usr/bin/env python3
"""Step 27 - flag probable label errors (EyePACS Mild vs No DR is the worst case).

Confident-learning style, on saved predictions:

    python scripts/04_evaluate.py --ckpt outputs/<run>/best.pt --split train
    python scripts/27_flag_label_noise.py --pred outputs/<run>/predictions_train.npz \
        --out outputs/<run>/noise_weights.csv
    python scripts/03_train.py ... --noise-weights outputs/<run>/noise_weights.csv

A sample is flagged when the model assigns its GIVEN label less probability than
that class's typical self-confidence AND another class clears that class's own
threshold. Flagged samples get a sampling weight < 1 (default 0.3); nothing is
deleted and no label is changed.

Caveats, stated because they change how far this can be trusted:
  * predictions on TRAINING images are in-sample, so a model that has memorised
    some wrong labels will not flag them. K-fold out-of-fold predictions are
    better; this script works with either.
  * flagged != wrong. The flagged list is a review queue and a down-weighting
    signal, not ground truth. Report how many samples were affected.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def flag_label_issues(y: np.ndarray, probs: np.ndarray, n_grades: int = 5,
                      margin: float = 0.5):
    """Return (flagged_mask, suggested_label) using per-class confidence thresholds."""
    t = np.array([probs[y == k, k].mean() if (y == k).any() else 1.0
                  for k in range(n_grades)])
    self_conf = probs[np.arange(len(y)), y]
    suggested = y.copy()
    flagged = np.zeros(len(y), bool)
    for i in np.where(self_conf < t[y])[0]:
        # another class must clear its own threshold AND beat the given label by
        # `margin` - without the margin most overlapping-class images get flagged
        cand = [k for k in range(n_grades)
                if k != y[i] and probs[i, k] >= t[k]
                and probs[i, k] - self_conf[i] >= margin]
        if cand:
            suggested[i] = max(cand, key=lambda k: probs[i, k])
            flagged[i] = True
    return flagged, suggested, self_conf, t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True,
                    help="predictions_<split>.npz from scripts/04_evaluate.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--margin", type=float, default=0.5,
                    help="required probability gap between the suggested class and "
                         "the given label (larger = fewer, more certain flags)")
    ap.add_argument("--weight", type=float, default=0.3,
                    help="sampling weight given to flagged images (default 0.3)")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from dr.stats import load_predictions
    z = load_predictions(args.pred)
    y, probs, ids = z["y"], z["probs"], z["image_id"]
    flagged, suggested, self_conf, t = flag_label_issues(y, probs, margin=args.margin)

    df = pd.DataFrame({"image_id": ids, "given": y, "suggested": suggested,
                       "self_confidence": self_conf, "flagged": flagged})
    df["weight"] = np.where(df.flagged, args.weight, 1.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    # a human-reviewable queue, worst first
    df[df.flagged].sort_values("self_confidence").to_csv(
        args.out.with_name(args.out.stem + "_review_queue.csv"), index=False)

    print(f"[noise] {int(flagged.sum())} of {len(y)} images flagged "
          f"({100 * flagged.mean():.1f}%); per-class self-confidence thresholds "
          f"{np.round(t, 3).tolist()}")
    print("[noise] flagged by given grade:",
          df[df.flagged].given.value_counts().sort_index().to_dict())
    pairs = df[df.flagged].groupby(["given", "suggested"]).size().sort_values(ascending=False)
    print("[noise] most common given -> suggested:\n" + pairs.head(8).to_string())
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
