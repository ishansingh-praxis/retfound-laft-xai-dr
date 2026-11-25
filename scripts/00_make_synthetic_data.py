#!/usr/bin/env python3
"""Step 00 - generate a tiny SYNTHETIC fundus dataset for smoke-testing the pipeline.

Not real data and not for any result: it exists so that `run_all.py --smoke` can exercise
every stage (manifest -> cache -> lesion pretraining -> staged training -> evaluation ->
screening -> reports) in minutes on a CPU and fail loudly if any interface is broken
before a long cloud run is started.

    python scripts/00_make_synthetic_data.py --out data/synthetic/eyepacs \
        --lesion-out data/synthetic/lesion --n-patients 48

Writes an EyePACS-layout folder (`trainLabels.csv` + `<id>_left.jpeg` / `<id>_right.jpeg`)
and, optionally, a DDR-like lesion-segmentation tree with per-channel masks.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CHANNELS = ("MA", "HE", "EX", "SE")


def fundus(size: int, grade: int, rng: np.random.Generator):
    """One synthetic fundus photograph and its per-channel lesion masks."""
    img = np.zeros((size, size, 3), np.uint8)
    c, r = size // 2, int(size * 0.46)
    yy, xx = np.mgrid[:size, :size]
    fov = ((yy - c) ** 2 + (xx - c) ** 2) <= r * r
    base = np.array([40, 90, 180]) + rng.integers(-12, 12, 3)
    img[fov] = np.clip(base, 0, 255)
    side = 1 if rng.random() < 0.5 else -1
    dx = int(c + side * r * 0.55)
    cv2.circle(img, (dx, c), max(3, int(size * 0.05)), (200, 235, 250), -1)
    for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        q = (int(dx + np.cos(a) * r * 0.85), int(c + np.sin(a) * r * 0.85))
        cv2.line(img, (dx, c), q, (25, 45, 120), 1)
    masks = {k: np.zeros((size, size), np.uint8) for k in CHANNELS}
    counts = {"MA": 3 * grade, "HE": 2 * grade, "EX": 2 * grade, "SE": max(0, grade - 1) * 2}
    colour = {"MA": (20, 35, 110), "HE": (25, 40, 150), "EX": (120, 230, 245), "SE": (190, 235, 235)}
    rad = {"MA": (1, 3), "HE": (3, 6), "EX": (2, 5), "SE": (5, 9)}
    for ch, n in counts.items():
        for _ in range(int(n)):
            ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(0, r * 0.8)
            x, y = int(c + np.cos(ang) * dist), int(c + np.sin(ang) * dist)
            rr = int(rng.integers(*rad[ch]))
            cv2.circle(img, (x, y), rr, colour[ch], -1)
            cv2.circle(masks[ch], (x, y), rr, 255, -1)
    img[~fov] = 0
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, masks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lesion-out", type=Path, default=None)
    ap.add_argument("--n-patients", type=int, default=48)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--n-lesion-images", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    probs = np.array([0.45, 0.12, 0.2, 0.12, 0.11])      # every grade present
    rows = []
    for p in range(args.n_patients):
        worst = int(rng.choice(5, p=probs))
        for eye in ("left", "right"):
            g = int(np.clip(worst + rng.integers(-1, 1), 0, 4))
            name = f"{1000 + p}_{eye}"
            img, _ = fundus(args.size, g, rng)
            cv2.imwrite(str(args.out / f"{name}.jpeg"), img)
            rows.append({"image": name, "level": g})
    pd.DataFrame(rows).to_csv(args.out / "trainLabels.csv", index=False)
    print(f"[synthetic] {len(rows)} images / {args.n_patients} patients -> {args.out}  "
          f"grades {pd.DataFrame(rows).level.value_counts().sort_index().to_dict()}")

    if args.lesion_out:
        base = args.lesion_out / "ddr" / "lesion_segmentation"
        splits = ["train"] * (args.n_lesion_images * 2 // 3) + ["valid"] * (args.n_lesion_images // 6) \
            + ["test"] * (args.n_lesion_images - args.n_lesion_images * 2 // 3 - args.n_lesion_images // 6)
        for i, sp in enumerate(splits):
            name = f"007-{i:04d}-000"
            img, masks = fundus(args.size, int(rng.integers(2, 5)), rng)
            (base / sp / "image").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(base / sp / "image" / f"{name}.jpg"), img)
            for ch, m in masks.items():
                (base / sp / "label" / ch).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(base / sp / "label" / ch / f"{name}.tif"), m)
        print(f"[synthetic] {len(splits)} lesion-annotated images -> {args.lesion_out}")


if __name__ == "__main__":
    main()
