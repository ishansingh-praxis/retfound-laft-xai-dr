#!/usr/bin/env python3
"""Cheap pre-flight check of the preprocessing, to run BEFORE building the full cache.

    python scripts/32_preprocess_check.py --root data/raw/eyepacs-hi --n 200
    python scripts/32_preprocess_check.py --root data/raw/fdr26 --fdr26 --n 100

It runs the exact worker that scripts/02_preprocess.py uses (same A1 gate, A2 extraction,
weak priors and crop selection) on a small stratified sample and reports what a multi-hour cache
build would otherwise only reveal at the end:

  * how images route through the Q* gate (retake / enhance / accept), overall and per DR grade -
    if nearly everything lands in one route the thresholds are not doing anything on this corpus
  * the mean of each Q* axis, and lesion visibility by grade
  * whether the local-crop windows land on the retina (fraction of crop centres inside the field)
  * throughput, extrapolated to the cache size you intend to build
  * overlay images (crop windows, optic disc, macula, decision) so the crop placement can be
    judged by eye: <out>/overlays/*.jpg
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dr.cohort import load_fdr26_manifest  # noqa: E402
from dr.data.eyepacs import build_manifest  # noqa: E402
from dr.modules import a1_quality  # noqa: E402


def _load_worker():
    spec = importlib.util.spec_from_file_location("pre02", ROOT / "scripts" / "02_preprocess.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _windows(crops: np.ndarray, size: int, crop_px: int) -> list[tuple[int, int]]:
    """Top-left corner of each window, clamped like CachedEyePACS._cut (no jitter)."""
    half = crop_px // 2
    return [(int(np.clip(round(float(cx)) - half, 0, max(0, size - crop_px))),
             int(np.clip(round(float(cy)) - half, 0, max(0, size - crop_px)))) for cx, cy in crops]


def _overlay(rgb: np.ndarray, crops: np.ndarray, meta: dict, crop_px: int, path: Path) -> None:
    S = rgb.shape[0]
    view = 640
    k = view / S
    img = cv2.cvtColor(cv2.resize(rgb, (view, view), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
    for i, (x0, y0) in enumerate(_windows(crops, S, crop_px)):
        # exactly the window CachedEyePACS._cut() extracts (shifted to stay inside the image)
        cv2.rectangle(img, (int(x0 * k), int(y0 * k)), (int((x0 + crop_px) * k), int((y0 + crop_px) * k)),
                      (255, 255, 0), 1)
        cv2.putText(img, str(i), (int(x0 * k) + 3, int(y0 * k) + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.circle(img, (int(meta["disc_x"] * k), int(meta["disc_y"] * k)), 9, (0, 255, 255), 2)
    cv2.circle(img, (int(meta["macula_x"] * k), int(meta["macula_y"] * k)), 9, (0, 255, 0), 2)
    cv2.putText(img, f"{meta['quality_decision']}  Q*={meta['quality_score']:.2f}  "
                     f"Qlesion={meta['q_lesion']:.2f}  Qblur={meta['q_blur']:.2f}",
                (8, view - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--fdr26", action="store_true", help="--root is the FDR26 folder")
    ap.add_argument("--n", type=int, default=200, help="images to process (stratified by grade)")
    ap.add_argument("--overlays", type=int, default=12)
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument("--mask-size", type=int, default=112)
    ap.add_argument("--n-crops", type=int, default=8)
    ap.add_argument("--crop-size", type=int, default=None,
                    help="window in cache pixels (default 320 at 1024 px, scaled with --image-size)")
    ap.add_argument("--quality-reference", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "preprocess_check")
    ap.add_argument("--total-images", type=int, default=15000,
                    help="cache size to extrapolate the build time to")
    ap.add_argument("--workers", type=int, default=5, help="workers assumed in the time estimate")
    args = ap.parse_args()
    crop_px = args.crop_size or int(round(320 * args.image_size / 1024 / 8) * 8)

    df = load_fdr26_manifest(args.root) if args.fdr26 else build_manifest(args.root)
    per = max(1, args.n // max(1, df.grade.nunique()))
    sample = pd.concat([g.sample(min(len(g), per), random_state=args.seed)
                        for _, g in df.groupby("grade")]).reset_index(drop=True)
    print(f"[sample] {len(sample)} of {len(df)} images, per grade "
          f"{sample.grade.value_counts().sort_index().to_dict()}")

    if args.quality_reference:
        qref = json.loads(Path(args.quality_reference).read_text())
    else:
        qref = a1_quality.calibrate_quality_reference(sample["path"].tolist(),
                                                      sample=min(len(sample), 400), seed=args.seed)

    pre = _load_worker()
    pre._init_worker(args.image_size, args.mask_size, args.n_crops, crop_px, qref, False)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "overlays").mkdir(exist_ok=True)

    rows, inside_frac, win_cov, t0 = [], [], [], time.time()
    for i, (path, grade) in enumerate(zip(sample["path"], sample["grade"])):
        res = pre._process_one((i, path))
        if res[0] != "ok":
            print(f"[skip] {path}: {res[0]} {res[2]}")
            continue
        _, _, rgb, _les, _ana, crops, meta = res
        meta = {**meta, "grade": int(grade), "path": path}
        rows.append(meta)
        fov = rgb.sum(2) > 0
        xs = np.clip(crops[:, 0].astype(int), 0, rgb.shape[1] - 1)
        ys = np.clip(crops[:, 1].astype(int), 0, rgb.shape[0] - 1)
        inside_frac.append(float(fov[ys, xs].mean()))
        for x0, y0 in _windows(crops, rgb.shape[0], crop_px):
            win_cov.append(float(fov[y0:y0 + crop_px, x0:x0 + crop_px].mean()))
        if len(rows) <= args.overlays:
            _overlay(rgb, crops, meta, crop_px, args.out / "overlays" / f"{len(rows):02d}_g{grade}_{meta['quality_decision']}.jpg")
    el = time.time() - t0
    out = pd.DataFrame(rows)
    out.to_csv(args.out / "routing.csv", index=False)

    n = len(out)
    print(f"\n[A1 gate] routing over {n} images (thresholds retake<{pre._CFG.quality.tau_r}, "
          f"accept>={pre._CFG.quality.tau_d}):")
    for k, v in out.quality_decision.value_counts().items():
        print(f"          {k:8s}: {v:5d} ({100 * v / n:5.1f}%)")
    print("[A1 gate] share routed to each decision, by DR grade:")
    print(pd.crosstab(out.grade, out.quality_decision, normalize="index").round(2).to_string())
    print("[A1 gate] mean of each axis:",
          {c: round(float(out[c].mean()), 3) for c in
           ("quality_score", "q_quality", "q_domain", "q_lesion", "q_blur", "q_illumination")})
    print("[A1 gate] mean Q_lesion by grade:",
          {int(g): round(float(v), 3) for g, v in out.groupby("grade").q_lesion.mean().items()})
    if inside_frac:
        print(f"[crops] fraction of crop centres inside the retinal field: "
              f"mean {np.mean(inside_frac):.3f}, worst image {np.min(inside_frac):.3f}")
    if win_cov:
        wc = np.asarray(win_cov)
        print(f"[crops] share of each extracted window that is retina: mean {wc.mean():.3f}; "
              f"{100 * float((wc < 0.5).mean()):.1f}% of windows are less than half retina "
              f"(they are mostly black surround - a wasted local view)")
    rate = n / max(el, 1e-6)
    eta = args.total_images / (rate * max(1, args.workers)) / 60
    print(f"[speed] {rate:.2f} img/s on one core -> ~{eta:.0f} min for {args.total_images} images "
          f"with {args.workers} workers (rough: ignores disk and IPC)")
    flat = out.quality_decision.value_counts(normalize=True).max()
    if flat > 0.95:
        print(f"[WARNING] {100 * flat:.0f}% of images take one route: the Q* thresholds are not "
              f"discriminating on this corpus - review tau_r/tau_d before building the cache")
    print(f"[out] {args.out}/routing.csv and {args.out}/overlays/")


if __name__ == "__main__":
    main()
