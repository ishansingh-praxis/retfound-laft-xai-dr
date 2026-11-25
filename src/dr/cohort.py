"""Cohort helpers added for the revised methodology.

  minority_first_subsample   keep EVERY patient with any DR (worst eye >= 1) and
                             subsample only the No-DR-only patients. The old
                             stratified subsample kept the 36:1 imbalance and left
                             ~120 grade-4 images in a 6,000-image cache.
  load_fdr26_manifest        manifest for the external FDR26 cohort (Zenodo
                             record 23101200), same columns as build_manifest.
  save_split / load_split    freeze the patient -> split assignment once so every
                             later run, seed and baseline uses identical splits.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def minority_first_subsample(df: pd.DataFrame, n_total: int, seed: int = 1337) -> pd.DataFrame:
    """Keep all DR patients; fill the remaining budget with No-DR-only patients.

    If the DR images alone already exceed `n_total`, they are all kept anyway:
    the point is never to throw minority-class images away.
    """
    rng = np.random.default_rng(seed)
    worst = df.groupby("patient_id")["grade"].max()
    dr_patients = set(worst[worst >= 1].index)
    keep_df = df[df.patient_id.isin(dr_patients)]
    budget = n_total - len(keep_df)
    nodr = worst[worst == 0].index.to_numpy().copy()
    rng.shuffle(nodr)
    chosen: list[str] = []
    used = 0
    per_patient = df.groupby("patient_id").size()
    for pid in nodr:
        if used >= budget:
            break
        chosen.append(pid)
        used += int(per_patient[pid])
    out = pd.concat([keep_df, df[df.patient_id.isin(set(chosen))]])
    return out.sort_values("image_id").reset_index(drop=True)


def load_fdr26_manifest(root: str | Path,
                        metadata_xlsx: str | Path | None = None) -> pd.DataFrame:
    """Manifest for FDR26: <root>/images/<image_id>.jpg + metadata_classification_DR.xlsx.

    `patient_id` is the first 8 characters of `image_id`. That is an INFERENCE from
    the naming pattern (up to 8 images share a prefix), not something the dataset
    documents - FDR26 is used as an external test cohort only, so a wrong guess
    cannot leak into training, but do not rely on it for patient-level claims.
    Metadata rows with no matching file (one known case) are dropped and reported.
    """
    root = Path(root)
    meta_path = Path(metadata_xlsx) if metadata_xlsx else root / "metadata_classification_DR.xlsx"
    if not meta_path.exists():
        raise FileNotFoundError(f"FDR26 metadata not found: {meta_path}")
    meta = pd.read_excel(meta_path)
    img_dir = root / "images"
    index: dict[str, Path] | None = None       # built lazily: only if the expected folder layout is not there
    rows, dropped = [], []
    for image_id, grade in zip(meta["image_id"].astype(str), meta["classe"]):
        p = img_dir / f"{image_id}.jpg"
        if not p.exists():
            if index is None:
                index = {q.stem: q for q in root.rglob("*")
                         if q.is_file() and q.suffix.lower() in (".jpg", ".jpeg", ".png")}
            p = index.get(image_id)
        if p is None or not Path(p).exists():
            dropped.append(image_id)
            continue
        rows.append({"path": str(p), "image_id": image_id,
                     "patient_id": image_id[:8], "eye": "unknown", "grade": int(grade)})
    if dropped:
        print(f"[fdr26] {len(dropped)} metadata row(s) have no image file and were "
              f"dropped: {dropped[:5]}")
    return pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)


def save_split(df: pd.DataFrame, path: str | Path) -> None:
    """Write {patient_id: split} so the split can be reused verbatim."""
    m = df.drop_duplicates("patient_id").set_index("patient_id")["split"].to_dict()
    Path(path).write_text(json.dumps(m, indent=0))


def load_split(df: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Apply a frozen split. Patients absent from the file are an error, not a guess."""
    m = json.loads(Path(path).read_text())
    missing = sorted(set(df.patient_id) - set(m))
    if missing:
        raise ValueError(f"{len(missing)} patients are not in the frozen split "
                         f"{path} (e.g. {missing[:3]}); rebuild the split deliberately.")
    out = df.copy()
    out["split"] = out.patient_id.map(m)
    return out


def load_kaggle_test_manifest(root: str | Path, labels_csv: str | Path) -> pd.DataFrame:
    """Official labelled Kaggle DR test set (the split the published QWK ~0.85 uses).

    `labels_csv` is the competition's `retinopathy_solution.csv` (columns `image`,
    `level`, optional `Usage`). Image stems look like `1234_left`. Rows whose image
    file cannot be found under `root` are dropped and counted. NOTE: this is ~53k
    images; cached at 1024 px that is ~170 GB, so use `--limit` (a stratified
    subsample) unless you have the disk - and say which you did in the paper.
    """
    root = Path(root)
    lab = pd.read_csv(labels_csv)
    cols = {c.lower().strip(): c for c in lab.columns}
    id_col, y_col = cols.get("image"), cols.get("level")
    if not id_col or not y_col:
        raise ValueError(f"{labels_csv}: expected columns 'image' and 'level', got {list(lab.columns)}")
    index: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".jpeg", ".jpg", ".png", ".tif", ".tiff"):
            index.setdefault(p.stem, p)
    rows, dropped = [], 0
    for stem, grade in zip(lab[id_col].astype(str), lab[y_col]):
        p = index.get(stem)
        if p is None or pd.isna(grade) or not 0 <= int(grade) <= 4:
            dropped += 1
            continue
        pid, eye = stem, "unknown"
        for e in ("left", "right"):
            if stem.lower().endswith("_" + e):
                pid, eye = stem[: -(len(e) + 1)], e
        rows.append({"path": str(p), "image_id": stem, "patient_id": pid,
                     "eye": eye, "grade": int(grade)})
    if dropped:
        print(f"[kaggle-test] {dropped} label rows had no image file / invalid grade")
    if not rows:
        raise FileNotFoundError(f"no labelled test images found under {root}")
    return pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
