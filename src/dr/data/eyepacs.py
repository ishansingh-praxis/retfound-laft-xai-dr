"""Real EyePACS / APTOS manifests, patient-level splits, and the memmap cache.

Three on-disk layouts are recognised by `build_manifest`, so the same
preprocessing step (02) serves the training cohort and the external cohort:

  1. grade-foldered   `<root>/**/<GradeName>/<image>.png`      (eyepacs-224)
  2. label CSV        `<root>/**/trainLabels.csv` + images     (eyepacs-hi)
  3. APTOS            `<root>/**/train.csv` (id_code,diagnosis) + train_images/

`split_by_patient` is the important one: each EyePACS patient contributes a left
and a right eye whose grades are strongly correlated, so a per-image split leaks
the same patient into train and test and inflates every number. Splitting on
patient id and stratifying on the patient's *worst* eye avoids that;
`02_preprocess.py` asserts afterwards that no patient crosses a split.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

GRADE_NAMES = ("No DR", "Mild", "Moderate", "Severe", "PDR")

# directory / label spellings seen across the public mirrors -> grade 0-4
_GRADE_ALIASES = {
    "0": 0, "no_dr": 0, "nodr": 0, "no dr": 0, "normal": 0, "healthy": 0,
    "1": 1, "mild": 1, "mild_dr": 1, "mild npdr": 1,
    "2": 2, "moderate": 2, "moderate_dr": 2, "moderate npdr": 2,
    "3": 3, "severe": 3, "severe_dr": 3, "severe npdr": 3,
    "4": 4, "pdr": 4, "proliferate_dr": 4, "proliferative_dr": 4,
    "proliferate dr": 4, "proliferative dr": 4,
}
_IMG_EXT = (".png", ".jpeg", ".jpg", ".tif", ".tiff")


def _grade_from_name(name: str) -> int | None:
    return _GRADE_ALIASES.get(str(name).strip().lower().replace("-", "_"))


def _patient_and_eye(stem: str) -> tuple[str, str]:
    """`10_left` -> ("10", "left"); anything else -> (stem, "unknown").

    EyePACS encodes the patient in the filename; APTOS ids are per-image, so
    each APTOS image is treated as its own patient (which is correct - the
    cohort has one photograph per eye with no pairing information).
    """
    low = stem.lower()
    for eye in ("left", "right"):
        for sep in ("_", "-", "."):
            suffix = f"{sep}{eye}"
            if low.endswith(suffix):
                return stem[: -len(suffix)], eye
    return stem, "unknown"


def _finalise(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        raise FileNotFoundError("no labelled images found")
    df = pd.DataFrame(rows)
    df["grade"] = df["grade"].astype(int)
    df = df.sort_values("image_id").drop_duplicates("image_id", keep="first")
    return df.reset_index(drop=True)


def _index_images(root: Path) -> dict[str, Path]:
    """Map image stem -> path for every image under `root` (first wins)."""
    out: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in _IMG_EXT and p.is_file():
            out.setdefault(p.stem, p)
    return out


def build_manifest(root: str | Path) -> pd.DataFrame:
    """Scan `root` and return columns: path, image_id, patient_id, eye, grade."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    # --- layout 2/3: a label CSV somewhere under root ---------------------
    csvs = [p for p in root.rglob("*.csv")
            if p.stem.lower() in ("trainlabels", "train", "labels", "train_1")]
    for csv_path in sorted(csvs):
        try:
            lab = pd.read_csv(csv_path)
        except Exception:                                   # noqa: BLE001
            continue
        cols = {c.lower().strip(): c for c in lab.columns}
        id_col = next((cols[c] for c in ("image", "id_code", "image_id", "name")
                       if c in cols), None)
        y_col = next((cols[c] for c in ("level", "diagnosis", "grade", "label")
                      if c in cols), None)
        if not id_col or not y_col:
            continue
        images = _index_images(root)
        if not images:
            continue
        rows = []
        for stem, grade in zip(lab[id_col].astype(str), lab[y_col]):
            p = images.get(stem) or images.get(Path(stem).stem)
            if p is None or pd.isna(grade):
                continue
            g = int(grade)
            if not 0 <= g <= 4:
                continue
            pid, eye = _patient_and_eye(p.stem)
            rows.append({"path": str(p), "image_id": p.stem,
                         "patient_id": pid, "eye": eye, "grade": g})
        if rows:
            return _finalise(rows)

    # --- layout 1: images foldered by grade name --------------------------
    rows = []
    for p in root.rglob("*"):
        if p.suffix.lower() not in _IMG_EXT or not p.is_file():
            continue
        g = _grade_from_name(p.parent.name)
        if g is None:
            continue
        pid, eye = _patient_and_eye(p.stem)
        rows.append({"path": str(p), "image_id": p.stem,
                     "patient_id": pid, "eye": eye, "grade": g})
    return _finalise(rows)


def subsample_stratified(df: pd.DataFrame, n: int, seed: int = 1337) -> pd.DataFrame:
    """Take ~`n` images while preserving the grade distribution.

    Sampling is done on *patients* so the patient-level split stays possible
    afterwards; a patient's images travel together.
    """
    if n >= len(df):
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    worst = df.groupby("patient_id")["grade"].max()
    frac = n / len(df)
    keep: list[str] = []
    for g, pool in worst.groupby(worst):
        ids = pool.index.to_numpy()
        rng.shuffle(ids)
        take = max(1, int(round(len(ids) * frac)))
        keep.extend(ids[:take].tolist())
    out = df[df.patient_id.isin(set(keep))]
    return out.reset_index(drop=True)


def split_by_patient(df: pd.DataFrame, fracs=(0.7, 0.15, 0.15),
                     seed: int = 1337) -> pd.DataFrame:
    """Assign train/val/test on patient id, stratified on the worst eye grade."""
    rng = np.random.default_rng(seed)
    worst = df.groupby("patient_id")["grade"].max()
    assign: dict[str, str] = {}
    f_tr, f_va = fracs[0], fracs[0] + fracs[1]
    for g, pool in worst.groupby(worst):
        ids = pool.index.to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        i_tr, i_va = int(round(n * f_tr)), int(round(n * f_va))
        if n >= 3:                        # guarantee val/test are non-empty
            i_tr = min(i_tr, n - 2)
            i_va = min(max(i_va, i_tr + 1), n - 1)
        for pid in ids[:i_tr]:
            assign[pid] = "train"
        for pid in ids[i_tr:i_va]:
            assign[pid] = "val"
        for pid in ids[i_va:]:
            assign[pid] = "test"
    out = df.copy()
    out["split"] = out.patient_id.map(assign).fillna("train")
    return out


# ---------------------------------------------------------------------------
# cache dataset
# ---------------------------------------------------------------------------
_Q_COLS = ("q_quality", "q_domain", "q_lesion", "q_blur", "q_illumination")


def _tiling_centres(size: int, n: int) -> np.ndarray:
    """Fixed fallback when a cache has no crops.npy: a regular grid of window centres."""
    side = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(side) + 0.5) / side * size
    pts = [(x, y) for y in ticks for x in ticks][:n]
    return np.asarray(pts, np.float32)


def photometric_jitter(img: np.ndarray, gain: float, bias: float) -> np.ndarray:
    """Brightness/contrast jitter applied ONLY inside the retinal circle.

    The cache stores the area outside the camera aperture as exactly 0. Adding a bias to the
    whole array turned that black surround grey by a random amount per image, which changes
    the visible edge of the field from sample to sample and is not a camera property.
    """
    inside = img.sum(axis=2, keepdims=True) > 0
    out = np.clip(img * gain + bias, 0.0, 1.0)
    return np.where(inside, out, 0.0).astype(img.dtype, copy=False)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: give every worker its own augmentation stream.

    The dataset keeps one numpy Generator for its flip / jitter / crop-jitter decisions. Forked workers each
    inherit a COPY of it, and train.py builds a new DataLoader every epoch, so without this every worker
    replays the same random stream in every epoch (the same position in the stream gets the same flip).
    torch hands each worker a distinct seed that also changes from one loader to the next.
    """
    import torch
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    ds = info.dataset
    if hasattr(ds, "rng"):
        ds.rng = np.random.default_rng(int(torch.initial_seed()) % (2 ** 63))


class CachedEyePACS:
    """Reads the memmaps written by `scripts/02_preprocess.py`.

    Cache layout: images.npy (N,S,S,3 uint8 RGB), lesions.npy (N,6,M,M uint8),
    anatomy.npy (N,5,M,M uint8), crops.npy (N,K,2 float32 pixel centres of the local
    lesion windows, in the S x S grid), meta.csv (one row per cached image, with
    `cache_row` = physical row in the arrays).

    `split=None` uses every cached row (how an external cohort is evaluated).

    Index spaces: `self.idx` holds *positional* indices into `self.meta` (callers
    subsample by overwriting it); `row_of(j)` maps one to the physical memmap row.

    Each item is model-ready: `image` is the global view at `cfg.preproc.global_size`
    in [0,1] (the model does its own normalisation), `crops` are `n_crops` windows of
    `crop_size` cache pixels cut at native resolution around the stored centres and
    resized to `crop_input`.
    """

    def __init__(self, cache_dir: str | Path, split: str | None = "train",
                 augment: bool = False, cfg=None, seed: int = 1337):
        from ..config import default_config
        self.cfg = cfg if cfg is not None else default_config()
        self.dir = Path(cache_dir)
        meta_path = self.dir / "meta.csv"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"no cache at {self.dir} - run scripts/02_preprocess.py first")
        self.meta = pd.read_csv(meta_path).reset_index(drop=True)
        if "cache_row" not in self.meta.columns:
            self.meta["cache_row"] = np.arange(len(self.meta))
        self.meta_full = self.meta
        self.split = split
        self.augment = bool(augment)
        self.rng = np.random.default_rng(seed)

        self.images = np.load(self.dir / "images.npy", mmap_mode="r")
        self.lesions = np.load(self.dir / "lesions.npy", mmap_mode="r")
        self.anatomy = np.load(self.dir / "anatomy.npy", mmap_mode="r")
        crops_path = self.dir / "crops.npy"
        self.crop_centres = (np.load(crops_path, mmap_mode="r")
                             if crops_path.exists() else None)

        if split is None or "split" not in self.meta.columns:
            sel = np.arange(len(self.meta))
        else:
            sel = np.where(self.meta["split"].to_numpy() == split)[0]
        self.idx = sel.astype(np.int64)

        # indexed by cache_row, which has gaps when an image failed preprocessing: size by the largest row index
        self._hardness = np.zeros(int(self.meta["cache_row"].max()) + 1, np.float32)
        have_q = all(c in self.meta.columns for c in _Q_COLS)
        self._qaxes = (self.meta[list(_Q_COLS)].to_numpy(np.float32) if have_q
                       else np.zeros((len(self.meta), 5), np.float32))
        self._domain = (self.meta["domain_id"].to_numpy(np.int64)
                        if "domain_id" in self.meta.columns else None)

    # -- introspection ------------------------------------------------------
    def __len__(self) -> int:
        return len(self.idx)

    def row_of(self, j: int) -> int:
        """Positional meta index -> physical memmap row."""
        return int(self.meta["cache_row"].iat[int(j)])

    def labels(self) -> np.ndarray:
        return self.meta["grade"].to_numpy()[self.idx].astype(np.int64)

    def meta_indices(self) -> np.ndarray:
        """`cache_row` of every selected sample (what the hard-example miner indexes by)."""
        return self.meta["cache_row"].to_numpy()[self.idx].astype(np.int64)

    def class_counts(self, n_grades: int = 5) -> np.ndarray:
        return np.bincount(self.labels(), minlength=n_grades).astype(np.float64)

    def set_hardness(self, values: np.ndarray) -> None:
        self._hardness = np.asarray(values, np.float32)

    # -- crops ----------------------------------------------------------------
    def _centres(self, r: int, S: int, k: int) -> np.ndarray:
        if self.crop_centres is None:
            return _tiling_centres(S, k)
        c = np.asarray(self.crop_centres[r], np.float32)
        return c[:k] if len(c) >= k else np.concatenate(
            [c, _tiling_centres(S, k - len(c))])

    def _cut(self, img: np.ndarray, centres: np.ndarray, jitter: bool) -> np.ndarray:
        pc = self.cfg.preproc
        S = img.shape[0]
        cs = int(pc.crop_size)
        half = cs // 2
        out = np.zeros((len(centres), pc.crop_input, pc.crop_input, 3), np.float32)
        for i, (cx, cy) in enumerate(centres):
            if jitter and pc.crop_jitter > 0:
                cx += self.rng.uniform(-1, 1) * pc.crop_jitter * cs
                cy += self.rng.uniform(-1, 1) * pc.crop_jitter * cs
            x0 = int(np.clip(round(float(cx)) - half, 0, max(0, S - cs)))
            y0 = int(np.clip(round(float(cy)) - half, 0, max(0, S - cs)))
            win = img[y0:y0 + cs, x0:x0 + cs]
            out[i] = cv2.resize(win, (pc.crop_input, pc.crop_input),
                                interpolation=cv2.INTER_AREA)
        return out

    # -- item -------------------------------------------------------------------
    def item(self, j: int, pixel_override: np.ndarray | None = None,
             augment: bool | None = None) -> dict:
        """Model-ready sample for positional meta index `j`.

        `pixel_override` (S,S,3 uint8 RGB) replaces the cached photograph - used by the
        robustness corruptions and the counterfactual lesion erasure in 04_evaluate.
        """
        pc = self.cfg.preproc
        aug = self.augment if augment is None else augment
        j = int(j)
        r = self.row_of(j)
        src = pixel_override if pixel_override is not None else np.asarray(self.images[r])
        img = src.astype(np.float32) / 255.0
        les = np.asarray(self.lesions[r]).astype(np.float32) / 255.0
        ana = np.asarray(self.anatomy[r]).astype(np.float32) / 255.0
        S = img.shape[0]
        k = int(pc.n_crops)
        centres = self._centres(r, S, k).copy()

        if aug:
            if self.rng.random() < 0.5:                       # horizontal flip
                img = img[:, ::-1]
                les, ana = les[:, :, ::-1], ana[:, :, ::-1]
                centres[:, 0] = (S - 1) - centres[:, 0]
            if self.rng.random() < 0.5:                       # photometric jitter
                img = photometric_jitter(img, self.rng.uniform(0.9, 1.1),
                                         self.rng.uniform(-0.05, 0.05))
        img = np.ascontiguousarray(img)

        crops = self._cut(img, centres, jitter=aug)           # (K,Hc,Wc,3)
        g = pc.global_size
        glob = img if S == g else cv2.resize(img, (g, g), interpolation=cv2.INTER_AREA)

        out = {
            "image": np.ascontiguousarray(glob.transpose(2, 0, 1), np.float32),
            "crops": np.ascontiguousarray(crops.transpose(0, 3, 1, 2), np.float32),
            "lesion_masks": np.ascontiguousarray(les, np.float32),
            "anatomy": np.ascontiguousarray(ana, np.float32),
            "quality_axes": self._qaxes[j].copy(),
            "hardness": np.float32(self._hardness[r]),
            "grade": int(self.meta["grade"].iat[j]),
            "index": j,
            "cache_row": r,
        }
        if self._domain is not None:
            out["domain_id"] = int(self._domain[j])
        return out

    def __getitem__(self, i: int) -> dict:
        return self.item(int(self.idx[int(i)]))


def batch_from_rows(ds: CachedEyePACS, rows, device, pixel_override=None) -> dict:
    """Collate the given positional meta indices into one batch on `device`.

    `pixel_override` is an (n,S,S,3) uint8 array aligned with `rows`.
    """
    import torch
    from torch.utils.data import default_collate
    items = [ds.item(int(j), None if pixel_override is None else pixel_override[n],
                     augment=False)
             for n, j in enumerate(rows)]
    b = default_collate(items)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
