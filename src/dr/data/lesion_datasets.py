"""Real pixel-level lesion annotations (DDR, IDRiD, FGADR) for A3 pretraining and XAI scoring.

The public mirrors of these corpora use different folder layouts, so discovery is
**layout-agnostic**: every file under `<root>/<source>/` is classified from its path
tokens, and masks are paired with images on (source, split, normalised filename stem).

  channel   from the path or filename:  MA / microaneurysm, HE / haemorrhage,
            EX / hard exudate -> EX_H, SE / soft exudate -> EX_S.  Optic-disc masks and
            anything else are ignored.
  split     train / test / val tokens in the path ("a. training set", "valid", ...).
            A mask pairs only with an image from the SAME split; an image from a different
            split with the same stem is used only as a fallback when it is unique.
  mask      a file is a mask if its path names a lesion channel or contains
            groundtruth / label / mask / annotation; everything else is an image.

NV and ME are never annotated by these corpora, so they are never marked valid and must be
excluded from any loss computed against them (see `lesion_supervision_loss(valid=...)`).
A channel the SOURCE annotates but that has no mask file for an image is a confirmed
negative for that image (valid = 1, empty mask) - that distinction is why `valid` exists.
"""
from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

LESION_NAMES = ("MA", "HE", "EX_H", "EX_S", "NV", "ME")
ANNOTATED_CHANNELS = ("MA", "HE", "EX_H", "EX_S")
_IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

_CHANNEL_RES = (
    ("MA", re.compile(r"microaneurysm|(?<![a-z])ma(?![a-z])")),
    ("HE", re.compile(r"h[a]?emorrhage|(?<![a-z])he(?![a-z])")),
    ("EX_H", re.compile(r"hard[ _-]?exudate|(?<![a-z])ex(?![a-z])")),
    ("EX_S", re.compile(r"soft[ _-]?exudate|cotton|(?<![a-z])se(?![a-z])")),
)
_IGNORE_RE = re.compile(r"optic|(?<![a-z])od(?![a-z])|disc")
_MASK_WORD_RE = re.compile(r"groundtruth|ground[ _-]?truth|label|mask|annotation")
_SUFFIX_RE = re.compile(r"[_-](ma|he|ex|se|od|mask|label)$")
_MEAN = np.float32([0.485, 0.456, 0.406])
_STD = np.float32([0.229, 0.224, 0.225])


@dataclass
class LesionRecord:
    source: str
    image: Path
    masks: dict                      # channel -> mask path
    split: str = ""
    valid_channels: tuple = ANNOTATED_CHANNELS

    def key(self) -> str:
        return f"{self.source}/{self.image.stem}"


def _split_of(parts: list[str]) -> str:
    for p in parts:
        if re.search(r"train", p):
            return "train"
    for p in parts:
        if re.search(r"test", p):
            return "test"
    for p in parts:
        if re.search(r"(?<![a-z])val", p):
            return "val"
    return ""


def _classify(rel: Path) -> tuple[str, str | None]:
    """-> ('image'|'mask'|'skip', channel)."""
    low = rel.as_posix().lower()
    parts = low.split("/")
    if _IGNORE_RE.search(low):
        return "skip", None
    # look at directory names and the filename stem separately so 'image12' style names
    # cannot masquerade as a channel token
    hay = "/".join(parts[:-1]) + "/" + Path(parts[-1]).stem
    for ch, rx in _CHANNEL_RES:
        if rx.search(hay):
            return "mask", ch
    if _MASK_WORD_RE.search(low):
        return "skip", None
    return "image", None


def _stem_key(p: Path) -> str:
    s = p.stem.lower()
    while True:
        n = _SUFFIX_RE.sub("", s)
        if n == s:
            return s
        s = n


def discover_all(root: str | Path) -> tuple[list[LesionRecord], dict]:
    """Find every annotated image under `root` -> (records, summary).

    summary[source] = {"images": n, "masks_per_channel": {channel: count}}
    """
    root = Path(root)
    if not root.exists():
        return [], {}
    sources: dict[str, Path] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        sources[d.name.lower()] = d
    if not sources:
        sources = {"unknown": root}

    records: list[LesionRecord] = []
    summary: dict = {}
    for source, sdir in sources.items():
        images: dict[tuple[str, str], Path] = {}
        masks: dict[tuple[str, str], dict[str, Path]] = {}
        for p in sorted(sdir.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in _IMG_EXT:
                continue
            rel = p.relative_to(sdir)
            kind, ch = _classify(rel)
            if kind == "skip":
                continue
            split = _split_of([x.lower() for x in rel.parts[:-1]])
            key = (split, _stem_key(p))
            if kind == "image":
                prev = images.get(key)
                if prev is None or (prev.suffix.lower() not in (".jpg", ".jpeg")
                                    and p.suffix.lower() in (".jpg", ".jpeg")):
                    images[key] = p
            else:
                masks.setdefault(key, {})[ch] = p

        by_stem: dict[str, list] = {}
        for (sp, st), path in images.items():
            by_stem.setdefault(st, []).append((sp, path))
        src_records: list[LesionRecord] = []
        for key, chans in masks.items():
            sp, st = key
            img = images.get(key)
            if img is None:                                 # unique-stem fallback
                cands = by_stem.get(st, [])
                if len(cands) == 1:
                    sp, img = cands[0]
            if img is None:
                continue
            src_records.append(LesionRecord(source, img, dict(chans), sp))
        src_records.sort(key=lambda r: str(r.image))

        present = sorted({c for r in src_records for c in r.masks},
                         key=LESION_NAMES.index)
        for r in src_records:
            r.valid_channels = tuple(present)
        counts = {c: sum(1 for r in src_records if c in r.masks) for c in present}
        if src_records:
            summary[source] = {"images": len(src_records), "masks_per_channel": counts}
        records.extend(src_records)
    return records, summary


def split_records(records: list[LesionRecord], seed: int = 1337,
                  val_frac: float = 0.2) -> tuple[list[LesionRecord], list[LesionRecord]]:
    """Deterministic train / held-out split, stratified by source.

    The held-out part is what XAI scoring must use: those images were never seen by the
    lesion encoder's training.
    """
    rng = np.random.default_rng(seed)
    train, val = [], []
    for source in sorted({r.source for r in records}):
        rs = sorted((r for r in records if r.source == source), key=lambda r: str(r.image))
        order = rng.permutation(len(rs))
        n_val = int(round(len(rs) * val_frac))
        if len(rs) >= 2:
            n_val = min(max(n_val, 1), len(rs) - 1)
        else:
            n_val = 0
        val_ids = set(order[:n_val].tolist())
        for i, r in enumerate(rs):
            (val if i in val_ids else train).append(r)
    return train, val


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
class _CoverageMixin:
    """channel_coverage() / lesion_prevalence for both dataset types."""
    _valid: np.ndarray           # (N, 6)
    _prev: np.ndarray            # (6,) mean positive-pixel fraction over valid samples

    def channel_coverage(self) -> dict:
        return {n: int(self._valid[:, i].sum()) for i, n in enumerate(LESION_NAMES)}

    @property
    def lesion_prevalence(self) -> dict:
        return {n: float(self._prev[i]) for i, n in enumerate(LESION_NAMES)
                if n in ANNOTATED_CHANNELS and self._valid[:, i].sum() > 0}


def _norm(rgb01: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(((rgb01 - _MEAN) / _STD).transpose(2, 0, 1), np.float32)


class LesionSegmentationDataset(_CoverageMixin):
    """Whole-image mode: the full photograph resized to `image_size`."""

    def __init__(self, records, image_size: int, mask_size: int, augment: bool = False,
                 seed: int = 1337):
        from ..modules.external_lesion_batches import resize_like_field
        self._resize = resize_like_field
        self.records, self.S, self.M = list(records), int(image_size), int(mask_size)
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self._valid = np.zeros((len(self.records), len(LESION_NAMES)), np.float32)
        for i, r in enumerate(self.records):
            for c in r.valid_channels:
                self._valid[i, LESION_NAMES.index(c)] = 1.0
        self._prev = np.zeros(len(LESION_NAMES), np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        r = self.records[int(i)]
        bgr = cv2.imread(str(r.image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"unreadable image {r.image}")
        fov = (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) > 12).astype(np.uint8)
        rgb = cv2.cvtColor(self._resize(bgr, fov, self.S, cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        m = np.zeros((len(LESION_NAMES), self.M, self.M), np.float32)
        for ch, mp in r.masks.items():
            mk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if mk is not None:
                m[LESION_NAMES.index(ch)] = (self._resize(mk, fov, self.M,
                                                          cv2.INTER_NEAREST) > 0)
        if self.augment and self.rng.random() < 0.5:
            rgb, m = rgb[:, ::-1], m[:, :, ::-1]
        return {"image": _norm(np.ascontiguousarray(rgb)),
                "masks": np.ascontiguousarray(m), "valid": self._valid[int(i)].copy()}


def _extract_one(args: tuple):
    """Worker: one annotated image -> (crops, masks, valid) windows."""
    (rec, field_fraction, crop_input, mask_size, n_pos, n_neg, seed) = args
    from ..config import default_config
    from ..modules import a1_quality, a2_preprocess
    from ..modules.external_lesion_batches import resize_like_field
    rng = np.random.default_rng(seed)
    bgr = cv2.imread(str(rec.image), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    cfg = default_config()
    fov0 = (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) > 12).astype(np.uint8)
    pre = a2_preprocess.run(bgr, a1_quality.assess(bgr, cfg.quality), cfg.preproc)
    field_rgb = cv2.cvtColor(pre.image, cv2.COLOR_BGR2RGB)
    S = field_rgb.shape[0]
    full = np.zeros((len(LESION_NAMES), S, S), np.uint8)
    for ch, mp in rec.masks.items():
        mk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mk is not None:
            full[LESION_NAMES.index(ch)] = (resize_like_field(mk, fov0, S,
                                                              cv2.INTER_NEAREST) > 0)
    win = max(8, int(round(field_fraction * S)))
    half = win // 2
    ann = np.where(full.any(0))
    centres = []
    if ann[0].size:
        for _ in range(n_pos):
            k = int(rng.integers(0, ann[0].size))
            centres.append((int(ann[1][k]), int(ann[0][k])))
    fov = pre.fov_mask > 0
    ys, xs = np.where(fov)
    for _ in range(n_neg):
        for _try in range(30):
            k = int(rng.integers(0, ys.size)) if ys.size else 0
            cx, cy = (int(xs[k]), int(ys[k])) if ys.size else (S // 2, S // 2)
            x0, y0 = int(np.clip(cx - half, 0, S - win)), int(np.clip(cy - half, 0, S - win))
            if not full[:, y0:y0 + win, x0:x0 + win].any():
                centres.append((cx, cy))
                break
    valid = np.zeros(len(LESION_NAMES), np.uint8)
    for c in rec.valid_channels:
        valid[LESION_NAMES.index(c)] = 1
    xs_out, ms_out = [], []
    for cx, cy in centres:
        cx += int(rng.integers(-win // 8, win // 8 + 1))
        cy += int(rng.integers(-win // 8, win // 8 + 1))
        x0, y0 = int(np.clip(cx - half, 0, S - win)), int(np.clip(cy - half, 0, S - win))
        xs_out.append(cv2.resize(field_rgb[y0:y0 + win, x0:x0 + win],
                                 (crop_input, crop_input), interpolation=cv2.INTER_AREA))
        ms_out.append(np.stack([cv2.resize(full[c, y0:y0 + win, x0:x0 + win],
                                           (mask_size, mask_size),
                                           interpolation=cv2.INTER_NEAREST)
                                for c in range(len(LESION_NAMES))]))
    if not xs_out:
        return None
    return np.stack(xs_out), np.stack(ms_out), np.tile(valid, (len(xs_out), 1))


def build_crop_cache(records, out_dir: str | Path, field_fraction: float, crop_input: int,
                     mask_size: int, crops_per_image: int = 8, negative_frac: float = 0.25,
                     workers: int = 1, seed: int = 1337) -> dict:
    """Cut native-resolution windows (positives on lesions, plus negatives) into a cache.

    Windows come from the same A1/A2-preprocessed retinal field the DR model sees, at the
    same fraction of the field, so the lesion experts are pretrained at the magnification
    they will run at.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_neg = int(round(crops_per_image * negative_frac))
    n_pos = max(1, crops_per_image - n_neg)
    jobs = [(r, field_fraction, crop_input, mask_size, n_pos, n_neg, seed + i)
            for i, r in enumerate(records)]
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_extract_one, jobs, chunksize=4))
    else:
        results = [_extract_one(j) for j in jobs]
    ok = [r for r in results if r is not None]
    if not ok:
        raise RuntimeError("no usable crops extracted - check the lesion annotations")
    np.save(out / "crops_x.npy", np.concatenate([r[0] for r in ok]).astype(np.uint8))
    np.save(out / "crops_m.npy", np.concatenate([r[1] for r in ok]).astype(np.uint8))
    np.save(out / "crops_v.npy", np.concatenate([r[2] for r in ok]).astype(np.uint8))
    n = int(sum(len(r[0]) for r in ok))
    return {"images": len(records), "images_used": len(ok), "crops": n,
            "positives_per_image": n_pos, "negatives_per_image": n_neg}


class CachedLesionCrops(_CoverageMixin):
    """Reads the cache written by `build_crop_cache`."""

    def __init__(self, cache_dir: str | Path, augment: bool = False, seed: int = 1337):
        d = Path(cache_dir)
        self.x = np.load(d / "crops_x.npy", mmap_mode="r")
        self.m = np.load(d / "crops_m.npy", mmap_mode="r")
        self._valid = np.load(d / "crops_v.npy").astype(np.float32)
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        mm = np.asarray(self.m, np.float32)
        frac = mm.reshape(len(mm), len(LESION_NAMES), -1).mean(2)
        v = np.maximum(self._valid.sum(0), 1.0)
        self._prev = ((frac * self._valid).sum(0) / v).astype(np.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> dict:
        rgb = np.asarray(self.x[int(i)]).astype(np.float32) / 255.0
        m = np.asarray(self.m[int(i)]).astype(np.float32)
        if self.augment:
            if self.rng.random() < 0.5:
                rgb, m = rgb[:, ::-1], m[:, :, ::-1]
            if self.rng.random() < 0.5:
                rgb, m = rgb[::-1], m[:, ::-1]
        return {"image": _norm(np.ascontiguousarray(rgb)),
                "masks": np.ascontiguousarray(m), "valid": self._valid[int(i)].copy()}
