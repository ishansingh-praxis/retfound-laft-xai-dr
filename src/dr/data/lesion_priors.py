"""Weak lesion + anatomy priors, derived from pixels by classical morphology.

EyePACS ships **no pixel-level lesion annotations**, so the six lesion maps and
five anatomical region maps used by A3 (mixture-of-experts supervision), A5
(pathology-anatomy graph edges) and A9 (attribution Dice) are produced here from
the image itself.

    Every lesion / XAI metric in this project therefore measures agreement with
    a morphological prior, not with an ophthalmologist.

Substitute real masks (IDRiD, DDR, FGADR) to turn these into true supervised
signals - the interface (`LesionPriorExtractor.__call__ -> LesionPriors`) is
unchanged, only the mask source differs.

Known weaknesses, stated so they are not mistaken for ground truth:
  * `NV` over-fires on dense normal vasculature - it keys on fine-vessel
    density, which is exactly what healthy arcades look like at low resolution.
  * `ME` is a circinate-exudate proxy: true macular oedema is a retinal
    thickening that colour fundus photography cannot see (it needs OCT).
  * At 224 px a microaneurysm is ~1 px, so `MA` is dominated by noise. The
    detectors are written for the ~1024 px mirror (`--dataset eyepacs-hi`).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

LESION_NAMES = ("MA", "HE", "EX_H", "EX_S", "NV", "ME")
REGION_NAMES = ("optic_disc", "macula", "sup_arcade", "inf_arcade", "peripheral")
ANATOMY_NAMES = REGION_NAMES        # alias used by a9_xai


@dataclass
class LesionPriors:
    """Weak labels for one image.

    masks    : (6, M, M) float32 [0,1], ordered as LESION_NAMES
    anatomy  : (5, M, M) float32 [0,1], ordered as REGION_NAMES
    """
    masks: np.ndarray
    anatomy: np.ndarray
    disc_center: tuple[int, int]      # (x, y) in source-image pixels
    macula_center: tuple[int, int]     # (x, y) in source-image pixels

    def burden(self) -> np.ndarray:
        """Mean activation per lesion channel - the 'lesion burden' per image."""
        return self.masks.reshape(len(LESION_NAMES), -1).mean(1).astype(np.float32)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _norm_inside(x: np.ndarray, fov: np.ndarray) -> np.ndarray:
    """Min-max normalise using only in-FOV pixels, then zero the outside."""
    inside = x[fov > 0]
    if inside.size:
        x = (x - inside.min()) / (np.ptp(inside) + 1e-6)
    x = np.clip(x, 0, 1).astype(np.float32)
    x[fov == 0] = 0.0
    return x


def _tophat(chan: np.ndarray, radius: int, dark: bool) -> np.ndarray:
    """White (bright) or black (dark) top-hat with an elliptical element."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    op = cv2.MORPH_BLACKHAT if dark else cv2.MORPH_TOPHAT
    return cv2.morphologyEx(chan, op, k).astype(np.float32)


def _vessel_response(green: np.ndarray, scales=(7, 11, 15)) -> np.ndarray:
    """Elongated dark response: high on vessels, low on round lesions.

    A line element is rotated through 6 orientations at several lengths; taking
    the max over orientations keeps vessels of any direction while round blobs -
    which never fill a long thin element - stay low.
    """
    acc = np.zeros(green.shape, np.float32)
    for length in scales:
        base = cv2.getStructuringElement(cv2.MORPH_RECT, (length * 2 + 1, 1))
        c = length
        for ang in range(0, 180, 30):
            M = cv2.getRotationMatrix2D((float(c), 0.0), ang, 1.0)
            kk = cv2.warpAffine(base.astype(np.float32), M,
                                (length * 2 + 1, length * 2 + 1)) > 0.5
            kk = kk.astype(np.uint8)
            if kk.sum() < 3:
                continue
            acc = np.maximum(acc, cv2.morphologyEx(green, cv2.MORPH_BLACKHAT,
                                                   kk).astype(np.float32))
    return acc


def _gaussian_blob(shape: tuple[int, int], cx: float, cy: float,
                   sigma: float) -> np.ndarray:
    """Soft circular region map centred at (cx, cy)."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return np.exp(-d2 / (2.0 * sigma * sigma + 1e-6)).astype(np.float32)


def locate_optic_disc(bgr: np.ndarray, fov: np.ndarray) -> tuple[int, int, float]:
    """Brightest large blob inside an eroded FOV -> (x, y, disc_radius_px).

    The disc is the brightest structure in a fundus photograph, but so is the
    FOV rim in many EyePACS cameras, so the mask is eroded before searching.
    """
    h, w = fov.shape
    disc_r = max(3.0, 0.075 * max(h, w))            # disc diameter ~ 15% of width
    er = max(3, int(disc_r))
    inner = cv2.erode(fov, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er, er)))
    if inner.sum() < 16:
        inner = fov
    bright = (0.5 * bgr[:, :, 1].astype(np.float32)
              + 0.5 * bgr[:, :, 2].astype(np.float32))
    bright = cv2.GaussianBlur(bright, (0, 0), disc_r * 0.5)
    bright[inner == 0] = -1.0
    y, x = np.unravel_index(int(np.argmax(bright)), bright.shape)
    return int(x), int(y), float(disc_r)


def locate_macula(bgr: np.ndarray, fov: np.ndarray, disc: tuple[int, int, float]
                  ) -> tuple[int, int]:
    """Fovea estimate: ~2.5 disc diameters from the disc, toward the FOV centre.

    The direction is inferred from which side of the field the disc sits on
    (temporal/nasal), then refined to the darkest smoothed spot in a window,
    since the fovea is the darkest part of a healthy retina.
    """
    dx, dy, disc_r = disc
    ys, xs = np.where(fov > 0)
    if ys.size == 0:
        return int(bgr.shape[1] // 2), int(bgr.shape[0] // 2)
    cx, cy = float(xs.mean()), float(ys.mean())
    direction = 1.0 if dx < cx else -1.0             # head toward the field centre
    mx = dx + direction * 5.0 * disc_r               # 2.5 disc *diameters*
    my = dy + 0.15 * (cy - dy)                       # fovea sits slightly inferior

    dark = cv2.GaussianBlur(bgr[:, :, 1].astype(np.float32), (0, 0), disc_r * 0.6)
    win = int(disc_r * 1.5)
    x0, x1 = int(max(0, mx - win)), int(min(bgr.shape[1], mx + win + 1))
    y0, y1 = int(max(0, my - win)), int(min(bgr.shape[0], my + win + 1))
    if x1 - x0 > 2 and y1 - y0 > 2:
        patch = dark[y0:y1, x0:x1].copy()
        pf = fov[y0:y1, x0:x1]
        patch[pf == 0] = np.inf
        if np.isfinite(patch).any():
            yy, xx = np.unravel_index(int(np.argmin(patch)), patch.shape)
            mx, my = x0 + xx, y0 + yy
    mx = float(np.clip(mx, 0, bgr.shape[1] - 1))
    my = float(np.clip(my, 0, bgr.shape[0] - 1))
    return int(mx), int(my)


# ---------------------------------------------------------------------------
# extractor
# ---------------------------------------------------------------------------
class LesionPriorExtractor:
    """Produces the six weak lesion maps and five anatomy maps at `out_size`."""

    def __init__(self, out_size: int = 56):
        self.out_size = int(out_size)

    # -- lesions ----------------------------------------------------------
    def _lesion_maps(self, bgr: np.ndarray, fov: np.ndarray,
                     disc: tuple[int, int, float],
                     macula: tuple[int, int]) -> np.ndarray:
        green = bgr[:, :, 1]
        red = bgr[:, :, 2]
        S = max(bgr.shape[:2])
        unit = max(3, int(round(S / 224.0)))          # 1 "lesion pixel" at 224 px
        odd = lambda v: int(v) | 1                     # noqa: E731

        vessels = _vessel_response(green)

        # --- dark lesions: microaneurysms vs haemorrhages differ in scale ---
        ma_raw = _tophat(green, odd(3 * unit), dark=True)
        ma = _norm_inside(np.clip(ma_raw - 0.9 * vessels, 0, None), fov)

        he_raw = _tophat(green, odd(9 * unit), dark=True)
        he = _norm_inside(np.clip(he_raw - 0.6 * vessels, 0, None), fov)
        # a haemorrhage is a *blob*: keep only what survives a closing
        he = _norm_inside(cv2.morphologyEx(
            he, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(2 * unit),) * 2)), fov)

        # --- bright lesions: hard exudates are small and sharp, soft are fuzzy ---
        ex_raw = _tophat(green, odd(5 * unit), dark=False)
        # exudates are yellow-white: bright in green AND red
        yellow = np.minimum(green.astype(np.float32), red.astype(np.float32))
        yellow = _norm_inside(cv2.GaussianBlur(yellow, (0, 0), unit), fov)
        ex_h = _norm_inside(ex_raw * (0.5 + 0.5 * yellow), fov)
        # sharpness: hard exudates have strong gradients, cotton-wool spots do not
        edge = np.abs(cv2.Laplacian(green.astype(np.float32), cv2.CV_32F,
                                    ksize=odd(unit)))
        edge = _norm_inside(edge, fov)
        ex_h = _norm_inside(ex_h * (0.4 + 0.6 * edge), fov)

        ex_s_raw = _tophat(green, odd(13 * unit), dark=False)
        ex_s = _norm_inside(ex_s_raw * (1.0 - 0.5 * edge), fov)

        # --- neovascularisation: dense fine-vessel tangle ------------------
        fine = _norm_inside(_vessel_response(green, scales=(3 * unit,)), fov)
        density = cv2.GaussianBlur(fine, (0, 0), 4.0 * unit)
        nv = _norm_inside(density * fine, fov)

        # --- macular oedema proxy: exudate burden near the fovea -----------
        mfield = _gaussian_blob(fov.shape, macula[0], macula[1],
                               sigma=2.0 * disc[2])
        me = _norm_inside(np.maximum(ex_h, ex_s) * mfield, fov)

        # the optic disc is bright and round: suppress it in the bright channels
        dfield = 1.0 - _gaussian_blob(fov.shape, disc[0], disc[1],
                                      sigma=1.2 * disc[2])
        ex_h, ex_s, nv = ex_h * dfield, ex_s * dfield, nv * dfield

        maps = np.stack([ma, he, ex_h, ex_s, nv, me]).astype(np.float32)
        # sparsify: keep the strong tail only, so the masks read as lesions and
        # not as a continuous texture score
        for i in range(maps.shape[0]):
            v = maps[i][fov > 0]
            if v.size:
                thr = float(np.percentile(v, 99.0))
                maps[i] = np.clip((maps[i] - thr) / (1.0 - thr + 1e-6), 0, 1)
        return maps

    # -- anatomy ----------------------------------------------------------
    def _anatomy_maps(self, bgr: np.ndarray, fov: np.ndarray,
                      disc: tuple[int, int, float],
                      macula: tuple[int, int]) -> np.ndarray:
        dx, dy, disc_r = disc
        shape = fov.shape
        fovf = (fov > 0).astype(np.float32)

        od = _gaussian_blob(shape, dx, dy, sigma=disc_r) * fovf
        mac = _gaussian_blob(shape, macula[0], macula[1], sigma=2.0 * disc_r) * fovf

        # arcades: vessel response above / below the disc-fovea axis
        vessels = _norm_inside(_vessel_response(bgr[:, :, 1]), fov)
        vessels = cv2.GaussianBlur(vessels, (0, 0), disc_r * 0.5)
        h, w = shape
        yy = np.mgrid[:h, :w][0].astype(np.float32)
        axis_y = dy + (macula[1] - dy) * (np.mgrid[:h, :w][1].astype(np.float32) - dx) \
            / (macula[0] - dx + 1e-6)
        sup = np.clip((axis_y - yy) / (2.0 * disc_r), 0, 1)     # above the axis
        inf = np.clip((yy - axis_y) / (2.0 * disc_r), 0, 1)     # below the axis
        sup_arcade = (vessels * sup * fovf).astype(np.float32)
        inf_arcade = (vessels * inf * fovf).astype(np.float32)

        # periphery: inside the FOV but far from disc and macula
        ys, xs = np.where(fov > 0)
        if ys.size:
            cx, cy = float(xs.mean()), float(ys.mean())
            radius = float(np.sqrt(fovf.sum() / np.pi)) + 1e-6
        else:
            cx, cy, radius = w / 2, h / 2, max(h, w) / 2
        yy2, xx2 = np.mgrid[:h, :w].astype(np.float32)
        rad = np.sqrt((xx2 - cx) ** 2 + (yy2 - cy) ** 2) / radius
        peripheral = (np.clip((rad - 0.55) / 0.45, 0, 1) * fovf).astype(np.float32)

        out = np.stack([od, mac, sup_arcade, inf_arcade, peripheral])
        return np.clip(out, 0, 1).astype(np.float32)

    # -- entry point -------------------------------------------------------
    def __call__(self, image_bgr: np.ndarray,
                 fov_mask: np.ndarray | None = None) -> LesionPriors:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 BGR image, got {image_bgr.shape}")
        bgr = np.ascontiguousarray(image_bgr)
        if fov_mask is None:
            fov = np.ones(bgr.shape[:2], np.uint8)
        else:
            fov = (np.asarray(fov_mask) > 0).astype(np.uint8)

        disc = locate_optic_disc(bgr, fov)
        macula = locate_macula(bgr, fov, disc)

        lesions = self._lesion_maps(bgr, fov, disc, macula)
        anatomy = self._anatomy_maps(bgr, fov, disc, macula)

        M = self.out_size
        lesions = np.stack([cv2.resize(m, (M, M), interpolation=cv2.INTER_AREA)
                            for m in lesions]).astype(np.float32)
        anatomy = np.stack([cv2.resize(a, (M, M), interpolation=cv2.INTER_AREA)
                            for a in anatomy]).astype(np.float32)

        return LesionPriors(masks=np.clip(lesions, 0, 1),
                            anatomy=np.clip(anatomy, 0, 1),
                            disc_center=(disc[0], disc[1]),
                            macula_center=macula)


def generate_lesion_crops(masks: np.ndarray, anatomy: np.ndarray, fov_mask: np.ndarray,
                          n_crops: int = 8, crop_frac: float = 0.3125,
                          disc_center=None, macula_center=None,
                          min_sep_frac: float = 0.6) -> np.ndarray:
    """Centres of the local lesion windows, as an (n_crops, 2) float32 array of (x, y).

    Centres are returned in `fov_mask`'s own pixel grid (the cache_size x cache_size
    retinal field), NOT normalised to 0-1. The macula and optic disc are always
    included; the rest go to non-max-suppressed peaks of the summed lesion prior, kept
    at least `min_sep_frac * window` apart. If the priors are too weak to supply enough
    peaks, the remainder is filled deterministically on a ring around the field centre,
    so the result always has `n_crops` distinct windows.

    `masks` (6,M,M) and `anatomy` (5,M,M) are the weak priors at evidence-grid
    resolution M; `crop_frac` is the window size as a fraction of the field.
    """
    S = int(fov_mask.shape[0])
    M = int(masks.shape[-1])
    scale = S / M
    win = crop_frac * S
    min_sep = max(1.0, win * min_sep_frac)
    centres: list[tuple[float, float]] = []

    def far_enough(c) -> bool:
        return all(np.hypot(c[0] - a, c[1] - b) >= min_sep * 0.5 for a, b in centres)

    def add(c) -> None:
        centres.append((float(np.clip(c[0], 0, S - 1)), float(np.clip(c[1], 0, S - 1))))

    if macula_center is not None:
        add(macula_center)
    if disc_center is not None and (not centres or far_enough(disc_center)):
        add(disc_center)

    fov_small = cv2.resize((fov_mask > 0).astype(np.uint8), (M, M),
                           interpolation=cv2.INTER_NEAREST)
    k = max(1, int(win / scale * 0.25))
    inside = cv2.erode(fov_small, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    heat = np.clip(masks, 0, 1).sum(0).astype(np.float32)
    heat = cv2.GaussianBlur(heat, (0, 0), max(1.0, M / 40.0)) * inside
    radius = max(1, int(min_sep / scale))
    while len(centres) < n_crops and float(heat.max()) > 1e-6:
        y, x = np.unravel_index(int(heat.argmax()), heat.shape)
        c = ((x + 0.5) * scale, (y + 0.5) * scale)
        if far_enough(c):
            add(c)
        cv2.circle(heat, (int(x), int(y)), radius, 0.0, -1)

    if len(centres) < n_crops:                      # deterministic ring fallback
        ys, xs = np.where(fov_mask > 0)
        cx, cy = (float(xs.mean()), float(ys.mean())) if xs.size else (S / 2, S / 2)
        rad = 0.30 * S
        for i in range(4 * n_crops):
            if len(centres) >= n_crops:
                break
            a = 2 * np.pi * i / max(n_crops, 1)
            c = (cx + rad * np.cos(a), cy + rad * np.sin(a))
            if far_enough(c):
                add(c)
        for i in range(n_crops):                    # last resort: still distinct
            if len(centres) >= n_crops:
                break
            add((cx + 3 * (i + 1), cy + 3 * (i + 1)))
    return np.asarray(centres[:n_crops], np.float32)
