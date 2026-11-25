"""Statistical helpers for the final evaluation and model comparison.

Everything here works on saved per-image predictions, so it needs no GPU and
no model:

  * patient-level (cluster) bootstrap CIs - EyePACS contributes two eyes per
    patient, so resampling images would give intervals that are too narrow
  * paired cluster bootstrap for metric differences between two models
  * DeLong test for paired AUCs, McNemar for paired accuracy
  * Holm correction across many baselines
  * Clopper-Pearson interval for NPV / sensitivity style proportions
  * QWK-optimal cut-points for the expected-grade decoder, fitted on VALIDATION
    data and then applied unchanged to test
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats as sps
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

N_GRADES = 5


# ---------------------------------------------------------------------------
# weighted metric functions: f(y, pred_or_score, w) -> float
# ---------------------------------------------------------------------------
def qwk(y, pred, w=None) -> float:
    return float(cohen_kappa_score(y, pred, weights="quadratic", sample_weight=w,
                                   labels=list(range(N_GRADES))))


def macro_f1(y, pred, w=None) -> float:
    return float(f1_score(y, pred, average="macro", sample_weight=w,
                          labels=list(range(N_GRADES)), zero_division=0))


def accuracy(y, pred, w=None) -> float:
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    return float((w * (y == pred)).sum() / max(w.sum(), 1e-12))


def grade_recall(y, pred, grade: int, w=None) -> float:
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    m = y == grade
    den = (w * m).sum()
    return float((w * (m & (pred == grade))).sum() / den) if den > 0 else float("nan")


def min_recall(y, pred, w=None, grades=range(N_GRADES)) -> float:
    r = [grade_recall(y, pred, g, w) for g in grades]
    r = [v for v in r if np.isfinite(v)]
    return float(min(r)) if r else float("nan")


def recall_spread(y, pred, w=None) -> float:
    r = [grade_recall(y, pred, g, w) for g in range(N_GRADES)]
    r = [v for v in r if np.isfinite(v)]
    return float(max(r) - min(r)) if r else float("nan")


def auc(y_bin, score, w=None) -> float:
    if len(np.unique(y_bin)) < 2:
        return float("nan")
    return float(roc_auc_score(y_bin, score, sample_weight=w))


# ---------------------------------------------------------------------------
# cluster bootstrap
# ---------------------------------------------------------------------------
def _cluster_weights(groups: np.ndarray, rng: np.random.Generator,
                     n_boot: int):
    """Yield per-image integer weights for each patient-level resample."""
    _, inv = np.unique(groups, return_inverse=True)
    n_groups = int(inv.max()) + 1
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n_groups, n_groups),
                             minlength=n_groups)
        yield counts[inv].astype(np.float64)


def cluster_bootstrap(metric: Callable, groups, *arrays, n_boot: int = 1000,
                      seed: int = 1337, alpha: float = 0.05) -> dict:
    """Patient-level bootstrap of `metric(*arrays, w)`; returns point and CI."""
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    point = metric(*arrays, None)
    vals = np.array([metric(*arrays, w) for w in _cluster_weights(groups, rng, n_boot)])
    vals = vals[np.isfinite(vals)]
    if len(vals) < 10:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n_boot": len(vals)}
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "n_boot": int(len(vals))}


def paired_cluster_bootstrap_delta(metric: Callable, groups, y, a, b, *,
                                   n_boot: int = 1000, seed: int = 7,
                                   alpha: float = 0.05) -> dict:
    """metric(y, a) - metric(y, b), resampling patients; two-sided bootstrap p."""
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    delta = metric(y, a, None) - metric(y, b, None)
    ds = []
    for w in _cluster_weights(groups, rng, n_boot):
        ds.append(metric(y, a, w) - metric(y, b, w))
    ds = np.array([d for d in ds if np.isfinite(d)])
    lo, hi = np.quantile(ds, [alpha / 2, 1 - alpha / 2])
    # fraction of resamples on the opposite side of zero, doubled
    p = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
    return {"delta": float(delta), "lo": float(lo), "hi": float(hi),
            "p": float(min(max(p, 1.0 / (len(ds) + 1)), 1.0))}


# ---------------------------------------------------------------------------
# paired tests
# ---------------------------------------------------------------------------
def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def _delong_structural(y_bin: np.ndarray, scores: np.ndarray):
    """Fast DeLong (Sun & Xu 2014). scores: (k, n). Returns aucs, covariance."""
    pos = scores[:, y_bin == 1]
    neg = scores[:, y_bin == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = scores.shape[0]
    tx, ty, tz = (np.empty((k, m)), np.empty((k, n)), np.empty((k, m + n)))
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(np.concatenate([pos[r], neg[r]]))
    aucs = tz[:, :m].sum(1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def delong_test(y_bin, score_a, score_b) -> dict:
    """Two-sided DeLong test for two correlated AUCs on the same cases."""
    y_bin = np.asarray(y_bin).astype(int)
    if len(np.unique(y_bin)) < 2:
        return {"auc_a": float("nan"), "auc_b": float("nan"), "z": float("nan"), "p": float("nan")}
    aucs, cov = _delong_structural(y_bin, np.vstack([score_a, score_b]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 1e-15:        # identical (or perfectly tied) scores: no detectable difference
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "z": 0.0, "p": 1.0}
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "z": float(z),
            "p": float(2 * sps.norm.sf(abs(z)))}


def mcnemar_test(y, pred_a, pred_b) -> dict:
    """Exact McNemar on paired correctness (a right/b wrong vs a wrong/b right)."""
    ca, cb = (np.asarray(pred_a) == y), (np.asarray(pred_b) == y)
    b01 = int((ca & ~cb).sum())
    b10 = int((~ca & cb).sum())
    n = b01 + b10
    p = 1.0 if n == 0 else float(sps.binomtest(min(b01, b10), n, 0.5).pvalue)
    return {"a_only_correct": b01, "b_only_correct": b10, "p": p}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, keyed like the input."""
    items = sorted(((k, p) for k, p in pvals.items() if np.isfinite(p)),
                   key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    for k, p in pvals.items():
        adj.setdefault(k, float("nan"))
    return adj


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial CI for k successes of n (used for NPV, sensitivity)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(sps.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(sps.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


# ---------------------------------------------------------------------------
# QWK-optimal cut-points (fit on validation, apply to test)
# ---------------------------------------------------------------------------
DEFAULT_CUTS = np.array([0.5, 1.5, 2.5, 3.5])


def expected_grade(probs: np.ndarray) -> np.ndarray:
    return np.asarray(probs, float) @ np.arange(probs.shape[1])


def apply_cutpoints(score: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    return np.digitize(score, np.sort(cuts)).astype(int)


def optimise_cutpoints(score: np.ndarray, y: np.ndarray, init=DEFAULT_CUTS,
                       n_rounds: int = 3, n_grid: int = 61) -> np.ndarray:
    """Coordinate ascent on the 4 cut-points to maximise QWK of `score` vs `y`.

    Fit this on VALIDATION predictions only. Fitting on the test split would
    inflate the reported QWK.
    """
    cuts = np.array(init, float)
    best = qwk(y, apply_cutpoints(score, cuts))
    for _ in range(n_rounds):
        for i in range(len(cuts)):
            lo = cuts[i - 1] + 1e-3 if i > 0 else float(score.min()) - 1e-3
            hi = cuts[i + 1] - 1e-3 if i < len(cuts) - 1 else float(score.max()) + 1e-3
            for c in np.linspace(lo, hi, n_grid):
                trial = cuts.copy()
                trial[i] = c
                s = qwk(y, apply_cutpoints(score, trial))
                if s > best:
                    best, cuts = s, trial
    return cuts


# ---------------------------------------------------------------------------
# screening operating point: fit on validation, report on test
# ---------------------------------------------------------------------------
def pick_threshold_for_npv(y_bin: np.ndarray, score: np.ndarray,
                           npv_target: float) -> float:
    """Highest threshold t (clear if score < t) whose validation NPV >= target.

    A larger t clears more patients, so we search from the top down for the
    largest threshold that still meets the NPV constraint.
    """
    order = np.unique(score)
    best = float(order[0])
    for t in order:
        cleared = score < t
        n_clear = int(cleared.sum())
        if n_clear == 0:
            continue
        npv = 1.0 - float(y_bin[cleared].sum()) / n_clear
        if npv >= npv_target:
            best = float(t)
    return best


def screening_at(y_bin: np.ndarray, score: np.ndarray, thr: float,
                 alpha: float = 0.05) -> dict:
    """Cleared fraction, NPV (+exact CI), sensitivity, specificity, PPV."""
    clear = score < thr
    refer = ~clear
    n = len(y_bin)
    n_clear = int(clear.sum())
    fn = int(y_bin[clear].sum())
    tn = n_clear - fn
    tp = int(y_bin[refer].sum())
    fp = int(refer.sum()) - tp
    npv_ci = clopper_pearson(tn, n_clear, alpha)
    return {
        "threshold": float(thr), "n": n,
        "prevalence": float(y_bin.mean()),
        "cleared_rate": n_clear / n,
        "npv": tn / n_clear if n_clear else float("nan"),
        "npv_ci": [npv_ci[0], npv_ci[1]],
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "ppv": tp / max(tp + fp, 1),
        "missed": fn,
    }


# ---------------------------------------------------------------------------
# prediction-level decoders, all fitted on VALIDATION and applied unchanged
# ---------------------------------------------------------------------------
def composite_score(y: np.ndarray, pred: np.ndarray,
                    minority=(1, 2, 3)) -> float:
    """0.5*QWK + 0.3*macro-F1 + 0.2*mean minority recall (the training selection score)."""
    rec = [grade_recall(y, pred, g) for g in minority]
    rec = [r for r in rec if np.isfinite(r)]
    return 0.5 * qwk(y, pred) + 0.3 * macro_f1(y, pred) + 0.2 * (float(np.mean(rec)) if rec else 0.0)


def logit_adjust(probs: np.ndarray, priors: np.ndarray, tau: float) -> np.ndarray:
    """Post-hoc logit adjustment: argmax_k  log p_k - tau * log prior_k.

    tau = 0 is the plain argmax; larger tau trades majority-class accuracy for
    minority-class recall without retraining the head.
    """
    adj = np.log(np.clip(probs, 1e-9, 1.0)) - tau * np.log(np.clip(priors, 1e-9, 1.0))
    return adj.argmax(1).astype(int)


def fit_logit_adjust(probs: np.ndarray, y: np.ndarray, priors: np.ndarray,
                     taus=np.linspace(0.0, 2.0, 21)) -> float:
    best_t, best_s = 0.0, -np.inf
    for t in taus:
        s = composite_score(y, logit_adjust(probs, priors, float(t)))
        if s > best_s:
            best_t, best_s = float(t), s
    return best_t


def eye_partner_index(image_ids: np.ndarray) -> np.ndarray:
    """Index of the other eye of the same patient (EyePACS 'NN_left'/'NN_right'), else -1."""
    ids = [str(i) for i in image_ids]
    pos = {k: i for i, k in enumerate(ids)}
    out = np.full(len(ids), -1, dtype=np.int64)
    for i, k in enumerate(ids):
        for a, b in (("_left", "_right"), ("_right", "_left")):
            if k.endswith(a):
                out[i] = pos.get(k[: -len(a)] + b, -1)
                break
    return out


def eye_fuse(score: np.ndarray, partner: np.ndarray, alpha: float) -> np.ndarray:
    """(1-alpha)*own + alpha*partner; eyes with no partner keep their own score."""
    other = np.where(partner >= 0, score[np.clip(partner, 0, None)], score)
    return (1 - alpha) * score + alpha * other


def fit_eye_fusion(score: np.ndarray, y: np.ndarray, partner: np.ndarray,
                   alphas=np.linspace(0.0, 0.5, 11)) -> tuple[float, np.ndarray]:
    """Pick alpha (and fresh QWK cut-points) on validation. Returns (alpha, cuts)."""
    best = (0.0, DEFAULT_CUTS.copy(), -np.inf)
    for a in alphas:
        f = eye_fuse(score, partner, float(a))
        cuts = optimise_cutpoints(f, y, n_rounds=2, n_grid=31)
        q = qwk(y, apply_cutpoints(f, cuts))
        if q > best[2]:
            best = (float(a), cuts, q)
    return best[0], best[1]


def load_predictions(path) -> dict:
    """Load a predictions_*.npz. IDs are stored as fixed-width strings; files written by an
    older version stored them as object arrays (newer pandas), so fall back to pickle for those
    (they are this project's own outputs) and convert to plain strings."""
    try:
        z = np.load(path, allow_pickle=False)
        d = {k: z[k] for k in z.files}
    except ValueError:
        z = np.load(path, allow_pickle=True)
        d = {k: z[k] for k in z.files}
    for k in ("patient_id", "image_id"):
        if k in d:
            d[k] = np.asarray(d[k]).astype(str)
    return d
