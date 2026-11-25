#!/usr/bin/env python3
"""Step 26 - final evaluation and model comparison from saved predictions.

Inputs are the `predictions_<split>.npz` files written by scripts/04_evaluate.py
(`--split val`, `--split test`, and `--external-cache ...` for FDR26). Nothing
here needs a GPU or a model.

    python scripts/26_final_evaluation.py \
        --val  outputs/main/predictions_val.npz \
        --test outputs/main/predictions_test.npz \
        --external fdr26=outputs/main/predictions_fdr26.npz \
        --baseline plain_retfound=outputs/plain/predictions_test.npz \
        --baseline resnet50=outputs/r50/predictions_test.npz \
        --seed-test outputs/s1/predictions_test.npz outputs/s2/predictions_test.npz \
        --out outputs/final

Protocol (fixed before looking at test):
  * primary endpoints: QWK, and the cleared rate at the NPV target (screening)
  * every CI is a patient-level bootstrap (a patient's two eyes move together)
  * QWK cut-points and the screening threshold are fitted on VALIDATION only
  * baseline comparisons are paired (same images), Holm-corrected across baselines
  * external cohorts: discrimination metrics only unless --screening-external
    (FDR26 is near-balanced, so NPV / cleared rate there say nothing about a
    real screening population)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dr import stats as S  # noqa: E402


def load(path: Path) -> dict:
    d = S.load_predictions(path)
    d["stdr"] = d["probs"][:, 3:].sum(1)       # P(grade >= 3)
    d["ref"] = d["probs"][:, 2:].sum(1)        # P(grade >= 2)
    d["exp"] = S.expected_grade(d["probs"])
    return d


def align(a: dict, b: dict) -> tuple[dict, dict]:
    """Restrict two prediction sets to their common images, in the same order."""
    common = np.intersect1d(a["image_id"], b["image_id"])
    if len(common) != len(a["image_id"]) or len(common) != len(b["image_id"]):
        print(f"[align] using {len(common)} common images "
              f"({len(a['image_id'])} vs {len(b['image_id'])})")

    def take(d):
        order = {k: i for i, k in enumerate(d["image_id"])}
        ix = np.array([order[k] for k in common])
        return {k: (v[ix] if isinstance(v, np.ndarray) and len(v) == len(d["image_id"]) else v)
                for k, v in d.items()}
    return take(a), take(b)


def headline(d: dict, pred: np.ndarray, n_boot: int, seed: int) -> dict:
    y, g = d["y"], d["patient_id"]
    out = {"n_images": int(len(y)), "n_patients": int(len(np.unique(g)))}

    def bs(metric, *arrays):
        return S.cluster_bootstrap(metric, g, *arrays, n_boot=n_boot, seed=seed)

    out["qwk"] = bs(S.qwk, y, pred)
    out["macro_f1"] = bs(S.macro_f1, y, pred)
    out["accuracy"] = bs(S.accuracy, y, pred)
    out["min_grade_recall"] = bs(S.min_recall, y, pred)
    out["recall_spread"] = bs(S.recall_spread, y, pred)
    out["recall_by_grade"] = {
        str(k): bs(lambda yy, pp, w, k=k: S.grade_recall(yy, pp, k, w), y, pred)
        for k in range(5)}
    out["ref_auc"] = bs(S.auc, (y >= 2).astype(int), d["ref"])
    out["stdr_auc"] = bs(S.auc, (y >= 3).astype(int), d["stdr"])
    return out


def screening(val: dict, d: dict, targets: list[float]) -> dict:
    """Thresholds on VAL (STDR = grade>=3), applied unchanged to `d`."""
    yv, yd = (val["y"] >= 3).astype(int), (d["y"] >= 3).astype(int)
    res = {}
    for t in targets:
        thr = S.pick_threshold_for_npv(yv, val["stdr"], t)
        res[f"npv_target_{t}"] = {
            "val": S.screening_at(yv, val["stdr"], thr),
            "applied": S.screening_at(yd, d["stdr"], thr)}
    return res


def compare(primary: dict, other: dict, p_pred: np.ndarray, o_pred: np.ndarray,
            n_boot: int, seed: int) -> dict:
    y, g = primary["y"], primary["patient_id"]
    out = {
        "qwk": S.paired_cluster_bootstrap_delta(S.qwk, g, y, p_pred, o_pred, n_boot=n_boot, seed=seed),
        "macro_f1": S.paired_cluster_bootstrap_delta(S.macro_f1, g, y, p_pred, o_pred, n_boot=n_boot, seed=seed),
        "ref_auc": S.delong_test((y >= 2).astype(int), primary["ref"], other["ref"]),
        "stdr_auc": S.delong_test((y >= 3).astype(int), primary["stdr"], other["stdr"]),
        "accuracy": S.mcnemar_test(y, p_pred, o_pred),
    }
    return out


def fmt(m: dict) -> str:
    return f"{m['point']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=Path, default=None,
                    help="validation predictions: needed for tuned cut-points, the extra "
                         "decoders and screening. Omit for a matched-budget comparison "
                         "(scripts/07 produces test predictions only); only the saved "
                         "argmax decoder is then used.")
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--external", action="append", default=[], metavar="NAME=NPZ")
    ap.add_argument("--baseline", action="append", default=[], metavar="NAME=NPZ")
    ap.add_argument("--seed-test", type=Path, nargs="*", default=[],
                    help="test predictions from other seeds of the SAME model")
    ap.add_argument("--npv-targets", type=float, nargs="*", default=[0.985, 0.99])
    ap.add_argument("--screening-external", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=Path, default=Path("outputs/final"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    val = load(args.val) if args.val else None
    test = load(args.test)

    if val is not None:
        # --- QWK cut-points fitted on validation only ----------------------------
        cuts = S.optimise_cutpoints(val["exp"], val["y"])
        print(f"[cuts] validation-fitted QWK cut-points: {np.round(cuts, 3).tolist()}")

        # --- extra decoders, every parameter fitted on VALIDATION only -----------
        # priors come from validation labels (a proxy: the true training priors are
        # not stored in the prediction files)
        priors = np.bincount(val["y"], minlength=5) / len(val["y"])
        tau = S.fit_logit_adjust(val["probs"], val["y"], priors)
        v_partner = S.eye_partner_index(val["image_id"])
        has_pairs = (v_partner >= 0).mean() > 0.5
        if has_pairs:
            alpha, eye_cuts = S.fit_eye_fusion(val["exp"], val["y"], v_partner)
        else:
            alpha, eye_cuts = 0.0, cuts
        print(f"[decoders] logit-adjust tau={tau:.2f}; eye-fusion "
              f"{'alpha=%.2f' % alpha if has_pairs else 'unavailable (no left/right pairs in validation ids)'}")

        def decoders(d):
            out = {"as_saved": d["pred"],
                   "tuned_cuts": S.apply_cutpoints(d["exp"], cuts),
                   "logit_adjust": S.logit_adjust(d["probs"], priors, tau)}
            partner = S.eye_partner_index(d["image_id"])
            if has_pairs and (partner >= 0).mean() > 0.5:
                out["eye_fusion"] = S.apply_cutpoints(S.eye_fuse(d["exp"], partner, alpha), eye_cuts)
            return out
    else:
        cuts, tau, alpha, has_pairs = None, 0.0, 0.0, False
        print("[val] no validation predictions: only the saved argmax decoder is used")

        def decoders(d):
            return {"as_saved": d["pred"]}

    report: dict = {"protocol": {
        "primary_endpoints": ["qwk", f"cleared_rate@npv>={args.npv_targets[0]}"],
        "ci": "patient-level bootstrap, 95%", "n_boot": args.n_boot,
        "cutpoints_fitted_on": "validation" if val is not None else None,
        "cutpoints": cuts.tolist() if cuts is not None else None,
        "primary_decoder": "as_saved (declared before test; the others are reported "
                           "as secondary analyses)",
        "logit_adjust_tau": tau, "eye_fusion_alpha": alpha if has_pairs else None}}

    report["test"] = {k: headline(test, p, args.n_boot, args.seed)
                      for k, p in decoders(test).items()}
    report["test_screening"] = (screening(val, test, args.npv_targets) if val is not None
                                else {})

    report["external"] = {}
    for item in args.external:
        name, path = item.split("=", 1)
        d = load(Path(path))
        entry = {k: headline(d, p, args.n_boot, args.seed) for k, p in decoders(d).items()}
        if args.screening_external:
            entry["screening"] = (screening(val, d, args.npv_targets) if val is not None
                                  else "needs --val")
        else:
            entry["screening"] = ("not reported: cohort prevalence is not "
                                  "representative of a screening population")
        report["external"][name] = entry

    # --- baselines: paired, Holm-corrected -----------------------------------
    report["baselines"] = {}
    comps, ptable = {}, {}
    for item in args.baseline:
        name, path = item.split("=", 1)
        a, b = align(test, load(Path(path)))
        # same decoder for both models, for a fair comparison
        if cuts is not None:
            a_pred, b_pred = S.apply_cutpoints(a["exp"], cuts), S.apply_cutpoints(b["exp"], cuts)
        else:
            a_pred, b_pred = a["pred"], b["pred"]
        comps[name] = compare(a, b, a_pred, b_pred, args.n_boot, args.seed)
        for ep in ("qwk", "macro_f1", "ref_auc", "stdr_auc", "accuracy"):
            ptable[(ep, name)] = comps[name][ep]["p"]
    for ep in ("qwk", "macro_f1", "ref_auc", "stdr_auc", "accuracy"):
        adj = S.holm({n: ptable[(ep, n)] for n in comps})
        for n in comps:
            comps[n][ep]["p_holm"] = adj[n]
    report["baselines"] = comps

    # --- seeds: mean +/- SD ---------------------------------------------------
    if args.seed_test:
        runs = [load(p) for p in [args.test, *args.seed_test]]
        vals = {"qwk": [], "macro_f1": [], "ref_auc": [], "stdr_auc": []}
        for r in runs:
            p = S.apply_cutpoints(r["exp"], cuts) if cuts is not None else r["pred"]
            vals["qwk"].append(S.qwk(r["y"], p))
            vals["macro_f1"].append(S.macro_f1(r["y"], p))
            vals["ref_auc"].append(S.auc((r["y"] >= 2).astype(int), r["ref"]))
            vals["stdr_auc"].append(S.auc((r["y"] >= 3).astype(int), r["stdr"]))
        report["seeds"] = {"n_runs": len(runs),
                           **{k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                                  "values": [float(x) for x in v]} for k, v in vals.items()}}

    (args.out / "final_evaluation.json").write_text(json.dumps(report, indent=2, default=float))

    # --- markdown ------------------------------------------------------------
    L = ["# Final evaluation", "",
         f"Patient-level bootstrap 95% CI, {args.n_boot} resamples. QWK cut-points "
         f"fitted on validation: `{np.round(cuts, 3).tolist() if cuts is not None else 'n/a (no validation set given)'}`.", "",
         "| cohort | decoder | n | QWK | macro-F1 | accuracy | min recall | refAUC | STDR AUC |",
         "|---|---|---|---|---|---|---|---|---|"]
    rows = [("EyePACS test", report["test"])] + [(f"{k} (external)", v) for k, v in report["external"].items()]
    for label, block in rows:
        for dec, h in block.items():
            if not isinstance(h, dict) or "qwk" not in h:
                continue        # e.g. the external cohort's 'screening' note
            L.append(f"| {label} | {dec} | {h['n_images']} | {fmt(h['qwk'])} | {fmt(h['macro_f1'])} | "
                     f"{fmt(h['accuracy'])} | {fmt(h['min_grade_recall'])} | {fmt(h['ref_auc'])} | "
                     f"{fmt(h['stdr_auc'])} |")
    if report["test_screening"]:
        L += ["", "## Screening (STDR, grade >= 3): threshold fitted on validation, applied to test", "",
              "| NPV target | cleared | NPV [exact 95% CI] | sensitivity | specificity | PPV | missed | prevalence |",
              "|---|---|---|---|---|---|---|---|"]
    for k, v in report["test_screening"].items():
        a = v["applied"]
        L.append(f"| {k.split('_')[-1]} | {a['cleared_rate']:.3f} | {a['npv']:.4f} "
                 f"[{a['npv_ci'][0]:.4f}, {a['npv_ci'][1]:.4f}] | {a['sensitivity']:.3f} | "
                 f"{a['specificity']:.3f} | {a['ppv']:.3f} | {a['missed']} | {a['prevalence']:.3f} |")
    if comps:
        L += ["", "## Paired comparison vs baselines (primary minus baseline, Holm-adjusted p)", "",
              "| baseline | dQWK [CI] | p_holm | dMacro-F1 [CI] | p_holm | refAUC p_holm | STDR AUC p_holm | McNemar p_holm |",
              "|---|---|---|---|---|---|---|---|"]
        for n, c in comps.items():
            L.append(f"| {n} | {c['qwk']['delta']:+.3f} [{c['qwk']['lo']:+.3f}, {c['qwk']['hi']:+.3f}] | "
                     f"{c['qwk']['p_holm']:.3g} | {c['macro_f1']['delta']:+.3f} "
                     f"[{c['macro_f1']['lo']:+.3f}, {c['macro_f1']['hi']:+.3f}] | {c['macro_f1']['p_holm']:.3g} | "
                     f"{c['ref_auc']['p_holm']:.3g} | {c['stdr_auc']['p_holm']:.3g} | {c['accuracy']['p_holm']:.3g} |")
    if "seeds" in report:
        s = report["seeds"]
        L += ["", f"## Seed stability ({s['n_runs']} runs, same split)", ""]
        L += [f"- {k}: {s[k]['mean']:.3f} +/- {s[k]['sd']:.3f}" for k in ("qwk", "macro_f1", "ref_auc", "stdr_auc")]
    (args.out / "final_evaluation.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {args.out / 'final_evaluation.json'}")


if __name__ == "__main__":
    main()
