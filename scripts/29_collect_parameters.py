#!/usr/bin/env python3
"""Step 29 - one master table of every parameter worth reporting, with provenance.

    python scripts/29_collect_parameters.py --run outputs/study_seed1337 \
        [--final outputs/study/final] [--matched outputs/study/final_matched] \
        [--cache data/cache] --out outputs/study/parameters

Writes `parameters_master.csv` and `.json` with columns
    section, parameter, value, source, status, note
where `status` is `computed` (read from a result file), `derived` (calculated here from
one) or `unavailable` (with the reason in `note`). Nothing is guessed: a parameter whose
source file is missing is listed as unavailable, never filled in.

Covers the requested list - dataset counts and imbalance, sampling, backbone / LoRA /
trainable parameters, schedule and learning rates, timing, every training and validation
metric (last epoch, best epoch, per stage), per-grade recall spread - plus environment,
resolution policy, loss-weight audit, preprocessing gate statistics, test-set results with
confidence intervals, external-cohort results and comparison outcomes.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GRADES = [f"val_recall_grade_{k}" for k in range(5)]
ROWS: list[dict] = []


def add(section, parameter, value, source, status="computed", note=""):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    ROWS.append({"section": section, "parameter": parameter, "value": value,
                 "source": source, "status": status, "note": note})


def missing(section, parameter, source, note):
    add(section, parameter, "", source, "unavailable", note)


def load_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:                                              # noqa: BLE001
        return None


def environment_rows(run: Path):
    add("environment", "python", platform.python_version(), "runtime")
    add("environment", "platform", platform.platform(), "runtime")
    for mod in ("torch", "torchvision", "timm", "numpy", "pandas", "sklearn", "cv2"):
        try:
            m = __import__(mod)
            add("environment", f"{mod}_version", getattr(m, "__version__", "?"), "runtime")
        except Exception:                                          # noqa: BLE001
            missing("environment", f"{mod}_version", "runtime", "not importable")
    try:
        import torch
        if torch.cuda.is_available():
            pr = torch.cuda.get_device_properties(0)
            add("environment", "gpu_name", pr.name, "runtime")
            add("environment", "gpu_memory_gb", round(pr.total_memory / 1e9, 1), "runtime")
            add("environment", "gpu_count", torch.cuda.device_count(), "runtime")
        else:
            add("environment", "gpu_name", "none (CPU)", "runtime")
    except Exception:                                              # noqa: BLE001
        pass
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parents[1], timeout=10).stdout.strip()
        add("environment", "git_commit", sha or "unknown", "git")
    except Exception:                                              # noqa: BLE001
        missing("environment", "git_commit", "git", "git not available")
    meta = load_json(run / "run_meta.json")
    if meta:
        for k, v in meta.items():
            add("run", k, json.dumps(v) if isinstance(v, (dict, list)) else v, "run_meta.json")


def dataset_rows(run: Path, cache: Path | None):
    ds = load_json(run / "dataset_summary.json")
    if ds is None and cache and (cache / "meta.csv").exists():
        m = pd.read_csv(cache / "meta.csv")
        ds = {"total_images": int(len(m)), "total_patients": int(m["patient_id"].nunique()),
              "per_split": {}}
        for sp, g in m.groupby("split"):
            c = g["grade"].value_counts().reindex(range(5), fill_value=0).tolist()
            ds["per_split"][sp] = {"images": int(len(g)), "patients": int(g["patient_id"].nunique()),
                                   "grade_counts": [int(x) for x in c]}
        src, st = "cache/meta.csv", "derived"
    else:
        src, st = "dataset_summary.json", "computed"
    if ds is None:
        for p in ("Total Images", "Total Patients", "Training Images", "Validation Images",
                  "Test Images", "Grade counts", "Class Imbalance Ratio"):
            missing("dataset", p, "dataset_summary.json / cache meta.csv", "no summary or cache found")
        return
    add("dataset", "Total Images", ds["total_images"], src, st)
    add("dataset", "Total Patients", ds["total_patients"], src, st)
    for sp, label in (("train", "Training"), ("val", "Validation"), ("test", "Test")):
        d = ds["per_split"].get(sp)
        if d:
            add("dataset", f"{label} Images", d["images"], src, st)
            add("dataset", f"{label} Patients", d["patients"], src, st)
            for g, n in enumerate(d["grade_counts"]):
                add("dataset", f"{label} Grade {g} Image Count", n, src, st)
            tot = max(sum(d["grade_counts"]), 1)
            add("dataset", f"{label} Grade-wise Class Distribution",
                json.dumps([round(n / tot, 4) for n in d["grade_counts"]]), src, "derived")
    tr = ds["per_split"].get("train", {}).get("grade_counts")
    if tr and min(tr) > 0:
        add("dataset", "Class Imbalance Ratio (max:min)", round(max(tr) / min(tr), 2), src, "derived")
        add("dataset", "Class Imbalance Ratio (grade0:grade4)", round(tr[0] / tr[4], 2), src, "derived")
    sm = ds.get("sampling", {})
    if sm:
        add("sampling", "Sampling exponent (n^power)", sm.get("power"), src)
        add("sampling", "Class-Weight Exponent", sm.get("class_weight_power"), src)
    if tr and min(tr) > 0 and sm:
        add("sampling", "Grade 4-to-Grade 0 Sampling Weight",
            round((tr[4] / tr[0]) ** sm["power"], 4), src, "derived",
            "ratio of per-image sampling probability, ignoring hard-example mining")
        add("sampling", "Sampling Ratio (grade 4 : grade 0, expected per epoch)",
            round(tr[4] * (tr[4] ** sm["power"]) / (tr[0] * (tr[0] ** sm["power"])), 4),
            src, "derived", "expected share of draws, class-level")
    elif sm:
        missing("sampling", "Grade 4-to-Grade 0 Sampling Weight", src, "a grade has zero training images")
    if "grade4_to_grade0_sampling_weight" in ds:
        add("sampling", "Grade 4-to-Grade 0 Sampling Weight (logged by 03)",
            ds["grade4_to_grade0_sampling_weight"], src)


def model_rows(run: Path):
    ps = load_json(run / "param_summary.json")
    gl = load_json(run / "gla_lora.json")
    cfg = load_json(run / "config.json") or {}
    if ps:
        add("model", "RETFound Backbone Parameters", ps.get("backbone_params"), "param_summary.json")
        add("model", "Model Total Parameters", ps.get("model_total_params"), "param_summary.json")
        lr = ps.get("retfound_load_report", {})
        if lr:
            add("model", "RETFound Tensor Loading Coverage", lr.get("backbone_coverage"), "param_summary.json")
            add("model", "RETFound matched tensors", lr.get("matched_tensors"), "param_summary.json")
            add("model", "RETFound loaded", lr.get("loaded"), "param_summary.json")
        trainable = []
        for name, d in ps.items():
            if not isinstance(d, dict) or "trainable_params" not in d:
                continue
            add("model", f"[{name}] Trainable Parameters", d["trainable_params"], "param_summary.json")
            add("model", f"[{name}] Trainable Parameter Percentage", d["trainable_pct"], "param_summary.json")
            add("model", f"[{name}] Unfrozen Transformer Blocks",
                f"{d['unfrozen_blocks']}/{d['total_blocks']}", "param_summary.json")
            add("model", f"[{name}] LoRA trainable", d["lora_trainable"], "param_summary.json")
            for k, v in d["lr"].items():
                add("schedule", f"[{name}] {k.capitalize()} Learning Rate", v, "param_summary.json")
            add("schedule", f"[{name}] Stage-wise Epoch Count (planned)", d["epochs"], "param_summary.json")
            trainable.append(d["trainable_params"])
        if trainable:
            add("model", "Trainable Parameters (peak over stages)", max(trainable), "param_summary.json", "derived")
    else:
        for p in ("RETFound Backbone Parameters", "RETFound Tensor Loading Coverage",
                  "Trainable Parameters", "Trainable Parameter Percentage", "Unfrozen Transformer Blocks",
                  "Learning rates per stage"):
            missing("model", p, "param_summary.json", "written by 03_train.py of this version; "
                    "not present for older runs")
    if gl:
        ranks = gl["ranks"]
        add("model", "GLA-LoRA Parameters", gl["lora_params"], "gla_lora.json")
        if ps and ps.get("model_total_params"):
            add("model", "GLA-LoRA Parameter Percentage",
                round(100 * gl["lora_params"] / ps["model_total_params"], 4), "gla_lora.json", "derived",
                "of all model parameters")
        add("model", "LoRA Rank Allocation", json.dumps(ranks), "gla_lora.json")
        add("model", "LoRA mean rank", round(float(np.mean(ranks)), 3), "gla_lora.json", "derived")
        add("model", "Blocks pinned at max rank", gl.get("n_at_max_rank"), "gla_lora.json")
    else:
        missing("model", "GLA-LoRA Parameters / Rank Allocation", "gla_lora.json", "file missing")
    pc = cfg.get("preproc", {})
    if pc:
        for k in ("cache_size", "global_size", "crop_size", "crop_input", "n_crops"):
            add("resolution", k, pc.get(k), "config.json")
    for k, v in cfg.get("train", {}).items():
        if k in ("batch_size", "grad_accum", "seed", "ema_decay", "weight_decay", "patience", "amp"):
            add("schedule", f"train.{k}", v, "config.json")


def history_rows(run: Path):
    hp = run / "history.csv"
    if not hp.exists():
        missing("training", "history", "history.csv", "file missing")
        return None
    h = pd.read_csv(hp)
    h["min_grade_recall"] = h[GRADES].min(axis=1)
    h["recall_spread"] = h[GRADES].max(axis=1) - h[GRADES].min(axis=1)
    cfg = load_json(run / "config.json") or {}
    planned = sum(s["epochs"] for s in cfg.get("train", {}).get("stages", [])) or None
    add("training", "Completed Epochs", len(h), "history.csv")
    add("training", "Planned Epochs", planned if planned else "", "config.json",
        "computed" if planned else "unavailable")
    add("training", "Training Stage (last)", h["stage"].iloc[-1], "history.csv")
    add("training", "Run complete", bool(planned and len(h) >= planned), "history.csv", "derived",
        "False means 'best' values are NOT final (early stop also leaves this False)")
    for st, g in h.groupby("stage", sort=False):
        add("training", f"Stage-wise Epoch Count (done) [{st}]", len(g), "history.csv")
        add("training", f"Average QWK [{st}]", round(g["val_quadratic_weighted_kappa"].mean(), 5), "history.csv", "derived")
        add("training", f"Best QWK [{st}]", round(g["val_quadratic_weighted_kappa"].max(), 5), "history.csv", "derived")
        add("training", f"Average Selection Score [{st}]", round(g["selection_score"].mean(), 5), "history.csv", "derived")
        add("training", f"Best Selection Score [{st}]", round(g["selection_score"].max(), 5), "history.csv", "derived")
        add("training", f"Average Referable DR AUC [{st}]", round(g["val_auc_referable_dr"].mean(), 5), "history.csv", "derived")
        add("training", f"Average Minimum Recall [{st}]", round(g["min_grade_recall"].mean(), 5), "history.csv", "derived")
    et = h["epoch_sec"]
    add("timing", "Epoch Wall-clock Time (last, s)", round(et.iloc[-1], 1), "history.csv")
    add("timing", "Average Training Time per Epoch (min)", round(et.mean() / 60, 3), "history.csv", "derived")
    add("timing", "Total training time (h)", round(et.sum() / 3600, 3), "history.csv", "derived")
    cum = et.cumsum() / 3600
    if len(h) > 2 and h["selection_score"].std() > 0:
        add("timing", "Score versus Training Time (Pearson r)",
            round(float(np.corrcoef(cum, h["selection_score"])[0, 1]), 4), "history.csv", "derived",
            "figure: paper_tables/score_vs_time.png and the PDF report")
    last, bi = h.iloc[-1], int(h["selection_score"].idxmax())
    best = h.iloc[bi]
    cols = {"train_loss": "Training Loss", "val_accuracy": "Accuracy", "val_f1_macro": "Macro F1",
            "val_quadratic_weighted_kappa": "Quadratic Weighted Kappa (QWK)",
            "min_grade_recall": "Minimum Grade Recall", "val_auc_referable_dr": "Referable DR AUC (refAUC)",
            "val_auc_sight_threatening": "Sight-Threatening DR AUC (STDR AUC)",
            "val_screen_cleared": "Screening Cleared Rate", "val_screen_npv": "Screening NPV",
            "val_screen_threshold": "Screening Threshold", "selection_score": "Composite Selection Score",
            "mine_p90_hardness": "Miner 90th-Percentile Hardness", "mine_boundary_rate": "Boundary Error Rate",
            "recall_spread": "Per-grade Recall Spread"}
    for k, lab in cols.items():
        if k in h.columns:
            add("metrics_last_epoch", lab, round(float(last[k]), 6), "history.csv")
            add("metrics_best_epoch", lab, round(float(best[k]), 6), "history.csv", "computed",
                f"epoch {int(best['epoch'])} (best composite selection score)")
        else:
            missing("metrics_last_epoch", lab, "history.csv", "column not in this run's history")
    for g in range(5):
        add("metrics_last_epoch", f"Grade {g} Recall", round(float(last[f"val_recall_grade_{g}"]), 6), "history.csv")
    add("metrics_best_over_run", "Best QWK", round(h["val_quadratic_weighted_kappa"].max(), 6), "history.csv", "derived")
    add("metrics_best_over_run", "Best Selection Score", round(h["selection_score"].max(), 6), "history.csv", "derived")
    add("metrics_best_over_run", "Best Minimum-Class Recall", round(h["min_grade_recall"].max(), 6), "history.csv", "derived")
    add("metrics_best_over_run", "Best Referable DR AUC", round(h["val_auc_referable_dr"].max(), 6), "history.csv", "derived")
    add("metrics_best_over_run", "Best STDR AUC", round(h["val_auc_sight_threatening"].max(), 6), "history.csv", "derived")
    # loss audit: dead or stuck terms, and learned loss weights that never moved
    for c in [c for c in h.columns if c.startswith("train_") and c not in ("train_loss",)]:
        v = h[c].astype(float)
        tail = v.tail(max(3, len(v) // 4))
        flag = ("near-zero" if abs(v.mean()) < 1e-3 else
                "flat" if len(v) >= 6 and tail.std() < 0.01 * (abs(tail.mean()) + 1e-9) else "ok")
        add("loss_audit", f"{c} (mean / flag)", f"{v.mean():.5f} / {flag}", "history.csv", "derived",
            "near-zero or flat terms contribute no learning signal - check with the abl_* runs")
    wcols = [c for c in h.columns if c.startswith("learned_loss_weights.")]
    if wcols:
        drift = max(abs(h[c].iloc[-1] - h[c].iloc[0]) for c in wcols)
        add("loss_audit", "learned loss weights: max drift first->last epoch", round(float(drift), 4),
            "history.csv", "derived", "< ~0.05 suggests the learned weighting is not differentiating terms")
    return h


def eval_rows(final: Path | None, matched: Path | None):
    if final is not None:
        fe = load_json(final / "final_evaluation.json")
        if fe:
            for dec, h in fe.get("test", {}).items():
                for m in ("qwk", "macro_f1", "accuracy", "min_grade_recall", "recall_spread", "ref_auc", "stdr_auc"):
                    x = h[m]
                    add("test_results", f"[{dec}] {m}", f"{x['point']:.4f} [{x['lo']:.4f}, {x['hi']:.4f}]",
                        "final_evaluation.json", "computed", "patient-level bootstrap 95% CI")
                for g, x in h["recall_by_grade"].items():
                    add("test_results", f"[{dec}] recall grade {g}",
                        f"{x['point']:.4f} [{x['lo']:.4f}, {x['hi']:.4f}]", "final_evaluation.json")
                add("test_results", f"[{dec}] n_images / n_patients", f"{h['n_images']} / {h['n_patients']}",
                    "final_evaluation.json")
            for t, v in fe.get("test_screening", {}).items():
                a = v["applied"]
                for k in ("threshold", "cleared_rate", "npv", "sensitivity", "specificity", "ppv", "missed", "prevalence"):
                    add("test_screening", f"[{t}] {k}", round(a[k], 5) if isinstance(a[k], float) else a[k],
                        "final_evaluation.json", "computed", "threshold fitted on validation")
                add("test_screening", f"[{t}] npv 95% CI (exact)", json.dumps([round(x, 5) for x in a["npv_ci"]]),
                    "final_evaluation.json")
            for name, ext in fe.get("external", {}).items():
                for dec, h in ext.items():
                    if isinstance(h, dict) and "qwk" in h:
                        for m in ("qwk", "macro_f1", "accuracy", "min_grade_recall", "ref_auc", "stdr_auc"):
                            x = h[m]
                            add("external", f"{name} [{dec}] {m}", f"{x['point']:.4f} [{x['lo']:.4f}, {x['hi']:.4f}]",
                                "final_evaluation.json")
            if "seeds" in fe:
                for m in ("qwk", "macro_f1", "ref_auc", "stdr_auc"):
                    s = fe["seeds"][m]
                    add("seeds", f"{m} mean +/- SD over {fe['seeds']['n_runs']} runs",
                        f"{s['mean']:.4f} +/- {s['sd']:.4f}", "final_evaluation.json")
            pr = fe.get("protocol", {})
            for k in ("cutpoints", "logit_adjust_tau", "eye_fusion_alpha"):
                if k in pr:
                    add("decoders", k, json.dumps(pr[k]), "final_evaluation.json")
        else:
            missing("test_results", "test metrics with CIs", "final_evaluation.json", "step 26 has not run")
    if matched is not None:
        fm = load_json(matched / "final_evaluation.json")
        if fm:
            for n, c in fm.get("baselines", {}).items():
                add("comparison", f"{n}: dQWK (reference - model)",
                    f"{c['qwk']['delta']:+.4f} [{c['qwk']['lo']:+.4f}, {c['qwk']['hi']:+.4f}] p_holm={c['qwk']['p_holm']:.3g}",
                    "final_evaluation.json", "computed", "matched-budget, paired, Holm-corrected")


def other_rows(run: Path, cache: Path | None, lesion_out: Path | None):
    if cache and (cache / "meta.csv").exists():
        m = pd.read_csv(cache / "meta.csv")
        if "quality_decision" in m.columns:
            for k, v in m["quality_decision"].value_counts().items():
                add("preprocessing", f"A1 gate decision: {k}", int(v), "cache/meta.csv")
            for c in ("q_quality", "q_domain", "q_lesion", "q_blur", "q_illumination", "quality_score"):
                if c in m.columns:
                    add("preprocessing", f"mean {c}", round(float(m[c].mean()), 4), "cache/meta.csv", "derived")
    if lesion_out and (lesion_out / "history.json").exists():
        h = load_json(lesion_out / "history.json")
        if h:
            last = h[-1] if isinstance(h, list) else h
            for k, v in last.items():
                if k.startswith("dice_") or k in ("train_loss", "epoch"):
                    add("lesion_pretraining", k, round(v, 5) if isinstance(v, float) else v, "lesion_pretrain/history.json")
    cal = run / "calibrator.pt"
    add("calibration", "calibrator.pt present", cal.exists(), "calibrator.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--final", type=Path, default=None)
    ap.add_argument("--matched", type=Path, default=None)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--lesion-out", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    environment_rows(a.run)
    dataset_rows(a.run, a.cache)
    model_rows(a.run)
    history_rows(a.run)
    eval_rows(a.final, a.matched)
    other_rows(a.run, a.cache, a.lesion_out)

    df = pd.DataFrame(ROWS)
    df.to_csv(a.out / "parameters_master.csv", index=False)
    (a.out / "parameters_master.json").write_text(json.dumps(ROWS, indent=2, default=str))
    n_un = int((df.status == "unavailable").sum())
    print(f"[saved] {len(df)} parameters -> {a.out / 'parameters_master.csv'}  "
          f"({n_un} unavailable)")
    if n_un:
        print(df[df.status == "unavailable"][["section", "parameter", "note"]].to_string(index=False))


if __name__ == "__main__":
    main()
