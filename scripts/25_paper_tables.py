#!/usr/bin/env python3
"""Step 25 - training-side tables for the paper, from files the run already writes.

    python scripts/25_paper_tables.py --run outputs/retfound_plus_laft_xai

Reads  history.csv, config.json, gla_lora.json, and (when present, written by
03_train.py) dataset_summary.json and param_summary.json. Writes CSV/Markdown
tables and a score-vs-time figure into <run>/paper_tables/.

Anything that is not available is listed in `missing.txt` instead of being
guessed, so an unfinished run or an older run cannot silently produce a table
that looks complete.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

GRADES = [f"val_recall_grade_{k}" for k in range(5)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    run, out = args.run, args.run / "paper_tables"
    out.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []

    h = pd.read_csv(run / "history.csv")
    cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
    planned = sum(s["epochs"] for s in cfg.get("train", {}).get("stages", [])) if cfg else None

    # derived columns
    h["min_grade_recall"] = h[GRADES].min(axis=1)
    h["recall_spread"] = h[GRADES].max(axis=1) - h[GRADES].min(axis=1)
    h["cum_hours"] = h["epoch_sec"].cumsum() / 3600
    h.to_csv(out / "epoch_table_with_derived.csv", index=False)

    # ---- run status -----------------------------------------------------
    status = {"completed_epochs": int(len(h)), "planned_epochs": planned,
              "complete": bool(planned and len(h) >= planned)}
    if not status["complete"]:
        missing.append(f"run is incomplete ({len(h)}/{planned} epochs): 'best' values "
                       "below are NOT final results")

    # ---- best values -----------------------------------------------------
    best = {
        "best_qwk": float(h["val_quadratic_weighted_kappa"].max()),
        "best_selection_score": float(h["selection_score"].max()),
        "best_min_grade_recall": float(h["min_grade_recall"].max()),
        "best_ref_auc": float(h["val_auc_referable_dr"].max()),
        "best_stdr_auc": float(h["val_auc_sight_threatening"].max()),
        "avg_epoch_minutes": float(h["epoch_sec"].mean() / 60),
        "total_hours": float(h["epoch_sec"].sum() / 3600)}
    # ---- per stage --------------------------------------------------------
    g = h.groupby("stage", sort=False)
    stage = pd.DataFrame({
        "epochs_done": g.size(),
        "avg_qwk": g["val_quadratic_weighted_kappa"].mean(),
        "best_qwk": g["val_quadratic_weighted_kappa"].max(),
        "avg_selection": g["selection_score"].mean(),
        "best_selection": g["selection_score"].max(),
        "avg_ref_auc": g["val_auc_referable_dr"].mean(),
        "avg_min_recall": g["min_grade_recall"].mean(),
        "avg_epoch_min": g["epoch_sec"].mean() / 60})
    stage.round(4).to_csv(out / "per_stage.csv")
    (out / "best_values.json").write_text(json.dumps({**status, **best}, indent=2))

    # ---- final-epoch snapshot (the 'Table of results' columns) -------------
    last = h.iloc[-1]
    snap_cols = {"epoch": "epoch", "stage": "stage", "train_loss": "train_loss",
                 "val_accuracy": "accuracy", "val_f1_macro": "macro_f1",
                 "val_quadratic_weighted_kappa": "qwk", "min_grade_recall": "min_recall",
                 "val_auc_referable_dr": "ref_auc", "val_auc_sight_threatening": "stdr_auc",
                 "val_screen_cleared": "screen_cleared", "val_screen_npv": "screen_npv",
                 "val_screen_threshold": "screen_threshold", "selection_score": "selection_score"}
    snap = {v: last[k] for k, v in snap_cols.items() if k in last.index}
    snap.update({f"recall_grade_{k}": last[f"val_recall_grade_{k}"] for k in range(5)})
    for k in ("mine_p90_hardness", "mine_boundary_rate"):
        if k in last.index:
            snap[k] = last[k]
        else:
            missing.append(f"{k} not in history.csv")
    pd.Series(snap).to_csv(out / "final_epoch_snapshot.csv", header=["value"])

    # ---- parameter budgets ---------------------------------------------------
    ps = run / "param_summary.json"
    if ps.exists():
        (out / "param_summary.json").write_text(ps.read_text())
    else:
        missing.append("param_summary.json (trainable/total params, unfrozen blocks, per-stage LRs, "
                       "RETFound tensor coverage): written by 03_train.py after this update; "
                       "re-run or read the old console log")
    gl = run / "gla_lora.json"
    if gl.exists():
        info = json.loads(gl.read_text())
        ranks = info["ranks"]
        (out / "gla_lora_summary.json").write_text(json.dumps(
            {"lora_params": info["lora_params"], "ranks": ranks,
             "n_blocks_at_max_rank": info.get("n_at_max_rank"),
             "mean_rank": float(np.mean(ranks))}, indent=2))
    else:
        missing.append("gla_lora.json")

    # ---- dataset counts -------------------------------------------------------
    ds = run / "dataset_summary.json"
    if ds.exists():
        (out / "dataset_summary.json").write_text(ds.read_text())
    else:
        missing.append("dataset_summary.json (images/patients/grade counts per split, imbalance "
                       "ratio, grade4:grade0 sampling weight): needs 03_train.py after this update, "
                       "or compute from data/cache/meta.csv")
    if cfg:
        s = cfg.get("sampling", {})
        pd.Series({"sampling_power": s.get("power"),
                   "class_weight_exponent": s.get("class_weight_power")}).to_csv(
            out / "sampling_params.csv", header=["value"])

    # ---- score vs time figure ----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for st, d in h.groupby("stage", sort=False):
            ax.plot(d["cum_hours"], d["selection_score"], marker="o", ms=3, label=st)
        ax.set_xlabel("cumulative training time (h)")
        ax.set_ylabel("composite selection score (validation)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / "score_vs_time.png", dpi=200)
    except Exception as e:                                            # noqa: BLE001
        missing.append(f"score_vs_time.png not drawn: {e}")

    (out / "missing.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
    print(f"[saved] tables in {out}")
    print(json.dumps({**status, **best}, indent=2))
    if missing:
        print("\n[not available]")
        for m in missing:
            print("  -", m)


if __name__ == "__main__":
    main()
