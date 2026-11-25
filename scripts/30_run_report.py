#!/usr/bin/env python3
"""Step 30 - the run report: one PDF (plus a CSV of every table) from the files a run wrote.

    python scripts/30_run_report.py --run outputs/study_seed1337 --study outputs/study \
        [--cache data/cache] --out outputs/study/report

Pages: summary and environment; data and class imbalance; training curves (stage-shaded);
per-stage table; test results with confidence intervals; screening; external cohort;
matched-budget comparison; loss audit; the full parameter table (paginated).
Every page degrades gracefully: a missing source produces a note on the page, not a crash.
Uses matplotlib only, so no extra dependency is needed on a fresh cloud machine.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

GRADES = [f"val_recall_grade_{k}" for k in range(5)]
A4 = (8.27, 11.69)


def jl(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except Exception:                                              # noqa: BLE001
        return None


def page_text(pdf, title: str, lines: list[str], size: int = 8):
    fig = plt.figure(figsize=A4)
    fig.text(0.07, 0.96, title, fontsize=14, weight="bold", va="top")
    y = 0.92
    for ln in lines:
        for seg in (textwrap.wrap(ln, 120) or [""]):
            fig.text(0.07, y, seg, fontsize=size, family="monospace", va="top")
            y -= 0.0165
            if y < 0.04:
                break
    pdf.savefig(fig)
    plt.close(fig)


def page_table(pdf, title: str, df: pd.DataFrame, rows_per_page: int = 38, size: float = 6.5,
               col_widths=None):
    if df is None or df.empty:
        page_text(pdf, title, ["(no data available for this section)"])
        return
    for i in range(0, len(df), rows_per_page):
        chunk = df.iloc[i:i + rows_per_page].astype(str)
        chunk = chunk.apply(lambda c: c.str.slice(0, 70))
        fig, ax = plt.subplots(figsize=A4)
        ax.axis("off")
        ax.set_title(f"{title}" + (f"  (page {i // rows_per_page + 1})" if len(df) > rows_per_page else ""),
                     fontsize=12, weight="bold", loc="left")
        t = ax.table(cellText=chunk.values, colLabels=list(chunk.columns), loc="upper center",
                     colWidths=col_widths, cellLoc="left")
        t.auto_set_font_size(False)
        t.set_fontsize(size)
        t.scale(1, 1.25)
        pdf.savefig(fig)
        plt.close(fig)


def shade_stages(ax, h):
    cols = ["#f4f4f4", "#e6eef7", "#f4f4f4", "#e6eef7", "#f4f4f4", "#e6eef7"]
    for k, (st, g) in enumerate(h.groupby("stage", sort=False)):
        ax.axvspan(g["epoch"].min() - 0.5, g["epoch"].max() + 0.5, color=cols[k % 6], zorder=0)
        ax.text((g["epoch"].min() + g["epoch"].max()) / 2, ax.get_ylim()[1], st, ha="center",
                va="bottom", fontsize=6, rotation=0)


def curves(pdf, h: pd.DataFrame):
    spec = [("train_loss", "training loss"), ("val_quadratic_weighted_kappa", "QWK (validation)"),
            ("selection_score", "composite selection score"), ("val_f1_macro", "macro-F1"),
            ("val_auc_referable_dr", "referable-DR AUC"), ("val_auc_sight_threatening", "STDR AUC"),
            ("val_screen_cleared", "screening: fraction cleared"), ("val_screen_npv", "screening: NPV"),
            ("epoch_sec", "epoch wall-clock (s)")]
    fig, axes = plt.subplots(5, 2, figsize=A4)
    for ax, (c, lab) in zip(axes.ravel(), spec):
        if c in h.columns:
            ax.plot(h["epoch"], h[c], lw=1.2)
            ax.set_title(lab, fontsize=8)
            shade_stages(ax, h)
        ax.tick_params(labelsize=6)
    ax = axes.ravel()[len(spec)]
    for g in GRADES:
        ax.plot(h["epoch"], h[g], lw=1, label=g.replace("val_recall_grade_", "grade "))
    ax.set_title("per-grade recall", fontsize=8)
    ax.legend(fontsize=5, ncol=3)
    ax.tick_params(labelsize=6)
    fig.suptitle("Training curves (validation)", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pdf.savefig(fig)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=A4)
    cum = h["epoch_sec"].cumsum() / 3600
    for st, g in h.groupby("stage", sort=False):
        axes[0].plot(cum[g.index], g["selection_score"], marker="o", ms=3, label=st)
    axes[0].set_xlabel("cumulative training time (h)")
    axes[0].set_ylabel("selection score")
    axes[0].set_title("Score versus training time")
    axes[0].legend(fontsize=7)
    cols = [c for c in h.columns if c.startswith("train_") and c != "train_loss"]
    for c in cols:
        axes[1].plot(h["epoch"], h[c], lw=1, label=c.replace("train_", ""))
    axes[1].set_yscale("symlog", linthresh=1e-2)
    axes[1].set_title("Individual loss terms (symlog) - flat or ~0 terms are not learning")
    axes[1].legend(fontsize=6, ncol=3)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--study", type=Path, default=None)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--params", type=Path, default=None, help="parameters_master.csv")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tables = a.out / "tables"
    tables.mkdir(exist_ok=True)

    h = pd.read_csv(a.run / "history.csv") if (a.run / "history.csv").exists() else None
    pm = pd.read_csv(a.params) if a.params and Path(a.params).exists() else None
    final = jl(a.study / "final" / "final_evaluation.json") if a.study else None
    matched = jl(a.study / "final_matched" / "final_evaluation.json") if a.study else None
    ds = jl(a.run / "dataset_summary.json")

    with PdfPages(a.out / "run_report.pdf") as pdf:
        # ---- 1. summary -------------------------------------------------------
        lines = [f"generated   : {datetime.now():%Y-%m-%d %H:%M:%S}", f"run folder  : {a.run}", ""]
        if h is not None:
            b = h.loc[h["selection_score"].idxmax()]
            lines += [f"epochs completed : {len(h)}   last stage: {h['stage'].iloc[-1]}",
                      f"best epoch       : {int(b['epoch'])} (selection score {b['selection_score']:.4f})",
                      f"validation at best epoch: QWK {b['val_quadratic_weighted_kappa']:.4f}  "
                      f"macro-F1 {b['val_f1_macro']:.4f}  refAUC {b['val_auc_referable_dr']:.4f}  "
                      f"STDR AUC {b['val_auc_sight_threatening']:.4f}",
                      f"training time    : {h['epoch_sec'].sum() / 3600:.2f} h total, "
                      f"{h['epoch_sec'].mean() / 60:.1f} min/epoch", ""]
        if final and final.get("test"):
            t = final["test"].get("as_saved", next(iter(final["test"].values())))
            lines += ["TEST (patient-level bootstrap 95% CI):",
                      f"  QWK      {t['qwk']['point']:.4f}  [{t['qwk']['lo']:.4f}, {t['qwk']['hi']:.4f}]",
                      f"  macro-F1 {t['macro_f1']['point']:.4f}  [{t['macro_f1']['lo']:.4f}, {t['macro_f1']['hi']:.4f}]",
                      f"  refAUC   {t['ref_auc']['point']:.4f}  [{t['ref_auc']['lo']:.4f}, {t['ref_auc']['hi']:.4f}]",
                      f"  STDR AUC {t['stdr_auc']['point']:.4f}  [{t['stdr_auc']['lo']:.4f}, {t['stdr_auc']['hi']:.4f}]", ""]
        else:
            lines += ["TEST results: not available (step 26 has not produced final_evaluation.json)", ""]
        if pm is not None:
            env = pm[pm.section == "environment"][["parameter", "value"]]
            lines += ["ENVIRONMENT"] + [f"  {r.parameter:22s} {r.value}" for r in env.itertuples()]
            unav = int((pm.status == "unavailable").sum())
            lines += ["", f"parameters logged: {len(pm)} ({unav} unavailable - see the last pages)"]
        lines += ["", "Read with care: validation numbers select checkpoints; only TEST numbers are results.",
                  "A run that stopped early or is incomplete reports 'best so far', not final values."]
        page_text(pdf, "RETFound Plus-LAFT-XAI - run report", lines, size=9)

        # ---- 2. data ----------------------------------------------------------
        if ds:
            fig, axes = plt.subplots(1, 2, figsize=(11.69, 5))
            splits = list(ds["per_split"])
            w = 0.8 / max(len(splits), 1)
            for i, sp in enumerate(splits):
                axes[0].bar(np.arange(5) + i * w, ds["per_split"][sp]["grade_counts"], w, label=sp)
            axes[0].set_xticks(np.arange(5) + 0.4 - w / 2)
            axes[0].set_xticklabels([f"grade {g}" for g in range(5)])
            axes[0].set_title("images per grade and split")
            axes[0].legend()
            tr = np.array(ds["per_split"].get("train", {}).get("grade_counts", [0] * 5), float)
            if tr.sum():
                axes[1].bar(range(5), tr / tr.sum())
                axes[1].set_title(f"training class distribution (max:min = "
                                  f"{tr.max() / max(tr.min(), 1):.1f}:1)")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        else:
            page_text(pdf, "Data", ["dataset_summary.json not found for this run (older version of 03_train.py)."])

        # ---- 3. training curves / stages ----------------------------------------
        if h is not None:
            curves(pdf, h)
            h2 = h.copy()
            h2["min_grade_recall"] = h2[GRADES].min(axis=1)
            st = h2.groupby("stage", sort=False).agg(
                epochs=("epoch", "count"), avg_qwk=("val_quadratic_weighted_kappa", "mean"),
                best_qwk=("val_quadratic_weighted_kappa", "max"), avg_sel=("selection_score", "mean"),
                best_sel=("selection_score", "max"), avg_refauc=("val_auc_referable_dr", "mean"),
                avg_min_recall=("min_grade_recall", "mean"), avg_epoch_min=("epoch_sec", lambda x: x.mean() / 60)
            ).round(4).reset_index()
            st.to_csv(tables / "per_stage.csv", index=False)
            page_table(pdf, "Per-stage summary (validation)", st, size=7)
        else:
            page_text(pdf, "Training", ["history.csv not found."])

        # ---- 4. test / screening / external ----------------------------------------
        if final:
            rows = []
            for dec, t in final.get("test", {}).items():
                rows.append({"decoder": dec, **{m: f"{t[m]['point']:.4f} [{t[m]['lo']:.4f},{t[m]['hi']:.4f}]"
                                                for m in ("qwk", "macro_f1", "accuracy", "min_grade_recall", "ref_auc", "stdr_auc")}})
            df = pd.DataFrame(rows)
            df.to_csv(tables / "test_results.csv", index=False)
            page_table(pdf, "Test results, patient-level bootstrap 95% CI", df, size=6, rows_per_page=20)
            sc = []
            for k, v in final.get("test_screening", {}).items():
                ap_ = v["applied"]
                sc.append({"npv_target": k.split("_")[-1], "cleared": f"{ap_['cleared_rate']:.3f}",
                           "npv": f"{ap_['npv']:.4f}", "npv_95ci": f"[{ap_['npv_ci'][0]:.4f},{ap_['npv_ci'][1]:.4f}]",
                           "sens": f"{ap_['sensitivity']:.3f}", "spec": f"{ap_['specificity']:.3f}",
                           "ppv": f"{ap_['ppv']:.3f}", "missed": ap_["missed"], "prevalence": f"{ap_['prevalence']:.3f}"})
            sdf = pd.DataFrame(sc)
            sdf.to_csv(tables / "screening.csv", index=False)
            page_table(pdf, "Screening (STDR grade>=3): threshold fitted on validation, applied to test", sdf, size=7, rows_per_page=20)
            ex = []
            for name, e in final.get("external", {}).items():
                for dec, t in e.items():
                    if isinstance(t, dict) and "qwk" in t:
                        ex.append({"cohort": name, "decoder": dec, "n": t["n_images"],
                                   **{m: f"{t[m]['point']:.3f} [{t[m]['lo']:.3f},{t[m]['hi']:.3f}]"
                                      for m in ("qwk", "macro_f1", "accuracy", "ref_auc", "stdr_auc")}})
            edf = pd.DataFrame(ex)
            edf.to_csv(tables / "external.csv", index=False)
            page_table(pdf, "External cohort (grading metrics only; not a screening population)", edf, size=6, rows_per_page=20)
            if "seeds" in final:
                s = final["seeds"]
                page_text(pdf, f"Seed stability ({s['n_runs']} runs, identical split)",
                          [f"{m}: {s[m]['mean']:.4f} +/- {s[m]['sd']:.4f}   values {np.round(s[m]['values'], 4).tolist()}"
                           for m in ("qwk", "macro_f1", "ref_auc", "stdr_auc")])
        else:
            page_text(pdf, "Test results", ["final_evaluation.json not found - step 26 did not run or failed; see logs."])
        if matched and matched.get("baselines"):
            rows = [{"model": n, "dQWK [CI]": f"{c['qwk']['delta']:+.3f} [{c['qwk']['lo']:+.3f},{c['qwk']['hi']:+.3f}]",
                     "p_holm": f"{c['qwk']['p_holm']:.3g}", "dMacroF1": f"{c['macro_f1']['delta']:+.3f}",
                     "refAUC p": f"{c['ref_auc']['p_holm']:.3g}", "STDR p": f"{c['stdr_auc']['p_holm']:.3g}"}
                    for n, c in matched["baselines"].items()]
            mdf = pd.DataFrame(rows)
            mdf.to_csv(tables / "matched_comparison.csv", index=False)
            page_table(pdf, "Matched-budget comparison: reference minus model (positive = reference better)", mdf, size=6.5)

        # ---- 5. full parameter table ----------------------------------------------------
        if pm is not None:
            pm[["section", "parameter", "value", "status", "note"]].to_csv(tables / "parameters_master.csv", index=False)
            page_table(pdf, "Complete parameter log", pm[["section", "parameter", "value", "status"]],
                       rows_per_page=52, size=5.5, col_widths=[0.18, 0.42, 0.28, 0.12])
        d = pdf.infodict()
        d["Title"] = "RETFound Plus-LAFT-XAI run report"
        d["CreationDate"] = datetime.now()
    print(f"[saved] {a.out / 'run_report.pdf'}")


if __name__ == "__main__":
    main()
