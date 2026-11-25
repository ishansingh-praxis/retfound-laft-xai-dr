# Runbook: one command on a cloud GPU

```bash
export KAGGLE_API_TOKEN=...                  # EyePACS + DDR: new-style Kaggle token (Kaggle -> Settings -> API)
# or the legacy pair:  export KAGGLE_USERNAME=...  KAGGLE_KEY=...   (or a ~/.kaggle/access_token / kaggle.json file)
export HF_TOKEN=...                          # RETFound weights (request access once on HuggingFace)
bash cloud_start.sh --smoke                  # ~30 min, synthetic data, CPU: proves every step works
bash cloud_start.sh                          # the real study
tail -f outputs/study/logs/run_all.log
```

`cloud_start.sh` creates a venv, installs `requirements.txt`, and starts `run_all.py` detached.
On any interruption or reboot, **run the same command again**: finished steps are skipped and
training resumes from its checkpoint. Run on the machine directly with `python run_all.py [options]`.

## What `run_all.py` does, in order

| step | what | critical |
|---|---|---|
| preflight | Python/packages (auto-installs), GPU, disk, credentials, network; writes `run_meta.json`; **stops before any long step** if something is missing | yes |
| selftest | all architecture blocks on synthetic tensors | yes |
| download_eyepacs / _retfound / _lesions / _fdr26 | EyePACS 1024 px mirror, RETFound CFP weights, DDR lesion masks, FDR26 external cohort (checksummed) | eyepacs, retfound |
| preprocess / preprocess_external | A1 quality gate, A2 field extraction, lesion priors, crop centres; **minority-first** subsample (every patient with any DR kept), frozen patient split | preprocess |
| lesion_pretrain | A3 lesion experts on real DDR masks | no |
| train_seed<N> (x3 seeds) | 5 stages: frozen -> LoRA -> partial unfreeze -> full -> **decoupled head** (class-balanced, from best checkpoint, kept only if the selection score improves). 512 px global view + 8 crops at 288 px | yes |
| eval_val / eval_test / eval_external | predictions (TTA), full 38-parameter evaluation, XAI, robustness, external cohort | test |
| screening | thresholds fitted on validation, applied to test | no |
| xai_expert_masks | attribution vs **held-out** expert masks (never used in lesion pretraining) | no |
| train_predictions / label_noise | flags probable label errors -> `noise_weights.csv` (+ review queue) | no |
| comparison / matched_stats | 16 baselines + 16 ablations (incl. plain RETFound) under one budget, paired + Holm | no |
| final_stats | patient-level bootstrap CIs, validation-fitted decoders (QWK cut-points, logit adjustment, eye-pair fusion), screening with exact NPV CIs, seed mean +/- SD | no |
| paper_tables / parameters / report | per-stage tables, **parameters_master.csv**, **run_report.pdf** | no |

Critical steps stop the run (with the fix printed); optional ones are recorded as failed and the run
continues. The reports are produced **even after a failure**, from whatever finished.

## Everything is logged (under `outputs/<run>/`)

| file | content |
|---|---|
| `RESULTS_INDEX.md` | step table + where every output is |
| `report/run_report.pdf`, `report/tables/*.csv` | summary, data/imbalance, training curves, per-stage, test CIs, screening, external, matched comparison, full parameter log |
| `parameters/parameters_master.csv` / `.json` | every requested parameter and more: section, value, **source**, status (`computed` / `derived` / `unavailable` + reason) |
| `final/final_evaluation.{md,json}`, `final_matched/` | statistics from saved predictions |
| `seed<N>/` | `history.csv`, `config.json`, `param_summary.json`, `dataset_summary.json`, `gla_lora.json`, `evaluation*.json`, `predictions_*.npz`, `best.pt` |
| `logs/<step>.log`, `logs/run_all.log`, `steps.csv`, `logs/resources.csv` | per-step logs, master log, step timings/status, CPU/RAM/GPU/disk every 60 s |
| `run_meta.json` | arguments, versions, GPU, git commit, seeds, batch/accum/AMP |

## Options

`--seeds 1337` (one seed), `--limit N` (cache size; 1024 px is ~3.2 MB/image, default 15,000 = ~47 GB),
`--bs/--accum` (default chosen from GPU memory), `--skip-comparison`, `--skip-fdr26`, `--skip-lesion`,
`--only a,b`, `--skip a,b`, `--force a,b|all`, `--preflight-only`, `--allow-cpu`, `--data-dir`.

## What is real and what is not

* **Verified here (CPU, synthetic data):** the architecture self-test (25/25), the rebuilt `dr.data` package,
  preprocessing, lesion pretraining, the statistics and the reports. Run `--smoke` on the cloud machine first:
  it exercises every step in the order the real run uses.
* **Not verifiable here, so expect first-contact fixes:** behaviour on the real EyePACS/DDR/FDR26 files and
  a CUDA run of ViT-Large at 512 px. The lesion-annotation loader matches images to masks by folder/file name
  tokens (tested on fake IDRiD- and DDR-style trees, not on the real downloads): check the
  `lesion_pretrain` log's "annotated images ... masks_per_channel" line.
* **Replaced `dr.data`:** `src/dr/data/` was missing from the repository and has been rebuilt from how the
  scripts use it (manifest/split code from an older copy, new cached dataset and lesion loaders). If you have the
  original, diff it against this one before trusting results.
* **Not implemented:** feature-level two-eye fusion (prediction-level fusion is); A7 temporal / A8 clinical /
  A11 domain adaptation remain unevaluable on these datasets; the inactive loss terms are *flagged* in the loss
  audit (`parameters_master.csv`, section `loss_audit`) but not fixed - the `abl_*` runs show whether they matter.
* FDR26 is near-balanced and curated: its NPV / cleared-rate are deliberately not reported; patient ids there are
  inferred from file names.
* The old `outputs/retfound_plus_laft_xai` results predate all of this (32/45 epochs, 6,000-image subset, 224 px).
