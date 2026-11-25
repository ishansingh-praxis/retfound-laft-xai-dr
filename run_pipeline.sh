#!/usr/bin/env bash
# Sequential end-to-end run.
#
# Deliberately sequential: on a laptop SSD with 8 GB of unified memory, running
# the download, the 1024 px preprocessing pass and a ViT-Large training loop at
# the same time makes all three I/O-starved and none of them finish. Each stage
# writes its own log under outputs/logs/ and is skipped if its output already
# exists, so the script is safe to re-run after an interruption.
set -uo pipefail
cd "$(dirname "$0")"
PY=".venv/bin/python"
mkdir -p outputs/logs

CACHE_PX=1024
GLOBAL_PX=${GLOBAL_PX:-512}   # revised: 512 px global view (1024 px is not used)
CROP_PX=320
CROP_IN=${CROP_IN:-288}
N_CROPS=${N_CROPS:-8}
# Minority-first cache: every patient with any DR is kept, No-DR patients fill the
# rest. 1024 px is ~3.2 MB/image, so 15000 images is ~47 GB. Raise on a big disk.
LIMIT=${LIMIT:-15000}
SEED=${SEED:-1337}
TAG=${TAG:-retfound_plus_laft_xai}
BS=${BS:-2}            # laptop default; raise on a 24 GB+ GPU
ACCUM=${ACCUM:-8}
AMP_FLAG=""; [ "${AMP:-0}" = "1" ] && AMP_FLAG="--amp"
STAGES="${STAGES:-2,4,4,4}"
SPE="${SPE:-2000}"
TRAIN_BUDGET_MIN="${TRAIN_BUDGET_MIN:-330}"

step () { echo; echo "=============================================================="
          echo " $1"; echo "=============================================================="; date; }

step "1  extract the 1024 px EyePACS mirror"
if [ ! -d data/raw/eyepacs-hi ] || [ -z "$(find data/raw/eyepacs-hi -name '*.jpeg' -print -quit 2>/dev/null)" ]; then
  $PY -u scripts/01_download_data.py --dataset eyepacs-hi 2>&1 | tee outputs/logs/01_download.log | tail -5
else
  echo "already extracted"
fi
find data/raw/eyepacs-hi -name "*.jpeg" | head -1 | xargs -I{} $PY -c \
  "import cv2,sys;print('sample image size:', cv2.imread('{}').shape)"

step "2  Q* gate + 1024 px field extraction + priors + lesion crops"
# ~2 h at 1024 px: skip if a cache with matching geometry is already present
if $PY -c "
import sys,numpy as np,pandas as pd,pathlib
d=pathlib.Path('data/cache')
sys.exit(0 if (d/'crops.npy').exists() and (d/'meta.csv').exists()
         and np.load(d/'images.npy', mmap_mode='r').shape[1]==$CACHE_PX
         and np.load(d/'crops.npy', mmap_mode='r').shape[1]==$N_CROPS
         and len(pd.read_csv(d/'meta.csv'))==np.load(d/'images.npy',mmap_mode='r').shape[0]
         else 1)" 2>/dev/null; then
  echo "cache already built at ${CACHE_PX}px with ${N_CROPS} crops - skipping"
else
$PY -u scripts/02_preprocess.py --root data/raw/eyepacs-hi --limit "$LIMIT" \
    --image-size "$CACHE_PX" --n-crops "$N_CROPS" --crop-size "$CROP_PX" \
    --minority-first --split-file data/split.json --seed "$SEED" \
    --workers 5 > outputs/logs/02_preprocess.log 2>&1
fi
tail -30 outputs/logs/02_preprocess.log

step "2b pretrain the A3 lesion experts on real DDR lesion masks"
# (scripts/11_download_lesions.py now fetches DDR only; pass --only idrid ddr for both)
if [ -f outputs/lesion_pretrain/lesion_encoder.pt ] && [ "${FORCE_PRETRAIN:-0}" != "1" ]; then
  echo "lesion encoder already trained - skipping (FORCE_PRETRAIN=1 to redo)"
else
  $PY -u scripts/10_lesion_pretrain.py --image-size "$CROP_IN" --batch-size "${PRETRAIN_BS:-6}" \
      --epochs "${PRETRAIN_EPOCHS:-10}" --crops-per-image 8 > outputs/logs/10_lesion_pretrain.log 2>&1
  tail -15 outputs/logs/10_lesion_pretrain.log
fi

step "3  staged training"
ENC=""
[ -f outputs/lesion_pretrain/lesion_encoder.pt ] && \
  ENC="--lesion-encoder outputs/lesion_pretrain/lesion_encoder.pt"
$PY -u scripts/03_train.py --stage-epochs "$STAGES" --samples-per-epoch "$SPE" \
    --max-val 5000 --batch-size "$BS" --grad-accum "$ACCUM" $AMP_FLAG \
    --seed "$SEED" --tag "$TAG" \
    --global-size "$GLOBAL_PX" --crop-input "$CROP_IN" --n-crops "$N_CROPS" \
    --time-budget-min "$TRAIN_BUDGET_MIN" $ENC \
    > outputs/logs/03_train.log 2>&1
tail -25 outputs/logs/03_train.log

step "4  full evaluation"
CK="outputs/$TAG/best.pt"
# validation + test predictions (with TTA) feed scripts/26_final_evaluation.py
$PY -u scripts/04_evaluate.py --ckpt "$CK" --split val --tta --xai-samples 0 \
    > outputs/logs/04_evaluate_val.log 2>&1
$PY -u scripts/04_evaluate.py --ckpt "$CK" --split test --tta --xai-samples 120 \
    > outputs/logs/04_evaluate.log 2>&1
tail -40 outputs/logs/04_evaluate.log

step "9  screening triage"
$PY -u scripts/09_screening.py > outputs/logs/09_screening.log 2>&1
tail -20 outputs/logs/09_screening.log

step "5  external cohort: FDR26 (grading metrics only)"
if [ ! -d data/raw/fdr26/images ]; then
  $PY -u scripts/01_download_data.py --dataset fdr26 2>&1 | tail -3
fi
if [ ! -f data/cache_fdr26/meta.csv ]; then
  $PY -u scripts/02_preprocess.py --root data/raw/fdr26 --fdr26 --out data/cache_fdr26 \
      --image-size "$CACHE_PX" --n-crops "$N_CROPS" --crop-size "$CROP_PX" \
      --quality-reference data/cache/quality_reference.json --workers 5 \
      > outputs/logs/02_preprocess_fdr26.log 2>&1
fi
$PY -u scripts/04_evaluate.py --ckpt "$CK" --split test --tta --cache data/cache_fdr26 \
    --pred-tag fdr26 --xai-samples 0 --out "outputs/$TAG/evaluation_fdr26.json" \
    > outputs/logs/04_evaluate_fdr26.log 2>&1

step "6  final evaluation + paper tables"
$PY -u scripts/26_final_evaluation.py \
    --val "outputs/$TAG/predictions_val_tta.npz" --test "outputs/$TAG/predictions_test_tta.npz" \
    --external fdr26="outputs/$TAG/predictions_fdr26.npz" --out "outputs/$TAG/final" \
    > outputs/logs/26_final.log 2>&1
tail -30 outputs/logs/26_final.log
$PY -u scripts/25_paper_tables.py --run "outputs/$TAG" > outputs/logs/25_tables.log 2>&1
tail -15 outputs/logs/25_tables.log

step "DONE"
