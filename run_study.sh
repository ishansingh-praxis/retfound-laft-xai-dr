#!/usr/bin/env bash
# Full study: several seeds of the main model, matched baselines and ablations,
# then the final statistics. Run on the training machine (CUDA GPU strongly advised).
#
#   RETFOUND_CKPT=data/checkpoints/RETFound_MAE/pytorch_model.bin \
#   SEEDS="1337 2024 7" BS=8 ACCUM=2 AMP=1 bash run_study.sh
#
# Every stage is resumable / skip-aware: re-running after an interruption continues
# instead of starting over (cache, lesion encoder and training checkpoints are reused).
#
# Matched-budget note: scripts/07_compare_models.py trains baselines and ablations with
# its OWN short loop (EPOCHS below), not the 4-stage schedule of 03_train.py. The
# baselines are therefore compared at a stated, smaller budget. Say so in the paper,
# or give 07 the same data/epochs as far as your compute allows.
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
SEEDS=${SEEDS:-"1337 2024 7"}
EPOCHS=${EPOCHS:-6}                    # per baseline / ablation in 07
CMP_BS=${CMP_BS:-32}
CMP_IMG=${CMP_IMG:-224}
RETFOUND_CKPT=${RETFOUND_CKPT:-data/checkpoints/RETFound_MAE/pytorch_model.bin}
BASE_TAG=${BASE_TAG:-study}
OUT=outputs/$BASE_TAG
mkdir -p "$OUT" outputs/logs

set -- $SEEDS
FIRST=$1
echo "== seeds: $SEEDS  (first: $FIRST)"

# ---- A. main model, one full pipeline per seed -------------------------------
for s in $SEEDS; do
  echo; echo "=== main model, seed $s ==="
  SEED=$s TAG="${BASE_TAG}_seed$s" bash run_pipeline.sh
done

# ---- B. baselines + ablations under one matched budget (first seed) -------------
echo; echo "=== baselines and ablations (07) ==="
RF=""
[ -f "$RETFOUND_CKPT" ] && RF="--retfound-ckpt $RETFOUND_CKPT"
$PY -u scripts/07_compare_models.py --cache data/cache --epochs "$EPOCHS" \
    --max-train 0 --batch-size "$CMP_BS" --image-size "$CMP_IMG" --seed "$FIRST" \
    $RF --out "$OUT/comparison" > outputs/logs/07_compare.log 2>&1
tail -45 outputs/logs/07_compare.log

# ---- C. label-noise pass (optional second round) --------------------------------
echo; echo "=== label-noise flags from the first seed's train predictions ==="
CK="outputs/${BASE_TAG}_seed${FIRST}/best.pt"
if [ -f "$CK" ]; then
  $PY -u scripts/04_evaluate.py --ckpt "$CK" --split train --xai-samples 0 \
      > outputs/logs/04_evaluate_train.log 2>&1
  $PY -u scripts/27_flag_label_noise.py \
      --pred "outputs/${BASE_TAG}_seed${FIRST}/predictions_train.npz" \
      --out "$OUT/noise_weights.csv" | tail -12
  echo "[next] retrain with:  --noise-weights $OUT/noise_weights.csv  (a new --tag) and compare"
fi

# ---- D. final statistics --------------------------------------------------------
MAIN="outputs/${BASE_TAG}_seed${FIRST}"

# D1. the main model (full 4-stage schedule): CIs, validation-fitted decoders,
#     screening, FDR26, and seed stability. No baselines here: they train on a
#     different (shorter) budget, so comparing them to this model would be unfair.
echo; echo "=== D1: main model, full schedule ==="
ARGS=(--val "$MAIN/predictions_val_tta.npz" --test "$MAIN/predictions_test_tta.npz"
      --out "$OUT/final")
[ -f "$MAIN/predictions_fdr26.npz" ] && ARGS+=(--external "fdr26=$MAIN/predictions_fdr26.npz")
OTHERS=()
for s in $SEEDS; do
  [ "$s" = "$FIRST" ] && continue
  p="outputs/${BASE_TAG}_seed$s/predictions_test_tta.npz"
  [ -f "$p" ] && OTHERS+=("$p")
done
[ ${#OTHERS[@]} -gt 0 ] && ARGS+=(--seed-test "${OTHERS[@]}")
$PY -u scripts/26_final_evaluation.py "${ARGS[@]}" > outputs/logs/26_final_study.log 2>&1
tail -50 outputs/logs/26_final_study.log

# D2. matched-budget comparison: 07's own 'complete' model is the reference, so the
#     baselines and every ablation share data, schedule and decoder with it.
echo; echo "=== D2: matched-budget baselines and ablations (reference = 07 'complete') ==="
if [ -f "$OUT/comparison/predictions_complete.npz" ]; then
  ARGS=(--test "$OUT/comparison/predictions_complete.npz" --out "$OUT/final_matched")
  for f in "$OUT"/comparison/predictions_*.npz; do
    n=$(basename "$f" .npz); n=${n#predictions_}
    [ "$n" = "complete" ] && continue
    ARGS+=(--baseline "$n=$f")
  done
  $PY -u scripts/26_final_evaluation.py "${ARGS[@]}" > outputs/logs/26_final_matched.log 2>&1
  tail -40 outputs/logs/26_final_matched.log
else
  echo "no 07 'complete' predictions found - skipping D2"
fi
echo; echo "Results: $OUT/final/final_evaluation.md  and  $OUT/final_matched/final_evaluation.md"
