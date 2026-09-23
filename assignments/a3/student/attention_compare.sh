#!/bin/bash
# Compare multiplicative, dot, and additive attention with a fixed small-model budget.

set -euo pipefail

ATTENTION_TYPES=(multiplicative dot additive)
SEED=0
EMBED=256
HIDDEN=256
MAX_ITER=400
LR=5e-4
DROPOUT=0.3
BATCH=32

mkdir -p outputs/attention_compare

for ATTENTION in "${ATTENTION_TYPES[@]}"; do
  echo "=== Training attention type: ${ATTENTION} ==="
  mkdir -p "outputs/attention_compare/${ATTENTION}"
  python run.py train \
    --train-src=./zh_en_data/train.zh \
    --train-tgt=./zh_en_data/train.en \
    --dev-src=./zh_en_data/dev.zh \
    --dev-tgt=./zh_en_data/dev.en \
    --vocab=vocab.json \
    --seed="${SEED}" \
    --embed-size="${EMBED}" \
    --hidden-size="${HIDDEN}" \
    --lr="${LR}" \
    --dropout="${DROPOUT}" \
    --batch-size="${BATCH}" \
    --max-iter="${MAX_ITER}" \
    --attention-type="${ATTENTION}" \
    --save-to="outputs/attention_compare/${ATTENTION}/model.bin" \
    --metrics-out="outputs/attention_compare/${ATTENTION}/metrics.json"
done

echo "=== Attention comparison complete ==="
for ATTENTION in "${ATTENTION_TYPES[@]}"; do
  echo "--- ${ATTENTION} ---"
  cat "outputs/attention_compare/${ATTENTION}/metrics.json"
done
