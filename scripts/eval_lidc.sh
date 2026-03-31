#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# eval_lidc.sh  —  evaluate a trained checkpoint and generate visualisations
#
# Usage:
#   sh scripts/eval_lidc.sh <checkpoint_path> [backbone]
#
# Example:
#   sh scripts/eval_lidc.sh output/LIDC/deit_tiny_patch16_224/0330-protopformer/checkpoints/best-auc.pth
# ─────────────────────────────────────────────────────────────────────────────

CKPT=${1:?"Please provide checkpoint path as first argument"}
BACKBONE=${2:-deit_tiny_patch16_224}
OUT_DIR=$(dirname $(dirname $CKPT))/eval_$(date +%m%d)

# ── 1) Evaluation metrics ─────────────────────────────────────────────────────
echo "=== Evaluating checkpoint: $CKPT ==="
python main.py \
    --eval \
    --data_path         datasets/LIDC \
    --lidc_format       folder \
    --base_architecture ${BACKBONE} \
    --prototype_shape   20 192 1 1 \
    --use_global        True \
    --global_proto_per_class 5 \
    --reserve_layers    11 \
    --reserve_token_nums 81 \
    --resume            ${CKPT} \
    --output_dir        ${OUT_DIR}

# ── 2) Prototype visualisations ───────────────────────────────────────────────
echo ""
echo "=== Generating prototype visualisations ==="
python visualize_lidc.py \
    --data_path         datasets/LIDC \
    --lidc_format       folder \
    --model_path        ${CKPT} \
    --output_dir        ${OUT_DIR}/visualisations \
    --base_architecture ${BACKBONE} \
    --prototype_shape   20 192 1 1 \
    --use_global        True \
    --global_proto_per_class 5 \
    --reserve_layers    11 \
    --reserve_token_nums 81 \
    --top_k             5

echo ""
echo "Done. Results saved to: ${OUT_DIR}"
