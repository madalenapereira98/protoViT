#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# train_lidc.sh  —  full ProtoPFormer training on LIDC-IDRI
#
# Usage:
#   sh scripts/train_lidc.sh <backbone> <batch_size> <num_gpus>
#
# Example (single GPU, DeiT-Tiny, batch 32):
#   sh scripts/train_lidc.sh deit_tiny_patch16_224 32 1
#
# backbone choices: deit_tiny_patch16_224 | deit_small_patch16_224 | cait_xxs24_224
# ─────────────────────────────────────────────────────────────────────────────

BACKBONE=${1:-deit_tiny_patch16_224}
BATCH=${2:-32}
NGPU=${3:-1}

# Prototype layout:
#   20 local prototypes  (10 per class: benign / malignant)
#   5  global prototypes per class  → 10 total global
# Increase these if you want richer explanations (e.g. 40 local, 10 global).
NUM_PROTO=20
PROTO_DIM=192          # must match ViT embedding dim (192 for DeiT-tiny)
GLOBAL_PPC=5

DATE=$(date +%m%d)
OUTPUT=output/LIDC/${BACKBONE}/${DATE}-protopformer

if [ "$NGPU" -gt 1 ]; then
    LAUNCHER="torchrun --nproc_per_node=${NGPU}"
else
    LAUNCHER="python"
fi

$LAUNCHER main.py \
    --data_path          datasets/LIDC \
    --lidc_format        folder \
    --base_architecture  ${BACKBONE} \
    --img_size           224 \
    --prototype_shape    ${NUM_PROTO} ${PROTO_DIM} 1 1 \
    --add_on_layers_type regular \
    --reserve_layers     11 \
    --reserve_token_nums 81 \
    --use_global         True \
    --global_proto_per_class ${GLOBAL_PPC} \
    --global_coe         0.3 \
    --use_ppc_loss       True \
    --ppc_cov_thresh     2.0 \
    --ppc_mean_thresh    2.0 \
    --ppc_cov_coe        0.1 \
    --ppc_mean_coe       0.5 \
    --use_class_weights  True \
    --epochs             100 \
    --batch_size         ${BATCH} \
    --opt                adamw \
    --weight_decay       0.05 \
    --lr                 5e-4 \
    --features_lr        1e-4 \
    --add_on_layers_lr   3e-3 \
    --prototype_vectors_lr 3e-3 \
    --sched              cosine \
    --warmup-epochs      5 \
    --min-lr             1e-5 \
    --seed               42 \
    --num_workers        4 \
    --output_dir         ${OUTPUT} \
    2>&1 | tee ${OUTPUT}/train.log
