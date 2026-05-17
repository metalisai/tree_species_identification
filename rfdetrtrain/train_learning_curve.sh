#!/usr/bin/env bash
source venv/bin/activate
set -euo pipefail
# NOTE: batch size and accumulation steps multiply to 16
BATCH_SIZE=4
ACCUM_STEPS=4
EPOCHS=80
# Learning-curve dataset root (copied on training machine)
LC_ROOT=../data/compiled_datasets_3_fold/gsd_0p05/datasets_learning_curve
# Temporary per-split dataset roots for train_param.py (contains train/ + valid/)
WORK_ROOT=/tmp/lc_gsd_0p05_runs
mkdir -p "${WORK_ROOT}"
train_split() {
    SPLIT_NAME=$1   # e.g. train_5
    CLASS_SET=$2    # fine|medium|coarse
    SPLIT_SRC="${LC_ROOT}/${SPLIT_NAME}"
    VALID_SRC="${LC_ROOT}/valid"
    DATASET_DIR="${WORK_ROOT}/${SPLIT_NAME}"
    OUTPUT_DIR="train_output_lc_gsd_0p05_${SPLIT_NAME}_${CLASS_SET}_a"
    rm -rf "${DATASET_DIR}"
    mkdir -p "${DATASET_DIR}"
    # Use symlinks so we do not duplicate image files
    ln -s "${SPLIT_SRC}" "${DATASET_DIR}/train"
    ln -s "${VALID_SRC}" "${DATASET_DIR}/valid"
    echo "Training ${SPLIT_NAME} with ${CLASS_SET} classes"
    cp "${DATASET_DIR}/train/_annotations.${CLASS_SET}.coco.json" "${DATASET_DIR}/train/_annotations.coco.json"
    cp "${DATASET_DIR}/valid/_annotations.${CLASS_SET}.coco.json" "${DATASET_DIR}/valid/_annotations.coco.json"
    python train_param.py \
        --aug \
        --dataset_dir "${DATASET_DIR}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --grad_accum_steps "${ACCUM_STEPS}" \
        --output_dir "${OUTPUT_DIR}"
}
train_lc_split() {
    SPLIT_NAME=$1
    train_split "${SPLIT_NAME}" fine
    train_split "${SPLIT_NAME}" medium
    train_split "${SPLIT_NAME}" coarse
}
train_lc_split train_5
train_lc_split train_10
train_lc_split train_20
train_lc_split train_40
train_lc_split train_80
train_lc_split train_100
