#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="."
DATASET_ROOT="../data/compiled_datasets_3_fold"
GSD="gsd_0p05"
SPLIT_ROOT_NAME="datasets_learning_curve"
TRAIN_OUTPUT_ROOT="${ROOT_DIR}/runs_lc_gsd0p05"
TEST_EVAL_OUTPUT_ROOT="${ROOT_DIR}/runs_lc_gsd0p05_testeval"
LOG_DIR="${ROOT_DIR}/logs_lc_gsd0p05"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "${TRAIN_OUTPUT_ROOT}" "${TEST_EVAL_OUTPUT_ROOT}" "${LOG_DIR}"

TRAIN_SPLITS=("train_5" "train_10" "train_20" "train_40" "train_80" "train_100")
FINENESS=("fine" "medium" "coarse")
EPOCHS=60

for train_split in "${TRAIN_SPLITS[@]}"; do
  for fineness in "${FINENESS[@]}"; do
    ann_file="_annotations.${fineness}.coco.json"
    run_name="${GSD}___${train_split}___${ann_file%.json}"
    run_suffix="${train_split}"
    train_log_file="${LOG_DIR}/${run_name}.train.log"
    test_log_file="${LOG_DIR}/${run_name}.testeval.log"

    echo "[$(date -Is)] Starting train ${run_name} epochs=${EPOCHS}" | tee -a "${LOG_DIR}/runner.log"
    "./.venv/bin/python" "${ROOT_DIR}/train_segformer_semantic.py" \
      --dataset-root "${DATASET_ROOT}" \
      --gsd "${GSD}" \
      --split-root-name "${SPLIT_ROOT_NAME}" \
      --train-split-name "${train_split}" \
      --run-suffix "${run_suffix}" \
      --annotation-file "${ann_file}" \
      --output-root "${TRAIN_OUTPUT_ROOT}" \
      --epochs "${EPOCHS}" \
      --batch-size 8 \
      --grad-accum-steps 2 \
      --loss focal \
      --focal-gamma 2.0 \
      --tensorboard \
      --no-use-class-weights \
      2>&1 | tee "${train_log_file}"

    train_exp_name="${GSD}__${ann_file%.json}__${run_suffix}"
    checkpoint_dir="${TRAIN_OUTPUT_ROOT}/${train_exp_name}/best"
    if [[ ! -d "${checkpoint_dir}" ]]; then
      echo "[$(date -Is)] Missing best checkpoint for ${run_name}: ${checkpoint_dir}" | tee -a "${LOG_DIR}/runner.log"
      exit 1
    fi

    test_exp_name="${GSD}__${ann_file%.json}__${run_suffix}"
    test_run_output="${TEST_EVAL_OUTPUT_ROOT}/${run_name}"
    mkdir -p "${test_run_output}"

    echo "[$(date -Is)] Starting test eval ${run_name}" | tee -a "${LOG_DIR}/runner.log"
    "./.venv/bin/python" "${ROOT_DIR}/train_segformer_semantic.py" \
      --dataset-root "${DATASET_ROOT}" \
      --gsd "${GSD}" \
      --split-root-name "${SPLIT_ROOT_NAME}" \
      --train-split-name "${train_split}" \
      --run-suffix "${run_suffix}" \
      --annotation-file "${ann_file}" \
      --output-root "${TEST_EVAL_OUTPUT_ROOT}" \
      --eval-only \
      --eval-split test \
      --checkpoint "${checkpoint_dir}" \
      --loss focal \
      --focal-gamma 2.0 \
      --tensorboard \
      --no-use-class-weights \
      2>&1 | tee "${test_log_file}"

    cp "${TEST_EVAL_OUTPUT_ROOT}/${test_exp_name}/eval_metrics.json" "${test_run_output}/eval_metrics.json"
    cp "${TEST_EVAL_OUTPUT_ROOT}/${test_exp_name}/data_config.json" "${test_run_output}/data_config.json"
    cp "${TEST_EVAL_OUTPUT_ROOT}/${test_exp_name}/confusion_matrix.csv" "${test_run_output}/confusion_matrix.csv" || true
    cp "${TEST_EVAL_OUTPUT_ROOT}/${test_exp_name}/confusion_matrix_full.csv" "${test_run_output}/confusion_matrix_full.csv" || true
    cp "${TEST_EVAL_OUTPUT_ROOT}/${test_exp_name}/confusion_matrix_fg.csv" "${test_run_output}/confusion_matrix_fg.csv" || true

    echo "[$(date -Is)] Finished train+test ${run_name}" | tee -a "${LOG_DIR}/runner.log"
  done
done

echo "[$(date -Is)] All learning-curve experiments finished for ${GSD}" | tee -a "${LOG_DIR}/runner.log"
