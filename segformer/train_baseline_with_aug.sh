#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="."
DATASET_ROOT="compiled_datasets_3_fold"
TRAIN_OUTPUT_ROOT="${ROOT_DIR}/runs_baseline_with_aug"
TEST_EVAL_OUTPUT_ROOT="${ROOT_DIR}/runs_baseline_with_aug_testeval"
LOG_DIR="${ROOT_DIR}/logs_baseline_with_aug"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "${TRAIN_OUTPUT_ROOT}" "${TEST_EVAL_OUTPUT_ROOT}" "${LOG_DIR}"

GSDS=("gsd_0p02" "gsd_0p05" "gsd_0p1")
FINENESS=("fine" "medium" "coarse")

epochs_for_gsd() {
  case "$1" in
    gsd_0p02) echo 40 ;;
    gsd_0p05) echo 60 ;;
    gsd_0p1) echo 90 ;;
    *) echo 40 ;;
  esac
}

for gsd in "${GSDS[@]}"; do
  for fineness in "${FINENESS[@]}"; do
    ann_file="_annotations.${fineness}.coco.json"
    run_name="${gsd}___${ann_file%.json}"
    train_log_file="${LOG_DIR}/${run_name}.train.log"
    test_log_file="${LOG_DIR}/${run_name}.testeval.log"
    epochs="$(epochs_for_gsd "${gsd}")"

    echo "[$(date -Is)] Starting train ${run_name} epochs=${epochs}" | tee -a "${LOG_DIR}/runner.log"
    "./.venv/bin/python" "${ROOT_DIR}/train_segformer_semantic.py" \
      --dataset-root "${DATASET_ROOT}" \
      --gsd "${gsd}" \
      --annotation-file "${ann_file}" \
      --output-root "${TRAIN_OUTPUT_ROOT}" \
      --epochs "${epochs}" \
      --batch-size 8 \
      --grad-accum-steps 2 \
      --loss focal \
      --focal-gamma 2.0 \
      --tensorboard \
      --no-use-class-weights \
      2>&1 | tee "${train_log_file}"

    checkpoint_dir="${TRAIN_OUTPUT_ROOT}/${gsd}__${ann_file%.json}/best"
    if [[ ! -d "${checkpoint_dir}" ]]; then
      echo "[$(date -Is)] Missing best checkpoint for ${run_name}: ${checkpoint_dir}" | tee -a "${LOG_DIR}/runner.log"
      exit 1
    fi

    echo "[$(date -Is)] Starting test eval ${run_name}" | tee -a "${LOG_DIR}/runner.log"
    "./.venv/bin/python" "${ROOT_DIR}/train_segformer_semantic.py" \
      --dataset-root "${DATASET_ROOT}" \
      --gsd "${gsd}" \
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

    echo "[$(date -Is)] Finished train+test ${run_name}" | tee -a "${LOG_DIR}/runner.log"
  done
done

echo "[$(date -Is)] All 9 focal-gamma2 3_fold experiments finished with test eval" | tee -a "${LOG_DIR}/runner.log"
