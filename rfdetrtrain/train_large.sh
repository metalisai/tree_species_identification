source venv/bin/activate

set -e

BATCH_SIZE=2
ACCUM_STEPS=8

train_ds() {
	DATASET_DIR=../data/compiled_datasets_3_fold/${1}/dataset
	echo "Training with fine classes"
	cp "${DATASET_DIR}/train/_annotations.fine.coco.json" "${DATASET_DIR}/train/_annotations.coco.json"
	cp "${DATASET_DIR}/valid/_annotations.fine.coco.json" "${DATASET_DIR}/valid/_annotations.coco.json"
	python train_param.py --dataset_dir $DATASET_DIR --epochs $2 --batch_size $BATCH_SIZE --grad_accum_steps $ACCUM_STEPS --output_dir "train_output_${1}_fine_2"

	echo "Training with medium classes"
	cp "${DATASET_DIR}/train/_annotations.medium.coco.json" "${DATASET_DIR}/train/_annotations.coco.json"
	cp "${DATASET_DIR}/valid/_annotations.medium.coco.json" "${DATASET_DIR}/valid/_annotations.coco.json"
	python train_param.py --dataset_dir $DATASET_DIR --epochs $2 --batch_size $BATCH_SIZE --grad_accum_steps $ACCUM_STEPS --output_dir "train_output_${1}_medium_2"

	echo "Training with coarse classes"
	cp "${DATASET_DIR}/train/_annotations.coarse.coco.json" "${DATASET_DIR}/train/_annotations.coco.json"
	cp "${DATASET_DIR}/valid/_annotations.coarse.coco.json" "${DATASET_DIR}/valid/_annotations.coco.json"
	python train_param.py --dataset_dir $DATASET_DIR --epochs $2 --batch_size $BATCH_SIZE --grad_accum_steps $ACCUM_STEPS --output_dir "train_output_${1}_coarse_2"
}

# NOTE: batch size and accumulation steps always multiply to 16
# which means it is equivalent
# need to use smaller batch size for lower resolution, because there are
# more objects, which affects VRAM usage

BATCH_SIZE=8
ACCUM_STEPS=2
train_ds gsd_0p02 60
BATCH_SIZE=4
ACCUM_STEPS=4
train_ds gsd_0p05 80
BATCH_SIZE=2
ACCUM_STEPS=8
train_ds gsd_0p1  120
train_ds gsd_0p25 200
