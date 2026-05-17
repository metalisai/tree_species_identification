# SegFormer Training

This folder contains a training scripts used in the thesis.

## 1) Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Check that installation works

```bash
python train_segformer_semantic.py --help
```

If this prints the CLI help, dependencies are installed correctly.

## 3) Dataset layout

By default, the script expects:

- dataset root: `../compiled_datasets_3_fold`
- GSD subfolders under that root, e.g. `gsd_0p02`, `gsd_0p05`, `gsd_0p1`
- split folders inside each GSD folder (for example `dataset/train`, `dataset/valid`, `dataset/test`)
- COCO annotation files inside each split, e.g. `_annotations.fine.coco.json`

The dataset is set up by the `../data/build.sh` script.
If your data lives elsewhere, pass `--dataset-root` and other split arguments explicitly.

## 4) Run training

Minimal example:

```bash
python train_segformer_semantic.py \
  --dataset-root ../data/compiled_datasets_3_fold \
  --gsd gsd_0p05 \
  --annotation-file _annotations.fine.coco.json \
  --output-root runs
```

### Helper scripts

- `train_baseline.sh`: 3 GSDs x 3 fineness levels, no augmentation
- `train_baseline_with_aug.sh`: same grid with augmentation
- `train_learning_curve.sh`: learning-curve runs for `gsd_0p05`

Run one with:

```bash
bash train_baseline.sh
```

## Notes

- `requirements.txt` pins `albumentations==1.4.24`, which is required by the training script.
- `torch==2.6.0` and `torchvision==0.21.0` will install CUDA-enabled wheels on Linux when available.
