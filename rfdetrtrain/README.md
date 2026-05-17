# RF-DETR Training Experiments

This repository is organized around experiment shell scripts. Use the `.sh` files to run training configurations instead of calling `train_param.py` directly.

## Prerequisites

- Python 3.10+ (3.12 works)
- A local checkout of RF-DETR at `../rf-detr`
- Datasets at paths expected by the scripts under `../data/...`

## Setup

1. Create a virtual environment named `venv` (the scripts activate `venv/bin/activate`):

```bash
python3 -m venv venv
```

2. Activate it:

```bash
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Experiments

Run scripts from the repository root:

```bash
bash train_large.sh
```

### Available experiment scripts

- `train_large.sh`: baseline training runs across `gsd_0p02`, `gsd_0p05`, `gsd_0p1`, `gsd_0p25` with fine/medium/coarse label sets.
- `train_large_aud.sh`: same grid as `train_large.sh`, with augmentation enabled (`--aug`).
- `train_learning_curve.sh`: learning-curve experiments for `gsd_0p05` across train splits (`train_5` ... `train_100`) with fine/medium/coarse label sets and augmentation.

## Notes

- `train_param.py` is used internally by the scripts, but you can also call it directly for custom runs. The scripts are just convenient wrappers for reproducible experiments.
