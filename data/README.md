# Dataset build setup

This directory contains scripts for building tiled COCO datasets and learning-curve subsets.

## Prerequisites

- Python 3.10+ (with `venv` support)
- Input data files in this directory (for example `rgb_z1.tif`, `rgb_z2.tif`, `rgb_z3.tif`, and `Z1_polygons.gpkg`)

## Setup from scratch

Run all commands from `data/`.

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Upgrade pip:

   ```bash
   python -m pip install --upgrade pip
   ```

3. Install dependencies (this installs `geodataset` from https://github.com/hugobaudchon/geodataset):

   ```bash
   python -m pip install -r requirements.txt
   ```

## Build datasets

Run:

```bash
bash build.sh
```

`build.sh` does two things:

- builds base datasets for multiple GSD values into `compiled_datasets_3_fold/`
- builds learning-curve subsets from the 0.05 GSD dataset

The `classes_*.txt` files define different class aggregations, in the thesis fine, medium and coarse classes were used.
