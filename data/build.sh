# generate base datasets with different GSDs
python build_datasets.py --gsd 0.05 0.02 0.1 --output-root compiled_datasets_3_fold --work-root . --tile-size 512 --tile-overlap 0.1 --min-intersection-ratio 0.05 --ignore-tiles-without-labels
# generate learning curve subsets for 5 cm GSD dataset
python generate_learning_curve_subsets.py --dataset-root compiled_datasets_3_fold/gsd_0p05/dataset
