# EDA - cifar10

Split file: `cifar10_split42_val0.1.json` (split_seed=42, val_fraction=0.1, stratified)

## Split sizes and class distribution

| class | train (n, %) | val (n, %) | test (n, %) |
|---|---|---|---|
| airplane | 4500 (10.00%) | 500 (10.00%) | 1000 (10.00%) |
| automobile | 4500 (10.00%) | 500 (10.00%) | 1000 (10.00%) |
| bird | 4500 (10.00%) | 500 (10.00%) | 1000 (10.00%) |
| cat | 4500 (10.00%) | 500 (10.00%) | 1000 (10.00%) |
| deer | 4500 (10.00%) | 500 (10.00%) | 1000 (10.00%) |
| dog | 4500 (10.00%) | 500 (10.00%) | 1000 (10.00%) |
| frog | 4500 (10.00%) | 500 (10.00%) | 1000 (10.00%) |
| horse | 4500 (10.00%) | 500 (10.00%) | 1000 (10.00%) |
| ship | 4500 (10.00%) | 500 (10.00%) | 1000 (10.00%) |
| truck | 4500 (10.00%) | 500 (10.00%) | 1000 (10.00%) |

## Imbalance

| split | min class (n) | max class (n) | max/min | CV | normalized entropy |
|---|---|---|---|---|---|
| train | airplane (4500) | airplane (4500) | 1.000 | 0.0000 | 1.0000 |
| val | airplane (500) | airplane (500) | 1.000 | 0.0000 | 1.0000 |
| test | airplane (1000) | airplane (1000) | 1.000 | 0.0000 | 1.0000 |

Stratification check: {'max_abs_class_pct_diff_train_vs_val': 0.0, 'max_abs_class_pct_diff_train_vs_test': 0.0}

## Input

```
{'raw_shape_per_image': [3, 32, 32], 'raw_dtype': 'uint8', 'raw_value_range_train': [0, 255], 'flattened_dim': 3072, 'sequence_views': {'rows': [32, 96], 'cols': [32, 96], 'patches_4x4': [64, 48]}, 'split': {'split_seed': 42, 'val_fraction': 0.1, 'stratified': True, 'protocol': 'official train set -> stratified train/val; official test set -> test', 'fingerprints': {'train': 'b99301bfd9655f1c', 'val': '280540fcdfde8a41'}}, 'normalization_train_only': {'mean': [0.49113735123910696, 0.4821111870234209, 0.4464489876089328], 'std': [0.24686913619557854, 0.24343012612627304, 0.2616033939151692], 'computed_on': 'train split only, pixels scaled to [0,1]'}}
```

## Most similar class mean images

| class A | class B | corr |
|---|---|---|
| deer | frog | 0.884 |
| automobile | truck | 0.884 |
| cat | frog | 0.882 |
| bird | horse | 0.850 |
| cat | deer | 0.811 |

## Duplicates / leakage

```
{'official_train': {'n_images': 50000, 'n_unique': 50000, 'duplicate_groups': 0, 'images_in_duplicate_groups': 0, 'label_conflicting_groups': 0}, 'official_test': {'n_images': 10000, 'n_unique': 10000, 'duplicate_groups': 0, 'images_in_duplicate_groups': 0, 'label_conflicting_groups': 0}, 'val_images_identical_to_a_train_image': 0, 'test_images_identical_to_a_train_image': 0}
```

## Preprocessed batch

```
{'x_shape': [128, 3, 32, 32], 'x_dtype': 'float32', 'y_shape': [128], 'y_dtype': 'int64', 'x_min': -1.9894644021987915, 'x_max': 2.1274638175964355, 'x_mean': -0.23807458579540253, 'x_std': 1.1311041116714478, 'augmentation_train': {'random_crop_padding': 4, 'hflip': True}, 'sequence_shapes': {'rows': [128, 32, 96], 'cols': [128, 32, 96], 'patches_4x4': [128, 64, 48]}}
```
