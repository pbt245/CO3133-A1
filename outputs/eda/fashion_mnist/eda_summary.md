# EDA - fashion_mnist

Split file: `fashion_mnist_split42_val0.1.json` (split_seed=42, val_fraction=0.1, stratified)

## Split sizes and class distribution

| class | train (n, %) | val (n, %) | test (n, %) |
|---|---|---|---|
| T-shirt/top | 5400 (10.00%) | 600 (10.00%) | 1000 (10.00%) |
| Trouser | 5400 (10.00%) | 600 (10.00%) | 1000 (10.00%) |
| Pullover | 5400 (10.00%) | 600 (10.00%) | 1000 (10.00%) |
| Dress | 5400 (10.00%) | 600 (10.00%) | 1000 (10.00%) |
| Coat | 5400 (10.00%) | 600 (10.00%) | 1000 (10.00%) |
| Sandal | 5400 (10.00%) | 600 (10.00%) | 1000 (10.00%) |
| Shirt | 5400 (10.00%) | 600 (10.00%) | 1000 (10.00%) |
| Sneaker | 5400 (10.00%) | 600 (10.00%) | 1000 (10.00%) |
| Bag | 5400 (10.00%) | 600 (10.00%) | 1000 (10.00%) |
| Ankle boot | 5400 (10.00%) | 600 (10.00%) | 1000 (10.00%) |

## Imbalance

| split | min class (n) | max class (n) | max/min | CV | normalized entropy |
|---|---|---|---|---|---|
| train | T-shirt/top (5400) | T-shirt/top (5400) | 1.000 | 0.0000 | 1.0000 |
| val | T-shirt/top (600) | T-shirt/top (600) | 1.000 | 0.0000 | 1.0000 |
| test | T-shirt/top (1000) | T-shirt/top (1000) | 1.000 | 0.0000 | 1.0000 |

Stratification check: {'max_abs_class_pct_diff_train_vs_val': 0.0, 'max_abs_class_pct_diff_train_vs_test': 0.0}

## Input

```
{'raw_shape_per_image': [1, 28, 28], 'raw_dtype': 'uint8', 'raw_value_range_train': [0, 255], 'flattened_dim': 784, 'sequence_views': {'rows': [28, 28], 'cols': [28, 28], 'patches_4x4': [49, 16]}, 'split': {'split_seed': 42, 'val_fraction': 0.1, 'stratified': True, 'protocol': 'official train set -> stratified train/val; official test set -> test', 'fingerprints': {'train': '2378c8b39f1d0606', 'val': 'c3422ec2263f2769'}}, 'normalization_train_only': {'mean': [0.2862090077697746], 'std': [0.3531597432932429], 'computed_on': 'train split only, pixels scaled to [0,1]'}}
```

## Most similar class mean images

| class A | class B | corr |
|---|---|---|
| Coat | Shirt | 0.967 |
| Pullover | Coat | 0.958 |
| Pullover | Shirt | 0.947 |
| Trouser | Dress | 0.879 |
| Sandal | Sneaker | 0.876 |

## Duplicates / leakage

```
{'official_train': {'n_images': 60000, 'n_unique': 60000, 'duplicate_groups': 0, 'images_in_duplicate_groups': 0, 'label_conflicting_groups': 0}, 'official_test': {'n_images': 10000, 'n_unique': 10000, 'duplicate_groups': 0, 'images_in_duplicate_groups': 0, 'label_conflicting_groups': 0}, 'val_images_identical_to_a_train_image': 0, 'test_images_identical_to_a_train_image': 0}
```

## Preprocessed batch

```
{'x_shape': [128, 1, 28, 28], 'x_dtype': 'float32', 'y_shape': [128], 'y_dtype': 'int64', 'x_min': -0.8104236721992493, 'x_max': 2.021156072616577, 'x_mean': 0.009468985721468925, 'x_std': 1.015068531036377, 'augmentation_train': {'random_crop_padding': 2, 'hflip': True}, 'sequence_shapes': {'rows': [128, 28, 28], 'cols': [128, 28, 28], 'patches_4x4': [128, 49, 16]}}
```
