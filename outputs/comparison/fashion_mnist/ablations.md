# Ablations - fashion_mnist

| Ablation | Reference | Changed factor | Ref macro-F1 (%) | Abl macro-F1 (%) | Δ macro-F1 (pp) | Δ acc (pp) | Params | Common seeds |
|---|---|---|---|---|---|---|---|---|
| cnn_no_aug | cnn | augmentation (random crop pad=2 + horizontal flip) -> none | 94.15 ± 0.14 | 93.88 ± 0.07 | -0.27 | -0.28 | 288,170 → 288,170 | 42 123 2026 |
| gru_patches | gru | timestep definition: rows (28 steps x 28 dims) -> 4x4 patches in raster order (49 steps x 16 dims) | 90.82 ± 0.21 | 89.69 ± 0.16 | -1.14 | -1.15 | 161,034 → 156,426 | 42 123 2026 |
| mlp_no_aug | mlp | augmentation (random crop pad=2 + horizontal flip) -> none | 88.81 ± 0.12 | 90.81 ± 0.28 | +2.00 | +1.97 | 536,586 → 536,586 | 42 123 2026 |
| transformer_no_pos | transformer | positional encoding: learned -> none | 87.97 ± 0.47 | 82.31 ± 0.81 | -5.66 | -5.66 | 306,826 → 302,026 | 42 123 2026 |
| transformer_rows | transformer | token representation: 4x4 patches (49 tokens) -> image rows (28 tokens of dim 28) | 87.97 ± 0.47 | 90.72 ± 0.25 | +2.75 | +2.71 | 306,826 → 305,962 | 42 123 2026 |

## Pre-registered hypotheses

- **cnn_no_aug**: Convolution + pooling already give approximate translation invariance, so removing augmentation should cause only a small drop in test macro-F1 but a larger train-val gap.
- **gru_patches**: Patch sequences are longer and each step carries less context, and vertically adjacent patches are 7 steps apart, so the GRU is expected to perform no better than with rows.
- **mlp_no_aug**: An MLP on flattened pixels has no built-in translation invariance, so augmentation acts as a stronger regularizer for it than for the CNN; removing it should widen the train-val gap more than for cnn_no_aug.
- **transformer_no_pos**: Self-attention is permutation-equivariant; without positional encoding the CLS output only sees an unordered set of patches, so macro-F1 is expected to drop.
- **transformer_rows**: Row tokens mix pixels from the whole width and lose 2D locality inside a token, so macro-F1 is expected to be lower than with patch tokens.
