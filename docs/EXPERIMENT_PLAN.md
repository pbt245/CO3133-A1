# Experiment plan - Assignment 1 (CO3133, Semester 261)

Written **before** running experiments. Hypotheses are predictions, not results; the report
must state whether each one was supported by the measured numbers.

## Datasets

| Role | Dataset | Split used |
|---|---|---|
| Debugging only | MNIST | 54,000 / 6,000 / 10,000 (same protocol) |
| Main comparison | Fashion-MNIST | 54,000 train / 6,000 val (stratified from official train, `split_seed=42`) / 10,000 official test |
| Extension | CIFAR-10 | 45,000 train / 5,000 val (stratified, `split_seed=42`) / 10,000 official test |

## Shared protocol (fairness constraints, handbook Sec. 12.2)

- Same split file (indices + fingerprint) for every model of a dataset; normalization statistics from the train split only.
- Same augmentation per dataset, same batch size (128), same epoch budget and early-stopping patience, same optimizer family (AdamW), cosine schedule with warm-up, gradient clipping 1.0.
- Checkpoint rule: best validation macro-F1; the test set is evaluated once, on that checkpoint.
- Metrics: accuracy, macro-F1 (+ bootstrap 95% CI), per-class P/R/F1, confusion matrix, trainable parameters, training time, inference throughput and batch-1 latency (FP32).
- Seeds 42, 123, 2026 on Fashion-MNIST (mean ± std). CIFAR-10: at least seed 42, three seeds if compute allows.
- Model-specific hyperparameters (lr, weight decay, width) are defaults chosen before running; any tuning must use the validation split only and be reported.

## Decision criterion (used by every experiment)

"A is better than B" is claimed only when (1) mean macro-F1 differs by more than the larger of the two seed standard deviations, and (2) McNemar test on the shared seed gives p < 0.05. Otherwise the result is reported as "no conclusive difference". Parameter count, training time and latency are reported alongside, so trade-offs are discussed, not only the highest score.

## E1 - Main comparison (Fashion-MNIST): linear, MLP, CNN, LSTM, GRU, Transformer

- **Hypothesis**: CNN reaches the highest macro-F1 because locality, weight sharing and pooling match 2D image structure. Linear is lowest (no hidden features). MLP improves on linear but ignores spatial structure. Row-sequence RNNs and the patch Transformer land between MLP and CNN; without pretraining the Transformer does not beat the CNN at this data scale.
- **Changed factor**: architecture family (and its input representation: flattened pixels / 2D grid / row sequence / patch tokens).
- **Fixed factors**: everything in the shared protocol.
- **Expected error pattern**: most confusions among upper-body garments (Shirt, T-shirt/top, Pullover, Coat).

## E2 - Extension: LSTM vs GRU

- **Hypothesis**: GRU matches LSTM macro-F1 within seed variation while using about 25% fewer recurrent parameters (3 gates instead of 4) and training faster per epoch.
- **Changed factor**: recurrent cell only (`lstm.yaml` vs `gru.yaml` are identical otherwise).
- **Fixed factors**: sequence mode (rows), hidden size, layers, dropout, head, optimizer, schedule, seeds, split.
- **Outputs**: `outputs/comparison/<dataset>/lstm_vs_gru.md`, curves, McNemar result. Evaluated on both Fashion-MNIST and CIFAR-10.

## E3 - Extension: CIFAR-10

- **Hypothesis**: all models score lower than on Fashion-MNIST; the gap between the CNN and non-spatial models (linear, MLP) widens on natural RGB images with background clutter and intra-class variation; row-RNNs lose more because each timestep is a full 96-dim RGB row.
- **Changed factor**: dataset (and dataset-appropriate widths / epochs, listed in `configs/cifar10/`).
- **Fixed factors**: protocol, model families, checkpoint rule.

## Ablations (Fashion-MNIST)

| ID | Experiment | Reference | Changed factor | Hypothesis |
|---|---|---|---|---|
| A1 | `transformer_rows` | `transformer` | tokens: 4×4 patches → rows | Lower macro-F1: row tokens lose within-token 2D locality |
| A2 | `transformer_no_pos` | `transformer` | positional encoding → none | Lower macro-F1: attention is permutation-equivariant, spatial order is lost |
| A3 | `gru_patches` | `gru` | timesteps: rows → 4×4 patches | No improvement: longer sequence, less context per step |
| A4 | `cnn_no_aug` | `cnn` | augmentation → none | Small test drop, larger train-val gap |
| A5 | `mlp_no_aug` | `mlp` | augmentation → none | Larger effect than A4: MLP has no built-in translation invariance |

Fixed factors for each ablation: every other setting of its reference config, same seeds, same split. Output: `outputs/comparison/fashion_mnist/ablations.md`.
