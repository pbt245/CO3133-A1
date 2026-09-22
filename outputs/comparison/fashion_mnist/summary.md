# Comparison - fashion_mnist

Mean ± sample std over training seeds (same data split for every run). Checkpoint = best validation macro-F1. Timing in FP32 on the device listed in runs.csv.

## Main models

| Experiment | Seeds | Params | Test acc (%) | Test macro-F1 (%) | Best epoch | Train s/epoch | Train time (s) | Throughput (img/s) | Latency bs=1 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| linear | 42 123 2026 | 7,850 | 76.00 ± 0.12 | 75.76 ± 0.18 | 23.7 ± 1.5 | 1.0 ± 0.0 | 32 ± 2 | 15972514 ± 28386 | 0.02 ± 0.00 |
| mlp | 42 123 2026 | 536,586 | 88.85 ± 0.14 | 88.81 ± 0.12 | 38.3 ± 1.5 | 1.0 ± 0.0 | 39 ± 1 | 2433204 ± 87240 | 0.09 ± 0.01 |
| cnn | 42 123 2026 | 288,170 | 94.16 ± 0.13 | 94.15 ± 0.14 | 37.3 ± 1.5 | 2.4 ± 0.1 | 94 ± 3 | 72057 ± 4167 | 0.33 ± 0.02 |
| lstm | 42 123 2026 | 214,282 | 90.76 ± 0.13 | 90.74 ± 0.12 | 35.0 ± 3.6 | 1.2 ± 0.1 | 48 ± 3 | 126022 ± 6210 | 0.25 ± 0.02 |
| gru | 42 123 2026 | 161,034 | 90.84 ± 0.21 | 90.82 ± 0.21 | 38.7 ± 1.2 | 1.2 ± 0.0 | 47 ± 2 | 206644 ± 9086 | 0.17 ± 0.01 |
| transformer | 42 123 2026 | 306,826 | 88.03 ± 0.47 | 87.97 ± 0.47 | 37.3 ± 0.6 | 3.3 ± 0.1 | 131 ± 4 | 37771 ± 2453 | 0.77 ± 0.10 |

## Ablations

| Experiment | Seeds | Params | Test acc (%) | Test macro-F1 (%) | Best epoch | Train s/epoch | Train time (s) | Throughput (img/s) | Latency bs=1 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| cnn_no_aug | 42 123 2026 | 288,170 | 93.88 ± 0.07 | 93.88 ± 0.07 | 32.0 ± 6.9 | 2.4 ± 0.0 | 93 ± 8 | 66662 ± 3023 | 0.34 ± 0.03 |
| gru_patches | 42 123 2026 | 156,426 | 89.69 ± 0.16 | 89.69 ± 0.16 | 37.7 ± 3.2 | 1.4 ± 0.2 | 58 ± 9 | 130865 ± 3338 | 0.24 ± 0.02 |
| mlp_no_aug | 42 123 2026 | 536,586 | 90.83 ± 0.27 | 90.81 ± 0.28 | 36.0 ± 5.3 | 0.9 ± 0.0 | 37 ± 1 | 2405012 ± 131849 | 0.10 ± 0.02 |
| transformer_no_pos | 42 123 2026 | 302,026 | 82.37 ± 0.84 | 82.31 ± 0.81 | 37.3 ± 2.9 | 3.5 ± 0.0 | 139 ± 1 | 34277 ± 399 | 0.85 ± 0.08 |
| transformer_rows | 42 123 2026 | 305,962 | 90.74 ± 0.27 | 90.72 ± 0.25 | 36.3 ± 3.2 | 3.0 ± 0.1 | 121 ± 3 | 51524 ± 1389 | 0.85 ± 0.02 |

## Traceability

Commits used: ['4255cd22b2646ade4032fc561812bbcb1670bf90']
Runs with uncommitted changes: 33
Train split fingerprints: ['2378c8b39f1d0606']
