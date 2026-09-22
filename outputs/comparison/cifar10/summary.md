# Comparison - cifar10

Mean ± sample std over training seeds (same data split for every run). Checkpoint = best validation macro-F1. Timing in FP32 on the device listed in runs.csv.

## Main models

| Experiment | Seeds | Params | Test acc (%) | Test macro-F1 (%) | Best epoch | Train s/epoch | Train time (s) | Throughput (img/s) | Latency bs=1 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| linear | 42 123 2026 | 30,730 | 38.83 ± 0.34 | 37.82 ± 0.39 | 57.3 ± 2.5 | 1.0 ± 0.0 | 59 ± 1 | 9469708 ± 913741 | 0.03 ± 0.00 |
| mlp | 42 123 2026 | 3,678,218 | 61.46 ± 0.15 | 61.10 ± 0.16 | 53.3 ± 2.3 | 1.1 ± 0.0 | 65 ± 0 | 651483 ± 45689 | 0.10 ± 0.01 |
| cnn | 42 123 2026 | 1,148,874 | 91.35 ± 0.32 | 91.33 ± 0.32 | 53.7 ± 7.6 | 5.0 ± 0.1 | 301 ± 6 | 19300 ± 139 | 0.36 ± 0.01 |
| lstm | 42 123 2026 | 891,402 | 68.86 ± 0.21 | 68.80 ± 0.29 | 48.0 ± 7.2 | 1.7 ± 0.0 | 101 ± 5 | 93289 ± 2322 | 0.50 ± 0.01 |
| gru | 42 123 2026 | 669,194 | 71.28 ± 0.13 | 71.19 ± 0.17 | 52.0 ± 4.6 | 1.5 ± 0.0 | 92 ± 2 | 122286 ± 4935 | 0.45 ± 0.01 |
| transformer | 42 123 2026 | 811,146 | 79.70 ± 0.34 | 79.56 ± 0.37 | 57.3 ± 2.9 | 6.3 ± 0.2 | 378 ± 10 | 16926 ± 247 | 1.15 ± 0.04 |

## Traceability

Commits used: ['4255cd22b2646ade4032fc561812bbcb1670bf90']
Runs with uncommitted changes: 18
Train split fingerprints: ['b99301bfd9655f1c']
