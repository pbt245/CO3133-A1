# LSTM vs GRU - cifar10

Seeds LSTM: 42 123 2026 | GRU: 42 123 2026. Only the recurrent cell differs.

| Metric | LSTM | GRU | GRU − LSTM |
|---|---|---|---|
| Trainable parameters | 891402 ± 0 | 669194 ± 0 | -222208 (-24.9%) |
| Test accuracy (%) | 68.86 ± 0.21 | 71.28 ± 0.13 | +2.42 (+3.5%) |
| Test macro-F1 (%) | 68.80 ± 0.29 | 71.19 ± 0.17 | +2.39 (+3.5%) |
| Test loss | 1.1293 ± 0.0495 | 0.9990 ± 0.0063 | -0.1302 (-11.5%) |
| Best epoch | 48.0 ± 7.2 | 52.0 ± 4.6 | +4.0 (+8.3%) |
| Train time per epoch (s) | 1.71 ± 0.03 | 1.53 ± 0.03 | -0.18 (-10.3%) |
| Total train time (s) | 100.7 ± 4.7 | 91.9 ± 1.6 | -8.8 (-8.8%) |
| Throughput (img/s) | 93289 ± 2322 | 122286 ± 4935 | +28997 (+31.1%) |
| Latency bs=1 (ms) | 0.496 ± 0.014 | 0.448 ± 0.011 | -0.048 (-9.6%) |

## Per-class F1 (%)

| Class | LSTM | GRU | GRU − LSTM |
|---|---|---|---|
| airplane | 74.48 | 75.85 | +1.37 |
| automobile | 81.39 | 81.77 | +0.38 |
| bird | 58.04 | 61.37 | +3.33 |
| cat | 49.40 | 53.00 | +3.60 |
| deer | 61.35 | 65.18 | +3.82 |
| dog | 57.53 | 59.93 | +2.39 |
| frog | 72.96 | 76.51 | +3.55 |
| horse | 73.28 | 76.31 | +3.02 |
| ship | 83.12 | 84.19 | +1.07 |
| truck | 76.44 | 77.75 | +1.32 |

McNemar (chi2 with continuity correction, seed 42): lstm right & gru wrong = 856, opposite = 1116, p = 5.464e-09
