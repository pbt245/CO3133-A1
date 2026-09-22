# LSTM vs GRU - fashion_mnist

Seeds LSTM: 42 123 2026 | GRU: 42 123 2026. Only the recurrent cell differs.

| Metric | LSTM | GRU | GRU − LSTM |
|---|---|---|---|
| Trainable parameters | 214282 ± 0 | 161034 ± 0 | -53248 (-24.8%) |
| Test accuracy (%) | 90.76 ± 0.13 | 90.84 ± 0.21 | +0.08 (+0.1%) |
| Test macro-F1 (%) | 90.74 ± 0.12 | 90.82 ± 0.21 | +0.08 (+0.1%) |
| Test loss | 0.2581 ± 0.0031 | 0.2547 ± 0.0032 | -0.0034 (-1.3%) |
| Best epoch | 35.0 ± 3.6 | 38.7 ± 1.2 | +3.7 (+10.5%) |
| Train time per epoch (s) | 1.19 ± 0.07 | 1.18 ± 0.04 | -0.01 (-0.6%) |
| Total train time (s) | 47.6 ± 2.9 | 47.4 ± 1.7 | -0.3 (-0.6%) |
| Throughput (img/s) | 126022 ± 6210 | 206644 ± 9086 | +80622 (+64.0%) |
| Latency bs=1 (ms) | 0.249 ± 0.018 | 0.169 ± 0.012 | -0.081 (-32.3%) |

## Per-class F1 (%)

| Class | LSTM | GRU | GRU − LSTM |
|---|---|---|---|
| T-shirt/top | 84.52 | 85.31 | +0.79 |
| Trouser | 98.66 | 98.54 | -0.11 |
| Pullover | 85.44 | 84.46 | -0.97 |
| Dress | 90.91 | 91.48 | +0.57 |
| Coat | 85.05 | 84.66 | -0.39 |
| Sandal | 97.49 | 97.62 | +0.13 |
| Shirt | 74.10 | 75.14 | +1.05 |
| Sneaker | 96.18 | 96.06 | -0.12 |
| Bag | 98.29 | 98.21 | -0.08 |
| Ankle boot | 96.78 | 96.73 | -0.05 |

McNemar (chi2 with continuity correction, seed 42): lstm right & gru wrong = 234, opposite = 248, p = 0.5538
