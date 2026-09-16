"""Linear / softmax classifier.

(B, C, H, W) -> flatten -> (B, C*H*W) -> Linear -> logits (B, K)

The model returns RAW LOGITS. Softmax is NOT applied here: nn.CrossEntropyLoss
applies log-softmax internally, so applying softmax first would be wrong.
Probabilities are computed only at evaluation time for confidence/visualization.
"""
from __future__ import annotations

import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, in_channels: int, image_size: int, num_classes: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_channels * image_size * image_size, num_classes)

    def forward(self, x):
        return self.fc(self.flatten(x))
