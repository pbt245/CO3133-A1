"""Multilayer perceptron on flattened pixels.

Each hidden layer: Linear -> (BatchNorm1d) -> activation -> (Dropout)
Regularization: dropout, batch normalization, weight decay (AdamW, set in config),
and data augmentation (dataset config).
"""
from __future__ import annotations

import torch.nn as nn

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


class MLPClassifier(nn.Module):
    def __init__(self, in_channels: int, image_size: int, num_classes: int,
                 hidden_dims=(512, 256), activation: str = "relu",
                 dropout: float = 0.3, batch_norm: bool = True):
        super().__init__()
        if len(hidden_dims) < 1:
            raise ValueError("MLP needs at least one hidden layer")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(ACTIVATIONS)}")
        layers = [nn.Flatten()]
        prev = in_channels * image_size * image_size
        for h in hidden_dims:
            layers.append(nn.Linear(prev, int(h), bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm1d(int(h)))
            layers.append(ACTIVATIONS[activation]())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = int(h)
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
