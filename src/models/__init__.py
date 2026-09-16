"""Model registry: build any of the five mandatory models from a config section."""
from __future__ import annotations

import copy

import torch.nn as nn

from src.models.cnn import SimpleCNN
from src.models.linear import LinearClassifier
from src.models.mlp import MLPClassifier
from src.models.rnn import RecurrentClassifier
from src.models.transformer import SequenceTransformer

MODEL_NAMES = ("linear", "mlp", "cnn", "lstm", "gru", "transformer")


def build_model(model_cfg: dict, in_channels: int, image_size: int, num_classes: int) -> nn.Module:
    cfg = copy.deepcopy(model_cfg)
    name = str(cfg.pop("name")).lower()
    if name == "linear":
        return LinearClassifier(in_channels=in_channels, image_size=image_size, num_classes=num_classes, **cfg)
    if name == "mlp":
        return MLPClassifier(in_channels=in_channels, image_size=image_size, num_classes=num_classes, **cfg)
    if name == "cnn":
        return SimpleCNN(in_channels=in_channels, num_classes=num_classes, **cfg)
    if name in ("lstm", "gru"):
        return RecurrentClassifier(cell=name, in_channels=in_channels, image_size=image_size,
                                   num_classes=num_classes, **cfg)
    if name == "transformer":
        return SequenceTransformer(in_channels=in_channels, image_size=image_size, num_classes=num_classes, **cfg)
    raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}")


def count_parameters(model: nn.Module) -> dict:
    return {
        "trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "total": int(sum(p.numel() for p in model.parameters())),
    }
