"""Self-designed CNN (no pretrained weights).

Block k:  Conv3x3(pad=1) -> BN -> ReLU -> Conv3x3(pad=1) -> BN -> ReLU -> MaxPool2x2 -> Dropout2d
  * 3x3 convolutions keep H, W (padding=1) and learn local filters shared over all positions
  * max pooling halves H and W, giving some translation tolerance and a larger receptive field
  * each block outputs a stack of feature maps (C_k, H/2^k, W/2^k)
Head: global average pooling -> Dropout -> Linear -> logits
Fashion-MNIST with channels [32, 64, 128]: 28x28 -> 14x14 -> 7x7 -> 3x3 -> GAP.
"""
from __future__ import annotations

import torch.nn as nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        ]
        if dropout and dropout > 0:
            layers.append(nn.Dropout2d(float(dropout)))
        super().__init__(*layers)


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, channels=(32, 64, 128),
                 block_dropout: float = 0.1, head_dropout: float = 0.3):
        super().__init__()
        blocks = []
        c = in_channels
        for out in channels:
            blocks.append(ConvBlock(c, int(out), block_dropout))
            c = int(out)
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(float(head_dropout)), nn.Linear(c, num_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))

    def feature_maps(self, x):
        """Return the output of every conv block (for visualization)."""
        outs = []
        for block in self.features:
            x = block(x)
            outs.append(x)
        return outs
