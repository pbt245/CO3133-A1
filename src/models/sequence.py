"""Turn an image batch into a sequence of tokens for the LSTM/GRU and Transformer.

x: (B, C, H, W)
  rows    -> (B, H, C*W)                 timestep t = row t (top to bottom)
  cols    -> (B, W, C*H)                 timestep t = column t (left to right)
  patches -> (B, (H/p)*(W/p), C*p*p)     non-overlapping p x p patches in raster order
"""
from __future__ import annotations

import torch

SEQUENCE_MODES = ("rows", "cols", "patches")


def sequence_spec(mode: str, channels: int, height: int, width: int, patch_size: int = 4):
    """Return (sequence_length, token_dim) for a representation."""
    if mode == "rows":
        return height, channels * width
    if mode == "cols":
        return width, channels * height
    if mode == "patches":
        if height % patch_size or width % patch_size:
            raise ValueError(f"Image {height}x{width} not divisible by patch_size={patch_size}")
        return (height // patch_size) * (width // patch_size), channels * patch_size * patch_size
    raise ValueError(f"sequence_mode must be one of {SEQUENCE_MODES}, got '{mode}'")


def patch_grid(mode: str, height: int, width: int, patch_size: int = 4):
    """Spatial layout (grid_h, grid_w) of the tokens, used for attention visualization."""
    if mode == "rows":
        return height, 1
    if mode == "cols":
        return 1, width
    return height // patch_size, width // patch_size


def image_to_sequence(x: torch.Tensor, mode: str, patch_size: int = 4) -> torch.Tensor:
    b, c, h, w = x.shape
    if mode == "rows":
        return x.permute(0, 2, 1, 3).reshape(b, h, c * w)
    if mode == "cols":
        return x.permute(0, 3, 1, 2).reshape(b, w, c * h)
    if mode == "patches":
        p = patch_size
        sequence_spec(mode, c, h, w, p)  # validates divisibility
        patches = x.unfold(2, p, p).unfold(3, p, p)            # (B, C, H/p, W/p, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5)             # (B, H/p, W/p, C, p, p)
        return patches.reshape(b, (h // p) * (w // p), c * p * p)
    raise ValueError(f"sequence_mode must be one of {SEQUENCE_MODES}, got '{mode}'")
