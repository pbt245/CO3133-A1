"""Transformer encoder for image classification over row/column/patch tokens.

Pipeline:
  image (B, C, H, W)
  -> tokens (B, N, token_dim)              image_to_sequence (rows / cols / patches)
  -> token embedding (B, N, d_model)       Linear projection
  -> [CLS] + tokens (B, N+1, d_model)      optional learnable CLS token
  -> + positional encoding                 learned | sinusoidal | none
  -> depth x pre-norm encoder blocks       x = x + MHSA(LN(x)); x = x + MLP(LN(x))
  -> LayerNorm -> CLS output (or mean) -> Linear -> logits (B, K)

Multi-head self-attention (written by hand, see MultiHeadSelfAttention):
  inputs : the token sequence X (B, N, d_model); Q = X W_Q, K = X W_K, V = X W_V
  weights: A = softmax(Q K^T / sqrt(d_k))  shape (B, heads, N, N), each row sums to 1
  outputs: A V, heads concatenated and projected back to (B, N, d_model)
Only nn.Linear / nn.LayerNorm / nn.Dropout / nn.GELU come from PyTorch.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models.sequence import image_to_sequence, sequence_spec


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.store_attention = False
        self.last_attention = None

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                                           # each (B, heads, N, d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)          # (B, heads, N, N)
        if self.store_attention:
            self.last_attention = attn.detach()
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, d)      # concat heads
        return self.proj_dropout(self.out_proj(out))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float = 2.0,
                 dropout: float = 0.1, attn_dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, attn_dropout, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, num_positions: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, num_positions, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        return x + self.pe[:, : x.shape[1]]


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, num_positions: int, d_model: int):
        super().__init__()
        if d_model % 2:
            raise ValueError("sinusoidal positional encoding needs an even d_model")
        position = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(num_positions, d_model)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.shape[1]].to(x.dtype)


class SequenceTransformer(nn.Module):
    def __init__(self, in_channels: int, image_size: int, num_classes: int,
                 sequence_mode: str = "patches", patch_size: int = 4, d_model: int = 96,
                 num_heads: int = 4, depth: int = 4, mlp_ratio: float = 2.0,
                 dropout: float = 0.1, attn_dropout: float = 0.0,
                 pos_encoding: str = "learned", pooling: str = "cls"):
        super().__init__()
        self.sequence_mode = sequence_mode
        self.patch_size = int(patch_size)
        self.image_size = image_size
        self.pooling = pooling
        self.seq_len, self.token_dim = sequence_spec(sequence_mode, in_channels, image_size,
                                                     image_size, self.patch_size)
        self.token_proj = nn.Linear(self.token_dim, d_model)
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        elif pooling == "mean":
            self.cls_token = None
        else:
            raise ValueError("pooling must be 'cls' or 'mean'")
        n_pos = self.seq_len + (1 if self.cls_token is not None else 0)
        if pos_encoding == "learned":
            self.pos_encoding = LearnedPositionalEncoding(n_pos, d_model)
        elif pos_encoding == "sinusoidal":
            self.pos_encoding = SinusoidalPositionalEncoding(n_pos, d_model)
        elif pos_encoding in ("none", None):
            self.pos_encoding = nn.Identity()
        else:
            raise ValueError("pos_encoding must be learned | sinusoidal | none")
        self.pos_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(int(depth))
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self.apply(self._init_weights)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if isinstance(self.pos_encoding, LearnedPositionalEncoding):
            nn.init.trunc_normal_(self.pos_encoding.pe, std=0.02)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def set_store_attention(self, flag: bool) -> None:
        for blk in self.blocks:
            blk.attn.store_attention = flag
            if not flag:
                blk.attn.last_attention = None

    def forward(self, x):
        tokens = self.token_proj(image_to_sequence(x, self.sequence_mode, self.patch_size))
        if self.cls_token is not None:
            tokens = torch.cat([self.cls_token.expand(tokens.shape[0], -1, -1), tokens], dim=1)
        tokens = self.pos_dropout(self.pos_encoding(tokens))
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        feat = tokens[:, 0] if self.cls_token is not None else tokens.mean(dim=1)
        return self.head(feat)
