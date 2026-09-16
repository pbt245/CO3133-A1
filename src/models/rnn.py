"""LSTM / GRU classifier over an image read as a sequence.

Timestep definition (sequence_mode):
  rows    : step t = row t        -> T = H,             input_size = C*W
  cols    : step t = column t     -> T = W,             input_size = C*H
  patches : step t = patch t      -> T = (H/p)*(W/p),   input_size = C*p*p
Hidden representation: h_t in R^{hidden_size} (per layer), summarizing steps 1..t.
Classification uses the final hidden state of the top layer, h_T (both directions
concatenated if bidirectional), followed by Dropout -> Linear.

The recurrent layer itself is torch.nn.LSTM / torch.nn.GRU (library); LSTM and GRU share
every other component so that the cell type is the only changed factor.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.sequence import image_to_sequence, sequence_spec


class RecurrentClassifier(nn.Module):
    def __init__(self, cell: str, in_channels: int, image_size: int, num_classes: int,
                 sequence_mode: str = "rows", patch_size: int = 4, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2, bidirectional: bool = False,
                 head_dropout: float = 0.2):
        super().__init__()
        cell = cell.lower()
        if cell not in ("lstm", "gru"):
            raise ValueError("cell must be 'lstm' or 'gru'")
        self.cell = cell
        self.sequence_mode = sequence_mode
        self.patch_size = int(patch_size)
        self.bidirectional = bool(bidirectional)
        self.seq_len, self.input_size = sequence_spec(sequence_mode, in_channels, image_size,
                                                      image_size, self.patch_size)
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=self.input_size,
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            bidirectional=self.bidirectional,
        )
        out_dim = int(hidden_size) * (2 if self.bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(float(head_dropout)), nn.Linear(out_dim, num_classes))

    def forward(self, x):
        seq = image_to_sequence(x, self.sequence_mode, self.patch_size)   # (B, T, input_size)
        _, hidden = self.rnn(seq)
        h_n = hidden[0] if self.cell == "lstm" else hidden               # (layers*dirs, B, H)
        if self.bidirectional:
            last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last = h_n[-1]
        return self.head(last)

    def forward_all_steps(self, x):
        """Logits after each timestep (unidirectional only): (B, T, K).
        Used to show how the prediction evolves as more rows/patches are read."""
        if self.bidirectional:
            raise ValueError("forward_all_steps is only defined for unidirectional RNNs")
        seq = image_to_sequence(x, self.sequence_mode, self.patch_size)
        out, _ = self.rnn(seq)                                            # (B, T, H)
        return self.head(out)
