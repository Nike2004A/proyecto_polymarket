"""Modelo GRU para series de tiempo de precios."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class PriceSequenceGRU(nn.Module):
    """Clasificador puro TS sobre secuencias de precio pre-snapshot."""

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        task: str = "classification",
    ):
        super().__init__()
        self.task = task
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        sequences: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        packed = pack_padded_sequence(
            sequences,
            sequence_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        final_hidden = hidden[-1]
        return self.head(final_hidden).squeeze(-1)
