"""Sequence model with a static branch for snapshot-based p_yes prediction."""

from __future__ import annotations

import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence


class _PriceSequenceRNNBase(nn.Module):
    def __init__(
        self,
        rnn_cls: type[nn.GRU] | type[nn.LSTM],
        rnn_attr_name: str,
        input_dim: int,
        static_num_features: int,
        num_categories: int = 10,
        category_embed_dim: int = 8,
        text_embed_dim: int = 384,
        hidden_dim: int = 128,
        static_hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        task: str = "classification",
    ):
        super().__init__()
        self.task = task
        self._rnn_attr_name = rnn_attr_name
        rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        setattr(self, rnn_attr_name, rnn)
        self.category_embedding = nn.Embedding(num_categories, category_embed_dim)
        static_input_dim = static_num_features + category_embed_dim + text_embed_dim
        self.static_net = nn.Sequential(
            nn.Linear(static_input_dim, static_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + static_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        sequences: torch.Tensor,
        sequence_lengths: torch.Tensor,
        static_numerical: torch.Tensor,
        category_ids: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        packed = pack_padded_sequence(
            sequences,
            sequence_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        rnn = getattr(self, self._rnn_attr_name)
        _, hidden = rnn(packed)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        sequence_hidden = hidden[-1]

        cat_emb = self.category_embedding(category_ids)
        static_input = torch.cat([static_numerical, cat_emb, text_embeddings], dim=1)
        static_hidden = self.static_net(static_input)

        fused = torch.cat([sequence_hidden, static_hidden], dim=1)
        return self.head(fused).squeeze(-1)


class PriceSequenceGRU(_PriceSequenceRNNBase):
    """GRU over fixed-grid sequences plus snapshot-compatible static context."""

    def __init__(self, *args, **kwargs):
        super().__init__(nn.GRU, "gru", *args, **kwargs)


class PriceSequenceLSTM(_PriceSequenceRNNBase):
    """LSTM over fixed-grid sequences plus snapshot-compatible static context."""

    def __init__(self, *args, **kwargs):
        super().__init__(nn.LSTM, "lstm", *args, **kwargs)
