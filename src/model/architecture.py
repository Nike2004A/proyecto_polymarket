"""Tabular snapshot model returning logits for p_yes."""

from __future__ import annotations

import torch
import torch.nn as nn


class MarketValueNet(nn.Module):
    """Simple MLP over numerical features + category/text embeddings."""

    def __init__(
        self,
        num_numerical_features: int,
        num_categories: int = 10,
        category_embed_dim: int = 8,
        text_embed_dim: int = 384,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        task: str = "classification",
    ):
        super().__init__()
        self.task = task
        hidden_dims = hidden_dims or [256, 128, 64]

        self.category_embedding = nn.Embedding(num_categories, category_embed_dim)

        layers = []
        input_dim = num_numerical_features + category_embed_dim + text_embed_dim
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        numerical_features: torch.Tensor,
        category_ids: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        cat_emb = self.category_embedding(category_ids)
        x = torch.cat([numerical_features, cat_emb, text_embeddings], dim=1)
        return self.network(x).squeeze(-1)
