"""Arquitectura del modelo MarketValueNet (Wide & Deep)."""

import torch
import torch.nn as nn


class MarketValueNet(nn.Module):
    """
    Modelo para evaluar si un mercado de Polymarket representa
    una oportunidad de compra (infravalorado).

    Arquitectura Wide & Deep:
    - Wide path: features numéricas directas
    - Deep path: features numéricas + embeddings de categoría + texto
    """

    def __init__(
        self,
        num_numerical_features: int = 14,
        num_categories: int = 20,
        category_embed_dim: int = 8,
        text_embed_dim: int = 384,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        task: str = "classification",
    ):
        super().__init__()
        self.task = task

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        # Category embedding
        self.category_embedding = nn.Embedding(num_categories, category_embed_dim)

        # Deep path
        deep_input_dim = num_numerical_features + category_embed_dim + text_embed_dim
        deep_layers = []
        prev_dim = deep_input_dim
        for hidden_dim in hidden_dims:
            deep_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        self.deep_net = nn.Sequential(*deep_layers)

        # Wide path (features numéricas directamente)
        self.wide_net = nn.Linear(num_numerical_features, 32)

        # Output head
        combined_dim = prev_dim + 32
        if task == "classification":
            self.head = nn.Sequential(
                nn.Linear(combined_dim, 1),
                nn.Sigmoid(),
            )
        else:  # regression
            self.head = nn.Linear(combined_dim, 1)

    def forward(
        self,
        numerical_features: torch.Tensor,
        category_ids: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            numerical_features: (batch, num_numerical)
            category_ids: (batch,) — int
            text_embeddings: (batch, text_embed_dim)

        Returns:
            (batch,) — scores
        """
        # Category embedding
        cat_emb = self.category_embedding(category_ids)

        # Deep path
        deep_input = torch.cat([numerical_features, cat_emb, text_embeddings], dim=1)
        deep_out = self.deep_net(deep_input)

        # Wide path
        wide_out = torch.relu(self.wide_net(numerical_features))

        # Combine
        combined = torch.cat([deep_out, wide_out], dim=1)
        return self.head(combined).squeeze(-1)
