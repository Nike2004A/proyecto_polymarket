"""Encoding de features categóricas para mercados de Polymarket."""

import json
from pathlib import Path

import numpy as np

# Categorías predefinidas comunes en Polymarket
DEFAULT_CATEGORIES = [
    "politics",
    "crypto",
    "sports",
    "entertainment",
    "science",
    "economics",
    "technology",
    "world",
    "finance",
    "elections",
    "climate",
    "health",
    "culture",
    "business",
    "legal",
    "ai",
    "social-media",
    "gaming",
    "other",
    "unknown",
]


class CategoryEncoder:
    """Encoder de categorías de mercados a IDs enteros para nn.Embedding."""

    def __init__(self, categories: list[str] | None = None):
        self.categories = categories or DEFAULT_CATEGORIES
        self.cat_to_id: dict[str, int] = {
            cat: idx for idx, cat in enumerate(self.categories)
        }
        self.id_to_cat: dict[int, str] = {
            idx: cat for idx, cat in enumerate(self.categories)
        }
        self.unknown_id = self.cat_to_id.get("unknown", len(self.categories) - 1)

    @property
    def num_categories(self) -> int:
        return len(self.categories)

    def encode(self, market: dict) -> int:
        """Devuelve el ID de categoría para un mercado."""
        tags = market.get("tags", [])
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except (json.JSONDecodeError, TypeError):
                tags = []

        # Buscar la primera etiqueta que coincida
        if isinstance(tags, list):
            for tag in tags:
                tag_label = tag.get("label", tag) if isinstance(tag, dict) else str(tag)
                tag_lower = tag_label.lower().strip()
                if tag_lower in self.cat_to_id:
                    return self.cat_to_id[tag_lower]

        # Intentar con el slug o la pregunta para inferir categoría
        question = market.get("question", "").lower()
        slug = market.get("slug", "").lower()
        text = f"{question} {slug}"

        keyword_map = {
            "politics": ["president", "election", "trump", "biden", "congress", "senate", "vote"],
            "crypto": ["bitcoin", "ethereum", "crypto", "btc", "eth", "solana", "token"],
            "sports": ["nba", "nfl", "soccer", "football", "tennis", "mlb", "game", "match"],
            "entertainment": ["oscar", "movie", "album", "grammy", "celebrity", "tv show"],
            "economics": ["gdp", "inflation", "fed", "interest rate", "recession"],
            "technology": ["apple", "google", "tesla", "ai", "spacex", "launch"],
            "finance": ["stock", "s&p", "dow", "nasdaq", "market cap"],
            "elections": ["primary", "electoral", "governor", "mayor", "ballot"],
        }

        for category, keywords in keyword_map.items():
            if any(kw in text for kw in keywords):
                return self.cat_to_id.get(category, self.unknown_id)

        return self.unknown_id

    def encode_batch(self, markets: list[dict]) -> np.ndarray:
        """Codifica un batch de mercados."""
        return np.array([self.encode(m) for m in markets], dtype=np.int64)

    def save(self, path: str) -> None:
        """Guarda el encoder en disco."""
        data = {"categories": self.categories}
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "CategoryEncoder":
        """Carga un encoder desde disco."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(categories=data["categories"])
