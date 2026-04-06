"""Encoding de features categóricas para mercados de Polymarket."""

import json
from pathlib import Path

import numpy as np

# Categorías definidas para Polymarket.
# El campo events[].tags está vacío en la API — la categoría se infiere del slug.
DEFAULT_CATEGORIES = [
    "crypto",       # btc, eth, sol, xrp, bnb, doge, hype, token launches...
    "esports",      # cs2, lol, dota2, fl1 (League of Legends, Counter-Strike...)
    "sports",       # atp, wta (tennis), mls, mlb, nfl, nba...
    "politics",     # elections, presidents, congress, senate...
    "economics",    # gdp, inflation, fed, interest rates...
    "technology",   # apple, google, tesla, spacex, ai...
    "entertainment",# oscars, movies, music...
    "health",       # medical, fda, disease...
    "other",        # mercados generales ("will X happen?")
    "unknown",      # no se pudo inferir
]

# Prefijos de slug → categoría. Basado en análisis de 20k mercados resueltos.
# Esto cubre ~75% de los mercados; el resto cae al keyword matching.
_SLUG_PREFIX_MAP: dict[str, str] = {
    # Crypto (tokens y precios)
    "btc": "crypto", "bitcoin": "crypto", "eth": "crypto", "ethereum": "crypto",
    "sol": "crypto", "xrp": "crypto", "bnb": "crypto", "doge": "crypto",
    "hype": "crypto", "highest": "crypto",
    # Esports
    "lol": "esports", "cs2": "esports", "dota2": "esports", "fl1": "esports",
    "valorant": "esports", "rl": "esports",
    # Sports
    "atp": "sports", "wta": "sports", "mls": "sports", "mlb": "sports",
    "nfl": "sports", "nba": "sports", "nhl": "sports", "ufc": "sports",
    "fifa": "sports",
    # Politics / elections
    "2024": "politics", "2025": "politics", "2026": "politics",
}

# Keyword matching sobre question+slug como fallback
_KEYWORD_MAP: dict[str, list[str]] = {
    "politics": [
        "president", "election", "trump", "biden", "harris", "congress",
        "senate", "vote", "governor", "mayor", "ballot", "party", "democrat",
        "republican", "primary", "electoral",
    ],
    "crypto": [
        "bitcoin", "ethereum", "crypto", "btc", "eth", "solana", "token",
        "blockchain", "defi", "nft", "dao", "stablecoin", "altcoin",
        "market cap", "fdv", "launch",
        # Tokens frecuentes que no tienen slug prefix propio
        "bnb", "xrp", "doge", "hyperliquid", "hype", "plasma", "aster",
        "zama", "infinex", "trove", "foresee",
        # Patrones de precio y venta crypto
        "public sale", "auction clearing", "dip to", "clearing price",
        "committed to the",
    ],
    "sports": [
        "nba", "nfl", "soccer", "football", "tennis", "mlb", "nhl",
        "game", "match", "tournament", "championship", "world cup",
        "olympics", "medal", "season",
    ],
    "esports": [
        "league of legends", "counter-strike", "dota", "valorant",
        "esport", "gaming tournament",
    ],
    "economics": [
        "gdp", "inflation", "fed", "interest rate", "recession",
        "unemployment", "cpi", "rate hike", "cut",
    ],
    "technology": [
        "apple", "google", "tesla", "spacex", "ai", "openai", "model",
        "microsoft", "meta", "launch", "ipo",
    ],
    "entertainment": [
        "oscar", "movie", "album", "grammy", "celebrity", "tv show",
        "netflix", "box office", "award",
    ],
    "health": [
        "fda", "vaccine", "covid", "cancer", "drug", "clinical", "disease",
        "hospital", "medical",
    ],
}


class CategoryEncoder:
    """
    Encoder de categorías de mercados a IDs enteros para nn.Embedding.

    Estrategia (en orden de prioridad):
    1. Prefijo del slug → cubre ~75% de los mercados (esports, crypto, sports)
    2. Keyword matching sobre question + slug → cubre la mayoría del resto
    3. Fallback: "unknown"

    NOTA: events[].tags está vacío en la API de Polymarket — no se puede usar.
    """

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
        slug = market.get("slug", "").lower().strip()
        question = market.get("question", "").lower()

        # 1. Prefijo del slug (más preciso para esports, crypto, sports)
        slug_prefix = slug.split("-")[0] if slug else ""
        if slug_prefix in _SLUG_PREFIX_MAP:
            cat = _SLUG_PREFIX_MAP[slug_prefix]
            return self.cat_to_id.get(cat, self.unknown_id)

        # 2. Keyword matching sobre question + slug
        text = f"{question} {slug}"
        for category, keywords in _KEYWORD_MAP.items():
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
