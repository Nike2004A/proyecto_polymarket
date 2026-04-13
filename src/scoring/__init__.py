"""Public API for active-market scoring."""

from .scorer import score_active_markets, summarize_scored_markets
from .signals import generate_signals

__all__ = [
    "score_active_markets",
    "summarize_scored_markets",
    "generate_signals",
]
