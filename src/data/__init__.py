"""API pública ligera para el paquete src.data."""

from .client import PolymarketDataClient
from .preprocessing import (
    build_snapshot_market,
    get_market_end_time,
    get_snapshot_cutoff_time,
    get_snapshot_price,
    infer_resolution_from_market,
    preprocess_markets,
)

__all__ = [
    "PolymarketDataClient",
    "DataFetcher",
    "build_snapshot_market",
    "get_market_end_time",
    "get_snapshot_cutoff_time",
    "get_snapshot_price",
    "infer_resolution_from_market",
    "preprocess_markets",
]


def __getattr__(name: str):
    if name == "DataFetcher":
        from .fetcher import DataFetcher

        return DataFetcher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
