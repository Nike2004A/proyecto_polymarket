"""Extracción de features numéricas de mercados de Polymarket."""

import numpy as np
import pandas as pd

# Orden de las features numéricas (14 features)
NUMERICAL_FEATURE_NAMES = [
    "price_yes",
    "price_no",
    "spread",
    "volume_24h",
    "volume_total",
    "liquidity",
    "volume_liquidity_ratio",
    "days_to_resolution",
    "market_age_days",
    "price_momentum_7d",
    "price_volatility_7d",
    "bid_depth",
    "ask_depth",
    "book_imbalance",
]

NUM_NUMERICAL_FEATURES = len(NUMERICAL_FEATURE_NAMES)


def extract_numerical_features(
    market: dict,
    price_history: list[dict] | None = None,
    order_book: dict | None = None,
) -> np.ndarray:
    """
    Extrae el vector de features numéricas de un mercado.

    Args:
        market: Diccionario con datos del mercado (ya parseado).
        price_history: Lista de registros de precio histórico [{t, p}, ...].
        order_book: Diccionario del order book {bids: [...], asks: [...]}.

    Returns:
        np.ndarray de forma (14,) con las features numéricas.
    """
    # Precios
    outcome_prices = market.get("outcomePrices", [])
    price_yes = float(outcome_prices[0]) if len(outcome_prices) > 0 else 0.5
    price_no = float(outcome_prices[1]) if len(outcome_prices) > 1 else 1 - price_yes

    # Spread
    best_bid = _safe_float(market.get("bestBid", 0))
    best_ask = _safe_float(market.get("bestAsk", 0))
    spread = _safe_float(market.get("spread", best_ask - best_bid))

    # Volumen y liquidez
    volume_24h = _safe_float(market.get("volume24hr", 0))
    volume_total = _safe_float(market.get("volume", 0))
    liquidity = _safe_float(market.get("liquidity", 0))
    vol_liq_ratio = volume_24h / liquidity if liquidity > 0 else 0.0

    # Features temporales
    days_to_resolution = _safe_float(market.get("days_to_resolution", 30))
    market_age_days = _safe_float(market.get("market_age_days", 0))

    # Momentum y volatilidad (de historial de precios)
    momentum_7d, volatility_7d = _compute_price_momentum(price_history)

    # Order book features
    bid_depth, ask_depth, book_imbalance = _compute_book_features(order_book)

    features = np.array([
        price_yes,
        price_no,
        spread,
        volume_24h,
        volume_total,
        liquidity,
        vol_liq_ratio,
        days_to_resolution,
        market_age_days,
        momentum_7d,
        volatility_7d,
        bid_depth,
        ask_depth,
        book_imbalance,
    ], dtype=np.float32)

    # Reemplazar NaN/inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def extract_numerical_features_batch(
    markets_df: pd.DataFrame,
    price_histories: dict | None = None,
    order_books: dict | None = None,
) -> np.ndarray:
    """
    Extrae features numéricas para un DataFrame de mercados.

    Args:
        markets_df: DataFrame con mercados preprocesados.
        price_histories: Dict {market_id: [price_records]}.
        order_books: Dict {market_id: order_book}.

    Returns:
        np.ndarray de forma (N, 14).
    """
    price_histories = price_histories or {}
    order_books = order_books or {}

    features_list = []
    for _, row in markets_df.iterrows():
        market_dict = row.to_dict()
        mid = market_dict.get("id", "")
        history = price_histories.get(mid)
        book = order_books.get(mid)
        features_list.append(extract_numerical_features(market_dict, history, book))

    return np.stack(features_list)


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def _compute_price_momentum(
    price_history: list[dict] | None,
) -> tuple[float, float]:
    """Calcula momentum y volatilidad de 7 días."""
    if not price_history or len(price_history) < 2:
        return 0.0, 0.0

    prices = [_safe_float(p.get("p", p.get("price", 0))) for p in price_history]
    prices = [p for p in prices if p > 0]

    if len(prices) < 2:
        return 0.0, 0.0

    # Últimos 7 registros como proxy de 7 días
    recent = prices[-7:] if len(prices) >= 7 else prices
    price_now = recent[-1]
    price_start = recent[0]

    momentum = (price_now - price_start) / price_start if price_start > 0 else 0.0
    volatility = float(np.std(recent)) if len(recent) > 1 else 0.0

    return momentum, volatility


def _compute_book_features(
    order_book: dict | None,
) -> tuple[float, float, float]:
    """Calcula features del order book: bid_depth, ask_depth, imbalance."""
    if not order_book:
        return 0.0, 0.0, 0.0

    bids = order_book.get("bids", [])
    asks = order_book.get("asks", [])

    bid_depth = sum(_safe_float(b.get("size", b.get("s", 0))) for b in bids)
    ask_depth = sum(_safe_float(a.get("size", a.get("s", 0))) for a in asks)

    total = bid_depth + ask_depth
    imbalance = (bid_depth - ask_depth) / total if total > 0 else 0.0

    return bid_depth, ask_depth, imbalance
