"""Extracción de features numéricas de mercados de Polymarket."""

import numpy as np
import pandas as pd

# Orden de las features numéricas (23 features)
NUMERICAL_FEATURE_NAMES = [
    # --- Precio y spread (snapshot) ---
    "price_yes",
    "price_no",
    "spread",
    # --- Volumen y liquidez ---
    "volume_24h",
    "volume_total",
    "liquidity",
    "volume_liquidity_ratio",
    # --- Temporales de mercado ---
    "days_to_resolution",
    "market_age_days",
    # --- Series de tiempo: momentum multi-ventana ---
    "price_momentum_7d",
    "price_volatility_7d",
    "price_momentum_14d",
    "price_momentum_30d",
    "price_volatility_30d",
    # --- Series de tiempo: tendencia y cobertura ---
    "price_trend_slope",      # pendiente de regresión lineal (proxy de tendencia ARIMA)
    "ewm_momentum",           # momentum exponencialmente ponderado (más peso a datos recientes)
    "ts_coverage",            # log(n_puntos + 1) — qué tan bien cubierta está la serie
    "ts_days_span",           # días entre primer y último registro en history
    # --- Order book ---
    "bid_depth",
    "ask_depth",
    "book_imbalance",
    # --- Indicador de trayectoria ---
    "price_at_halflife",      # precio al 50% del tiempo de vida del mercado
    # --- Mercado estructural ---
    "neg_risk",               # 1 si es negRisk market (complementario en un grupo de eventos)
                              # negRisk=True resuelve Yes solo 11% vs 43% sin negRisk
]

NUM_NUMERICAL_FEATURES = len(NUMERICAL_FEATURE_NAMES)  # 23


def extract_numerical_features(
    market: dict,
    price_history: list[dict] | None = None,
    order_book: dict | None = None,
) -> np.ndarray:
    """
    Extrae el vector de features numéricas de un mercado.

    Args:
        market: Diccionario con datos del mercado (ya parseado).
        price_history: Lista de registros [{t: unix_ts, p: price}, ...].
        order_book: Diccionario del order book {bids: [...], asks: [...]}.

    Returns:
        np.ndarray de forma (23,) con las features numéricas.
    """
    # Precios snapshot
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

    # Features temporales de mercado
    days_to_resolution = _safe_float(market.get("days_to_resolution", 30))
    market_age_days = _safe_float(market.get("market_age_days", 0))

    # Features de series de tiempo
    ts_features = _compute_ts_features(price_history)

    # Order book
    bid_depth, ask_depth, book_imbalance = _compute_book_features(order_book)

    # Precio al 50% de vida del mercado (de la serie de tiempo)
    price_halflife = _compute_price_at_halflife(price_history, market_age_days, days_to_resolution)

    # negRisk: mercados complementarios dentro de un grupo de eventos.
    # Resuelven Yes solo ~11% del tiempo vs ~43% en mercados normales.
    neg_risk = float(bool(market.get("negRisk", False)))

    # El orden debe coincidir exactamente con NUMERICAL_FEATURE_NAMES
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
        ts_features["momentum_7d"],
        ts_features["volatility_7d"],
        ts_features["momentum_14d"],
        ts_features["momentum_30d"],
        ts_features["volatility_30d"],
        ts_features["trend_slope"],
        ts_features["ewm_momentum"],
        ts_features["coverage"],
        ts_features["days_span"],
        bid_depth,
        ask_depth,
        book_imbalance,
        price_halflife,
        neg_risk,
    ], dtype=np.float32)

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
        price_histories: Dict {market_id: [{t, p}, ...]}.
        order_books: Dict {market_id: order_book}.

    Returns:
        np.ndarray de forma (N, 23).
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


# ── Helpers privados ──────────────────────────────────────────────────────────

def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def _compute_ts_features(price_history: list[dict] | None) -> dict:
    """
    Calcula todas las features de series de tiempo a partir del historial de precios.

    Retorna un dict con claves:
        momentum_7d, volatility_7d, momentum_14d, momentum_30d, volatility_30d,
        trend_slope, ewm_momentum, coverage, days_span
    """
    zeros = {
        "momentum_7d": 0.0, "volatility_7d": 0.0,
        "momentum_14d": 0.0, "momentum_30d": 0.0, "volatility_30d": 0.0,
        "trend_slope": 0.0, "ewm_momentum": 0.0,
        "coverage": 0.0, "days_span": 0.0,
    }

    if not price_history or len(price_history) < 2:
        return zeros

    # Extraer precios y timestamps, filtrando valores inválidos
    records = sorted(price_history, key=lambda x: x.get("t", 0))
    prices = [_safe_float(r.get("p", r.get("price", 0))) for r in records]
    timestamps = [_safe_float(r.get("t", 0)) for r in records]

    prices = [p for p in prices if 0 < p <= 1]
    if len(prices) < 2:
        return zeros

    # Cobertura y span temporal
    n = len(prices)
    coverage = float(np.log1p(n))
    days_span = (timestamps[-1] - timestamps[0]) / 86400.0 if len(timestamps) >= 2 else 0.0

    # Ventanas de momentum y volatilidad.
    # NOTA: window_size son los últimos N *registros* de la serie, no días calendario.
    # La densidad de la serie varía por mercado, así que "7 registros" puede ser
    # 1 semana o 1 mes según la actividad del mercado.
    def _window_stats(window_size: int) -> tuple[float, float]:
        """Retorna (momentum, volatilidad) de los últimos window_size registros."""
        recent = prices[-window_size:] if len(prices) >= window_size else prices
        if len(recent) < 2:
            return 0.0, 0.0
        mom = (recent[-1] - recent[0]) / recent[0] if recent[0] > 0 else 0.0
        vol = float(np.std(recent))
        return mom, vol

    mom_7d, vol_7d = _window_stats(7)
    mom_14d, _ = _window_stats(14)
    mom_30d, vol_30d = _window_stats(30)

    # Pendiente de tendencia lineal (sobre todos los puntos normalizados 0→1)
    x = np.linspace(0, 1, n)
    trend_slope = float(np.polyfit(x, prices, 1)[0]) if n >= 3 else 0.0

    # EWM momentum: diferencia entre EWM rápida (span=7) y lenta (span=30)
    prices_arr = np.array(prices)
    if len(prices_arr) >= 7:
        alpha_fast = 2 / (7 + 1)
        alpha_slow = 2 / (30 + 1)
        ewm_fast = _ewm_last(prices_arr, alpha_fast)
        ewm_slow = _ewm_last(prices_arr, alpha_slow)
        ewm_momentum = (ewm_fast - ewm_slow) / ewm_slow if ewm_slow > 0 else 0.0
    else:
        ewm_momentum = mom_7d

    return {
        "momentum_7d": mom_7d,
        "volatility_7d": vol_7d,
        "momentum_14d": mom_14d,
        "momentum_30d": mom_30d,
        "volatility_30d": vol_30d,
        "trend_slope": trend_slope,
        "ewm_momentum": float(ewm_momentum),
        "coverage": coverage,
        "days_span": days_span,
    }


def _ewm_last(prices: np.ndarray, alpha: float) -> float:
    """Calcula el último valor del EWM (exponential weighted mean)."""
    result = float(prices[0])
    for p in prices[1:]:
        result = alpha * float(p) + (1 - alpha) * result
    return result


def _compute_price_at_halflife(
    price_history: list[dict] | None,
    market_age_days: float,
    days_to_resolution: float,
) -> float:
    """
    Precio al ~50% del tiempo de vida del mercado.

    Útil para detectar si el mercado empezó caro y bajó (o viceversa),
    lo que es informativo para clasificar oportunidades de compra.
    """
    if not price_history or len(price_history) < 3:
        return 0.5

    total_life = market_age_days + days_to_resolution
    if total_life <= 0:
        return 0.5

    records = sorted(price_history, key=lambda x: x.get("t", 0))
    t_start = records[0].get("t", 0)
    t_end = records[-1].get("t", t_start)
    t_range = t_end - t_start
    if t_range <= 0:
        return 0.5

    t_half = t_start + t_range * 0.5
    # Buscar el registro más cercano al punto medio
    closest = min(records, key=lambda r: abs(r.get("t", 0) - t_half))
    return _safe_float(closest.get("p", closest.get("price", 0.5)), default=0.5)


def _compute_book_features(order_book: dict | None) -> tuple[float, float, float]:
    """Calcula bid_depth, ask_depth e imbalance del order book."""
    if not order_book:
        return 0.0, 0.0, 0.0

    bids = order_book.get("bids", [])
    asks = order_book.get("asks", [])

    bid_depth = sum(_safe_float(b.get("size", b.get("s", 0))) for b in bids)
    ask_depth = sum(_safe_float(a.get("size", a.get("s", 0))) for a in asks)

    total = bid_depth + ask_depth
    imbalance = (bid_depth - ask_depth) / total if total > 0 else 0.0

    return bid_depth, ask_depth, imbalance
