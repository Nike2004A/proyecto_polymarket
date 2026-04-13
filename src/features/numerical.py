"""Observable numerical features built from snapshot-compatible histories."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.snapshots import PreparedHistory, get_points_until, latest_price_before, prepare_price_history


NUMERICAL_FEATURE_NAMES = [
    "snapshot_price_yes",
    "days_to_end",
    "market_age_days_at_snapshot",
    "days_since_last_trade",
    "history_points_1d",
    "history_points_3d",
    "history_points_7d",
    "history_points_30d",
    "history_span_days",
    "return_1d",
    "return_3d",
    "return_7d",
    "return_14d",
    "return_30d",
    "realized_vol_1d",
    "realized_vol_3d",
    "realized_vol_7d",
    "realized_vol_14d",
    "realized_vol_30d",
    "trend_slope_7d",
    "trend_slope_30d",
    "price_percentile_30d",
    "distance_to_30d_min",
    "distance_to_30d_max",
    "neg_risk",
]

NUM_NUMERICAL_FEATURES = len(NUMERICAL_FEATURE_NAMES)
WINDOW_DAYS = (1, 3, 7, 14, 30)


def extract_numerical_features(
    market: dict,
    price_history: PreparedHistory | list[dict] | None = None,
    snapshot_time: pd.Timestamp | None = None,
    snapshot_price_yes: float | None = None,
    days_to_end: float | None = None,
) -> np.ndarray:
    """Extract snapshot-compatible numerical features for a single market."""
    history = _coerce_history(price_history)
    if snapshot_time is None:
        snapshot_time = pd.Timestamp.now(tz="UTC")

    history_until_snapshot = get_points_until(history, snapshot_time)
    if snapshot_price_yes is None:
        snapshot_price_yes = latest_price_before(history, snapshot_time)
        if snapshot_price_yes is None:
            snapshot_price_yes = _market_price_fallback(market)

    snapshot_ts = int(snapshot_time.timestamp())

    end_date = _parse_time(market.get("endDate"))
    if days_to_end is None:
        if end_date is not None:
            days_to_end = max(0.0, (end_date - snapshot_time).total_seconds() / 86400.0)
        else:
            days_to_end = 0.0

    created_at = _parse_time(market.get("createdAt"))
    if created_at is not None:
        market_age_days = max(0.0, (snapshot_time - created_at).total_seconds() / 86400.0)
    else:
        market_age_days = 0.0

    if history_until_snapshot.is_empty:
        days_since_last_trade = market_age_days
        history_span_days = 0.0
    else:
        days_since_last_trade = max(
            0.0,
            (snapshot_ts - int(history_until_snapshot.timestamps[-1])) / 86400.0,
        )
        history_span_days = max(
            0.0,
            (snapshot_ts - int(history_until_snapshot.timestamps[0])) / 86400.0,
        )

    counts: dict[int, int] = {}
    returns: dict[int, float] = {}
    vols: dict[int, float] = {}
    slopes: dict[int, float] = {}

    for days in WINDOW_DAYS:
        counts[days] = _count_points_in_window(history_until_snapshot, snapshot_ts, days)
        returns[days], vols[days], slopes[days] = _window_summary(
            history_until_snapshot,
            snapshot_ts,
            days,
            snapshot_price=float(snapshot_price_yes),
        )

    percentile_30d, dist_min_30d, dist_max_30d = _window_price_position(
        history_until_snapshot,
        snapshot_ts,
        30,
        snapshot_price=float(snapshot_price_yes),
    )
    features = np.array([
        float(snapshot_price_yes),
        float(days_to_end),
        float(market_age_days),
        float(days_since_last_trade),
        float(counts[1]),
        float(counts[3]),
        float(counts[7]),
        float(counts[30]),
        float(history_span_days),
        float(returns[1]),
        float(returns[3]),
        float(returns[7]),
        float(returns[14]),
        float(returns[30]),
        float(vols[1]),
        float(vols[3]),
        float(vols[7]),
        float(vols[14]),
        float(vols[30]),
        float(slopes[7]),
        float(slopes[30]),
        float(percentile_30d),
        float(dist_min_30d),
        float(dist_max_30d),
        float(bool(market.get("negRisk", False))),
    ], dtype=np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def extract_numerical_features_batch(samples: list[dict]) -> np.ndarray:
    """Extract numerical features for a batch of snapshot samples."""
    rows = [
        extract_numerical_features(
            sample["market"],
            price_history=sample.get("history"),
            snapshot_time=sample.get("snapshot_time"),
            snapshot_price_yes=sample.get("snapshot_price_yes"),
            days_to_end=sample.get("days_to_end"),
        )
        for sample in samples
    ]
    return np.stack(rows).astype(np.float32)


def _coerce_history(price_history: PreparedHistory | list[dict] | None) -> PreparedHistory:
    if isinstance(price_history, PreparedHistory):
        return price_history
    return prepare_price_history(price_history)


def _parse_time(value) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        return pd.to_datetime(value, utc=True)
    except (ValueError, TypeError):
        return None


def _market_price_fallback(market: dict) -> float:
    outcome_prices = market.get("outcomePrices", [])
    if isinstance(outcome_prices, str):
        try:
            import json

            outcome_prices = json.loads(outcome_prices)
        except (ValueError, TypeError):
            outcome_prices = []

    if isinstance(outcome_prices, list) and outcome_prices:
        try:
            return float(outcome_prices[0])
        except (ValueError, TypeError):
            pass
    try:
        return float(market.get("lastTradePrice") or 0.5)
    except (ValueError, TypeError):
        return 0.5


def _count_points_in_window(history: PreparedHistory, snapshot_ts: int, window_days: int) -> int:
    if history.is_empty:
        return 0
    window_start = snapshot_ts - int(window_days * 86400)
    idx = int(np.searchsorted(history.timestamps, window_start, side="left"))
    return int(history.timestamps.size - idx)


def _window_summary(
    history: PreparedHistory,
    snapshot_ts: int,
    window_days: int,
    snapshot_price: float,
) -> tuple[float, float, float]:
    if history.is_empty:
        return 0.0, 0.0, 0.0

    window_start = snapshot_ts - int(window_days * 86400)
    start_price = _price_at_or_before(history, window_start)
    if start_price is None:
        start_price = _first_price_after(history, window_start)
    if start_price is None or start_price <= 0:
        return 0.0, 0.0, 0.0

    actual_prices, actual_timestamps = _window_points(history, window_start)
    prices_series, ts_series = _series_with_carry(
        actual_prices,
        actual_timestamps,
        carry_price=start_price,
        window_start=window_start,
    )

    realized_vol = 0.0
    if prices_series.size >= 2:
        realized_vol = float(np.std(np.diff(prices_series)))

    trend_slope = 0.0
    if prices_series.size >= 2:
        x_days = (ts_series - ts_series[0]) / 86400.0
        if np.unique(x_days).size >= 2:
            trend_slope = float(np.polyfit(x_days, prices_series, 1)[0])

    total_return = float(snapshot_price / start_price - 1.0) if start_price > 0 else 0.0
    return total_return, realized_vol, trend_slope


def _window_price_position(
    history: PreparedHistory,
    snapshot_ts: int,
    window_days: int,
    snapshot_price: float,
) -> tuple[float, float, float]:
    if history.is_empty:
        return 0.5, 0.0, 0.0

    window_start = snapshot_ts - int(window_days * 86400)
    actual_prices, actual_timestamps = _window_points(history, window_start)
    carry_price = _price_at_or_before(history, window_start)
    if carry_price is None:
        carry_price = _first_price_after(history, window_start)
    if carry_price is None:
        return 0.5, 0.0, 0.0

    prices_series, _ = _series_with_carry(
        actual_prices,
        actual_timestamps,
        carry_price=carry_price,
        window_start=window_start,
    )
    if prices_series.size == 0:
        return 0.5, 0.0, 0.0

    percentile = float(np.mean(prices_series <= snapshot_price))
    min_price = float(np.min(prices_series))
    max_price = float(np.max(prices_series))
    return percentile, float(snapshot_price - min_price), float(max_price - snapshot_price)


def _series_with_carry(
    actual_prices: np.ndarray,
    actual_timestamps: np.ndarray,
    carry_price: float,
    window_start: int,
) -> tuple[np.ndarray, np.ndarray]:
    if actual_prices.size == 0:
        return (
            np.asarray([carry_price], dtype=np.float32),
            np.asarray([window_start], dtype=np.int64),
        )

    prices = actual_prices
    timestamps = actual_timestamps
    if timestamps[0] > window_start:
        prices = np.concatenate([np.asarray([carry_price], dtype=np.float32), actual_prices])
        timestamps = np.concatenate([np.asarray([window_start], dtype=np.int64), actual_timestamps])
    return prices, timestamps


def _window_points(history: PreparedHistory, window_start: int) -> tuple[np.ndarray, np.ndarray]:
    idx = int(np.searchsorted(history.timestamps, window_start, side="left"))
    return history.prices[idx:], history.timestamps[idx:]


def _price_at_or_before(history: PreparedHistory, ts: int) -> float | None:
    idx = int(np.searchsorted(history.timestamps, ts, side="right")) - 1
    if idx < 0:
        return None
    return float(history.prices[idx])


def _first_price_after(history: PreparedHistory, ts: int) -> float | None:
    idx = int(np.searchsorted(history.timestamps, ts, side="left"))
    if idx >= history.prices.size:
        return None
    return float(history.prices[idx])
