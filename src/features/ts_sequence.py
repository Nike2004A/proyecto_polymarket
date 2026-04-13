"""Fixed-grid temporal sequences for snapshot-based models."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ..data.preprocessing import get_market_end_time
from ..data.snapshots import PreparedHistory, get_points_until, prepare_price_history


SEQUENCE_FEATURE_NAMES = [
    "price_yes_ffill",
    "delta_price",
    "observed_mask",
]


def extract_sequence_points(
    price_history: Iterable[dict] | PreparedHistory | None,
    cutoff_time: pd.Timestamp | None = None,
) -> list[tuple[pd.Timestamp, float]]:
    """Return parsed points up to a cutoff for diagnostics/tests."""
    history = _coerce_history(price_history)
    if cutoff_time is not None:
        history = get_points_until(history, cutoff_time)
    return [
        (pd.to_datetime(int(ts), unit="s", utc=True), float(price))
        for ts, price in zip(history.timestamps.tolist(), history.prices.tolist())
    ]


def build_grid_sequence(
    price_history: Iterable[dict] | PreparedHistory | None,
    snapshot_time: pd.Timestamp,
    lookback_days: int = 30,
    step_hours: int = 12,
) -> tuple[np.ndarray | None, int | None, float | None]:
    """Build a fixed-grid ffilled sequence ending at the snapshot time."""
    history = _coerce_history(price_history)
    history_until_snapshot = get_points_until(history, snapshot_time)
    if history_until_snapshot.is_empty:
        return None, None, None

    steps = int(lookback_days * 24 / step_hours)
    if steps <= 0:
        raise ValueError("lookback_days y step_hours producen 0 pasos.")

    snapshot_ts = int(snapshot_time.timestamp())
    step_seconds = int(step_hours * 3600)
    window_start = snapshot_ts - int(lookback_days * 86400)
    grid_targets = np.arange(window_start + step_seconds, snapshot_ts + 1, step_seconds, dtype=np.int64)
    if grid_targets.size != steps:
        raise ValueError("La rejilla temporal no coincide con el número esperado de pasos.")

    timestamps = history_until_snapshot.timestamps
    prices = history_until_snapshot.prices

    initial_price = _price_at_or_before(history_until_snapshot, window_start)
    if initial_price is None:
        initial_price = 0.5

    sequence = np.zeros((steps, len(SEQUENCE_FEATURE_NAMES)), dtype=np.float32)
    prev_price = float(initial_price)
    prev_idx = int(np.searchsorted(timestamps, window_start, side="right"))

    for i, target_ts in enumerate(grid_targets):
        idx = int(np.searchsorted(timestamps, target_ts, side="right"))
        observed = 1.0 if idx > prev_idx else 0.0
        if idx > 0:
            current_price = float(prices[idx - 1])
        else:
            current_price = float(initial_price)
        delta_price = 0.0 if i == 0 else current_price - prev_price
        sequence[i] = np.asarray([current_price, delta_price, observed], dtype=np.float32)
        prev_price = current_price
        prev_idx = idx

    snapshot_price = float(prices[-1])
    return sequence, steps, snapshot_price


def build_market_price_sequence(
    market: dict,
    price_histories: dict | None,
    snapshot_offset_days: int = 7,
    lookback_days: int = 30,
    step_hours: int = 12,
) -> tuple[np.ndarray | None, int | None, float | None, pd.Timestamp | None]:
    """Build a resolved-market grid sequence using endDate - offset_days."""
    end_time = get_market_end_time(market)
    if end_time is None:
        return None, None, None, None

    snapshot_time = end_time - pd.Timedelta(days=snapshot_offset_days)
    market_id = str(market.get("id", ""))
    history = (price_histories or {}).get(market_id)
    sequence, length, snapshot_price = build_grid_sequence(
        history,
        snapshot_time=snapshot_time,
        lookback_days=lookback_days,
        step_hours=step_hours,
    )
    return sequence, length, snapshot_price, snapshot_time


def build_live_price_sequence(
    price_history: Iterable[dict] | PreparedHistory | None,
    snapshot_time: pd.Timestamp | None = None,
    lookback_days: int = 30,
    step_hours: int = 12,
) -> tuple[np.ndarray | None, int | None, float | None]:
    """Build a live grid sequence up to the provided snapshot time."""
    if snapshot_time is None:
        snapshot_time = pd.Timestamp.now(tz="UTC")
    return build_grid_sequence(
        price_history,
        snapshot_time=snapshot_time,
        lookback_days=lookback_days,
        step_hours=step_hours,
    )


def _coerce_history(price_history: Iterable[dict] | PreparedHistory | None) -> PreparedHistory:
    if isinstance(price_history, PreparedHistory):
        return price_history
    return prepare_price_history(price_history)


def _price_at_or_before(history: PreparedHistory, ts: int) -> float | None:
    idx = int(np.searchsorted(history.timestamps, ts, side="right")) - 1
    if idx < 0:
        return None
    return float(history.prices[idx])
