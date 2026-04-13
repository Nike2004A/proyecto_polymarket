"""Utilities to build snapshot-based training samples from cached histories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .preprocessing import get_market_end_time, infer_resolution_from_market


@dataclass(frozen=True)
class PreparedHistory:
    """Normalized price history sorted by timestamp."""

    timestamps: np.ndarray
    prices: np.ndarray

    @property
    def is_empty(self) -> bool:
        return self.timestamps.size == 0


def parse_history_point(point: dict) -> tuple[int, float] | None:
    """Parse a history point into unix seconds + price, filtering invalid rows."""
    t = point.get("t", point.get("timestamp"))
    p = point.get("p", point.get("price"))
    if t is None or p is None:
        return None

    try:
        if isinstance(t, (int, float)):
            ts = int(pd.to_datetime(t, unit="s", utc=True).timestamp())
        else:
            ts = int(pd.to_datetime(t, utc=True).timestamp())
        price = float(p)
    except (ValueError, TypeError, OverflowError):
        return None

    if ts <= 0 or not (0.0 < price < 1.0):
        return None
    return ts, price


def prepare_price_history(price_history: Iterable[dict] | None) -> PreparedHistory:
    """Normalize and sort a raw price history."""
    if not price_history:
        return PreparedHistory(
            timestamps=np.array([], dtype=np.int64),
            prices=np.array([], dtype=np.float32),
        )

    rows: list[tuple[int, float]] = []
    for point in price_history:
        parsed = parse_history_point(point)
        if parsed is not None:
            rows.append(parsed)

    if not rows:
        return PreparedHistory(
            timestamps=np.array([], dtype=np.int64),
            prices=np.array([], dtype=np.float32),
        )

    rows.sort(key=lambda item: item[0])

    dedup_ts: list[int] = []
    dedup_prices: list[float] = []
    for ts, price in rows:
        if dedup_ts and ts == dedup_ts[-1]:
            dedup_prices[-1] = price
        else:
            dedup_ts.append(ts)
            dedup_prices.append(price)

    return PreparedHistory(
        timestamps=np.asarray(dedup_ts, dtype=np.int64),
        prices=np.asarray(dedup_prices, dtype=np.float32),
    )


def get_points_until(
    history: PreparedHistory,
    snapshot_time: pd.Timestamp | int,
) -> PreparedHistory:
    """Return history points up to and including a snapshot timestamp."""
    if history.is_empty:
        return history

    snapshot_ts = _to_unix_seconds(snapshot_time)
    idx = int(np.searchsorted(history.timestamps, snapshot_ts, side="right"))
    return PreparedHistory(
        timestamps=history.timestamps[:idx],
        prices=history.prices[:idx],
    )


def latest_price_before(
    history: PreparedHistory,
    snapshot_time: pd.Timestamp | int,
) -> float | None:
    """Resolve the latest observed price at or before the snapshot time."""
    sliced = get_points_until(history, snapshot_time)
    if sliced.is_empty:
        return None
    return float(sliced.prices[-1])


def latest_timestamp_before(
    history: PreparedHistory,
    snapshot_time: pd.Timestamp | int,
) -> int | None:
    """Resolve the latest observed timestamp at or before the snapshot time."""
    sliced = get_points_until(history, snapshot_time)
    if sliced.is_empty:
        return None
    return int(sliced.timestamps[-1])


def build_snapshot_samples(
    market: dict,
    history: PreparedHistory,
    horizons_days: Iterable[int],
    min_history_points: int = 2,
) -> list[dict]:
    """Generate one snapshot sample per valid horizon for a resolved market."""
    if history.is_empty:
        return []

    end_time = get_market_end_time(market)
    if end_time is None:
        return []

    resolution = infer_resolution_from_market(market)
    if resolution not in {"yes", "no"}:
        return []

    created_at = _parse_market_time(market.get("createdAt"))
    label_yes = 1 if resolution == "yes" else 0
    market_id = str(market.get("id", ""))
    samples: list[dict] = []

    for horizon in sorted({int(h) for h in horizons_days if int(h) > 0}):
        snapshot_time = end_time - pd.Timedelta(days=horizon)
        history_until_snapshot = get_points_until(history, snapshot_time)
        if history_until_snapshot.timestamps.size < min_history_points:
            continue

        if created_at is not None and snapshot_time <= created_at:
            continue

        samples.append({
            "market_id": market_id,
            "market": market,
            "snapshot_time": snapshot_time,
            "snapshot_ts": int(snapshot_time.timestamp()),
            "end_time": end_time,
            "end_ts": int(end_time.timestamp()),
            "created_at": created_at,
            "created_ts": int(created_at.timestamp()) if created_at is not None else None,
            "days_to_end": int(horizon),
            "snapshot_price_yes": float(history_until_snapshot.prices[-1]),
            "label_yes": int(label_yes),
            "target_residual": float(label_yes - float(history_until_snapshot.prices[-1])),
            "history": history_until_snapshot,
        })

    return samples


def prepare_history_map(price_histories: dict | None) -> dict[str, PreparedHistory]:
    """Prepare all market histories once for repeated snapshot generation."""
    histories = price_histories or {}
    prepared: dict[str, PreparedHistory] = {}
    for market_id, history in histories.items():
        prepared[str(market_id)] = prepare_price_history(history)
    return prepared


def _parse_market_time(value) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        return pd.to_datetime(value, utc=True)
    except (ValueError, TypeError):
        return None


def _to_unix_seconds(value: pd.Timestamp | int) -> int:
    if isinstance(value, pd.Timestamp):
        return int(value.timestamp())
    return int(value)
