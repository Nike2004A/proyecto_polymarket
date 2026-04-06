"""Helpers para construir secuencias temporales de precios."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ..data.preprocessing import get_snapshot_cutoff_time

SEQUENCE_FEATURE_NAMES = [
    "price_yes",
    "delta_price",
    "delta_time_scaled",
]


def _parse_history_point(point: dict) -> tuple[pd.Timestamp, float] | None:
    """Parsea un punto del history y filtra precios inválidos."""
    t = point.get("t", point.get("timestamp"))
    p = point.get("p", point.get("price"))
    if t is None or p is None:
        return None

    try:
        if isinstance(t, (int, float)):
            ts = pd.to_datetime(t, unit="s", utc=True)
        else:
            ts = pd.to_datetime(t, utc=True)
        price = float(p)
    except (ValueError, TypeError):
        return None

    if not (0 < price < 1):
        return None
    return ts, price


def extract_sequence_points(
    price_history: Iterable[dict] | None,
    cutoff_time: pd.Timestamp | None = None,
) -> list[tuple[pd.Timestamp, float]]:
    """
    Filtra y ordena puntos válidos del historial hasta un cutoff temporal.
    """
    if not price_history:
        return []

    points: list[tuple[pd.Timestamp, float]] = []
    for point in price_history:
        parsed = _parse_history_point(point)
        if parsed is None:
            continue
        ts, price = parsed
        if cutoff_time is not None and ts > cutoff_time:
            continue
        points.append((ts, price))

    points.sort(key=lambda item: item[0])
    return points


def _delta_time_scaled(delta_hours: float) -> float:
    delta_hours = max(0.0, float(delta_hours))
    return float(np.clip(np.log1p(delta_hours) / np.log1p(168.0), 0.0, 1.0))


def build_price_sequence(
    price_history: Iterable[dict] | None,
    cutoff_time: pd.Timestamp | None = None,
    seq_len: int = 64,
    min_points: int = 5,
) -> tuple[np.ndarray | None, int | None, float | None]:
    """
    Construye una secuencia left-padded con las últimas observaciones válidas.
    """
    points = extract_sequence_points(price_history, cutoff_time=cutoff_time)
    if len(points) < min_points:
        return None, None, None

    points = points[-seq_len:]
    rows: list[list[float]] = []
    prev_time: pd.Timestamp | None = None
    prev_price: float | None = None

    for point_time, price in points:
        if prev_time is None or prev_price is None:
            delta_price = 0.0
            delta_hours = 0.0
        else:
            delta_price = float(price - prev_price)
            delta_hours = (point_time - prev_time).total_seconds() / 3600.0

        rows.append([float(price), delta_price, _delta_time_scaled(delta_hours)])
        prev_time = point_time
        prev_price = float(price)

    sequence = np.zeros((seq_len, len(SEQUENCE_FEATURE_NAMES)), dtype=np.float32)
    length = len(rows)
    sequence[-length:] = np.asarray(rows, dtype=np.float32)

    snapshot_price = float(points[-1][1])
    return sequence, length, snapshot_price


def build_market_price_sequence(
    market: dict,
    price_histories: dict | None,
    snapshot_offset_days: int = 7,
    seq_len: int = 64,
    min_points: int = 5,
) -> tuple[np.ndarray | None, int | None, float | None, pd.Timestamp | None]:
    """
    Construye la secuencia temporal de un mercado resuelto usando el cutoff TS.
    """
    market_id = str(market.get("id", ""))
    history = (price_histories or {}).get(market_id)
    cutoff_time = get_snapshot_cutoff_time(
        market,
        snapshot_offset_days=snapshot_offset_days,
    )
    if cutoff_time is None:
        return None, None, None, None

    sequence, length, snapshot_price = build_price_sequence(
        history,
        cutoff_time=cutoff_time,
        seq_len=seq_len,
        min_points=min_points,
    )
    return sequence, length, snapshot_price, cutoff_time


def build_live_price_sequence(
    price_history: Iterable[dict] | None,
    seq_len: int = 64,
    min_points: int = 5,
) -> tuple[np.ndarray | None, int | None, float | None]:
    """
    Construye una secuencia live usando toda la historia disponible hasta ahora.
    """
    return build_price_sequence(
        price_history,
        cutoff_time=None,
        seq_len=seq_len,
        min_points=min_points,
    )
