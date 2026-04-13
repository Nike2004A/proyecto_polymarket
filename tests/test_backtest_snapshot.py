import unittest

import numpy as np
import pandas as pd

from src.data.snapshots import build_snapshot_samples, prepare_price_history
from src.features.numerical import NUMERICAL_FEATURE_NAMES, extract_numerical_features


class SnapshotSampleTests(unittest.TestCase):
    def test_tabular_feature_set_excludes_live_only_fields(self):
        forbidden = {
            "price_no",
            "volume_24h",
            "volume_total",
            "liquidity",
            "volume_liquidity_ratio",
            "best_bid",
            "best_ask",
            "spread",
            "bid_depth",
            "ask_depth",
            "book_imbalance",
        }
        self.assertTrue(forbidden.isdisjoint(set(NUMERICAL_FEATURE_NAMES)))

    def test_snapshot_generator_uses_only_points_before_each_horizon(self):
        market = {
            "id": "m1",
            "createdAt": "2026-01-01T00:00:00Z",
            "endDate": "2026-01-10T00:00:00Z",
            "outcomePrices": [1.0, 0.0],
        }
        history = prepare_price_history([
            {"t": "2026-01-01T00:00:00Z", "p": 0.20},
            {"t": "2026-01-03T00:00:00Z", "p": 0.40},
            {"t": "2026-01-09T00:00:00Z", "p": 0.90},
            {"t": "2026-01-10T00:00:00Z", "p": 0.99},
        ])

        samples = build_snapshot_samples(market, history, horizons_days=[1, 7], min_history_points=2)
        by_horizon = {sample["days_to_end"]: sample for sample in samples}

        self.assertAlmostEqual(by_horizon[7]["snapshot_price_yes"], 0.40, places=6)
        self.assertAlmostEqual(by_horizon[1]["snapshot_price_yes"], 0.90, places=6)
        self.assertEqual(by_horizon[7]["label_yes"], 1)

    def test_temporal_features_use_snapshot_time_not_now(self):
        market = {
            "id": "m1",
            "createdAt": "2026-01-01T00:00:00Z",
            "endDate": "2026-01-10T00:00:00Z",
            "outcomePrices": [0.40, 0.60],
            "negRisk": False,
        }
        snapshot_time = pd.Timestamp("2026-01-03T00:00:00Z")
        history = prepare_price_history([
            {"t": "2026-01-01T00:00:00Z", "p": 0.20},
            {"t": "2026-01-03T00:00:00Z", "p": 0.40},
        ])

        features = extract_numerical_features(
            market,
            price_history=history,
            snapshot_time=snapshot_time,
            snapshot_price_yes=0.40,
            days_to_end=7,
        )

        idx_days_to_end = NUMERICAL_FEATURE_NAMES.index("days_to_end")
        idx_market_age = NUMERICAL_FEATURE_NAMES.index("market_age_days_at_snapshot")
        idx_last_trade = NUMERICAL_FEATURE_NAMES.index("days_since_last_trade")

        self.assertAlmostEqual(float(features[idx_days_to_end]), 7.0, places=4)
        self.assertAlmostEqual(float(features[idx_market_age]), 2.0, places=4)
        self.assertAlmostEqual(float(features[idx_last_trade]), 0.0, places=4)

    def test_calendar_returns_do_not_depend_on_tick_count(self):
        market = {
            "id": "m1",
            "createdAt": "2026-01-01T00:00:00Z",
            "endDate": "2026-01-20T00:00:00Z",
            "outcomePrices": [0.50, 0.50],
        }
        snapshot_time = pd.Timestamp("2026-01-10T00:00:00Z")
        sparse = prepare_price_history([
            {"t": "2026-01-03T00:00:00Z", "p": 0.25},
            {"t": "2026-01-10T00:00:00Z", "p": 0.50},
        ])
        dense = prepare_price_history([
            {"t": "2026-01-03T00:00:00Z", "p": 0.25},
            {"t": "2026-01-04T00:00:00Z", "p": 0.30},
            {"t": "2026-01-05T00:00:00Z", "p": 0.35},
            {"t": "2026-01-06T00:00:00Z", "p": 0.40},
            {"t": "2026-01-07T00:00:00Z", "p": 0.45},
            {"t": "2026-01-10T00:00:00Z", "p": 0.50},
        ])

        sparse_features = extract_numerical_features(
            market,
            price_history=sparse,
            snapshot_time=snapshot_time,
            snapshot_price_yes=0.50,
            days_to_end=10,
        )
        dense_features = extract_numerical_features(
            market,
            price_history=dense,
            snapshot_time=snapshot_time,
            snapshot_price_yes=0.50,
            days_to_end=10,
        )

        idx_return_7d = NUMERICAL_FEATURE_NAMES.index("return_7d")
        self.assertAlmostEqual(float(sparse_features[idx_return_7d]), 1.0, places=6)
        self.assertAlmostEqual(float(dense_features[idx_return_7d]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
