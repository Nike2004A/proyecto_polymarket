import unittest

import numpy as np
import pandas as pd
import torch

from src.data.snapshots import prepare_price_history
from src.features.numerical import NUM_NUMERICAL_FEATURES
from src.scoring.scorer import score_active_markets


class ConstantTabularModel(torch.nn.Module):
    def __init__(self, logit: float = 1.2):
        super().__init__()
        self.logit = logit

    def eval(self):
        return self

    def to(self, device):
        return self

    def forward(self, numerical, category, text_embedding):
        batch_size = numerical.shape[0]
        return torch.full((batch_size,), self.logit, dtype=torch.float32, device=numerical.device)


class ConstantCalibrator:
    def __init__(self, probability: float):
        self.probability = probability

    def predict_proba(self, logits):
        logits = np.asarray(logits)
        return np.full(logits.shape[0], self.probability, dtype=np.float64)


class SpyFeaturePipeline:
    def __init__(self):
        self.calls = []

    def warm_text_cache(self, markets):
        return None

    def transform_single(self, market, price_history=None, snapshot_time=None, days_to_end=None):
        self.calls.append(
            {
                "market_id": market["id"],
                "price_history": price_history,
                "snapshot_time": snapshot_time,
                "days_to_end": days_to_end,
            }
        )
        return {
            "numerical": np.zeros(NUM_NUMERICAL_FEATURES, dtype=np.float32),
            "category_id": 0,
            "text_embedding": np.zeros(16, dtype=np.float32),
        }


class ScoringContextTests(unittest.TestCase):
    def setUp(self):
        self.market = {
            "id": "m1",
            "question": "Will X happen?",
            "outcomePrices": [0.42, 0.58],
            "volume24hr": 1500,
            "liquidity": 5000,
            "spread": 0.03,
            "slug": "will-x-happen",
            "endDate": "2026-01-10T00:00:00Z",
        }
        self.snapshot_time = pd.Timestamp("2026-01-09T00:00:00Z")
        self.history = prepare_price_history([
            {"t": "2026-01-07T00:00:00Z", "p": 0.30},
            {"t": "2026-01-09T00:00:00Z", "p": 0.40},
            {"t": "2026-01-10T00:00:00Z", "p": 0.95},
        ])
        self.bundle = {
            "model_name": "market_value_baseline",
            "model": ConstantTabularModel(),
            "pipeline": SpyFeaturePipeline(),
            "calibrator": ConstantCalibrator(0.80),
            "run_config": {},
        }

    def test_scoring_uses_snapshot_history_and_emits_ev_columns(self):
        df = score_active_markets(
            self.bundle,
            active_markets=[self.market],
            price_histories={"m1": self.history},
            snapshot_time=self.snapshot_time,
            top_k=5,
        )

        self.assertEqual(len(df), 1)
        self.assertEqual(len(self.bundle["pipeline"].calls), 1)
        self.assertIs(self.bundle["pipeline"].calls[0]["price_history"], self.history)
        self.assertEqual(self.bundle["pipeline"].calls[0]["snapshot_time"], self.snapshot_time)
        self.assertEqual(self.bundle["pipeline"].calls[0]["days_to_end"], 1.0)

        row = df.iloc[0]
        self.assertEqual(row["model_name"], "market_value_baseline")
        self.assertAlmostEqual(float(row["price_yes"]), 0.40, places=6)
        self.assertAlmostEqual(float(row["p_yes_calibrated"]), 0.80, places=6)
        self.assertAlmostEqual(float(row["ev_per_share"]), 0.40, places=6)
        self.assertAlmostEqual(float(row["expected_roi"]), 1.0, places=6)
        self.assertAlmostEqual(float(row["days_to_end"]), 1.0, places=6)
        self.assertEqual(row["signal"], "STRONG BUY")
        for required in [
            "p_yes_raw",
            "p_yes_calibrated",
            "ev_per_share",
            "expected_roi",
            "days_to_end",
            "signal",
        ]:
            self.assertIn(required, df.columns)


if __name__ == "__main__":
    unittest.main()
