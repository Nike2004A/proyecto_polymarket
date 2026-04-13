import unittest

import numpy as np
import pandas as pd
import torch

from src.data.snapshots import prepare_price_history
from src.features.numerical import NUM_NUMERICAL_FEATURES
from src.scoring.scorer import score_active_markets


class ConstantTSModel(torch.nn.Module):
    def __init__(self, logit: float = 0.9):
        super().__init__()
        self.logit = logit

    def eval(self):
        return self

    def to(self, device):
        return self

    def forward(self, sequences, lengths, static_numerical, category_ids, text_embeddings):
        batch_size = sequences.shape[0]
        return torch.full((batch_size,), self.logit, dtype=torch.float32, device=sequences.device)


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
        self.calls.append({
            "market_id": market["id"],
            "price_history": price_history,
            "snapshot_time": snapshot_time,
            "days_to_end": days_to_end,
        })
        return {
            "numerical": np.zeros(NUM_NUMERICAL_FEATURES, dtype=np.float32),
            "category_id": 0,
            "text_embedding": np.zeros(16, dtype=np.float32),
        }


class TimeSeriesScorerTests(unittest.TestCase):
    def setUp(self):
        self.markets = [
            {
                "id": "m1",
                "question": "Will X happen?",
                "clobTokenIds": ["token-1"],
                "outcomePrices": [0.42, 0.58],
                "volume24hr": 1500,
                "liquidity": 5000,
                "spread": 0.03,
                "slug": "will-x-happen",
                "endDate": "2026-01-10T00:00:00Z",
            },
            {
                "id": "m2",
                "question": "Will Y happen?",
                "outcomePrices": [0.61, 0.39],
                "volume24hr": 800,
                "liquidity": 2000,
                "spread": 0.04,
                "slug": "will-y-happen",
                "endDate": "2026-01-11T00:00:00Z",
            },
        ]
        self.snapshot_time = pd.Timestamp("2026-01-09T00:00:00Z")
        self.pipeline = SpyFeaturePipeline()
        self.bundle = {
            "model_name": "price_sequence_gru",
            "model": ConstantTSModel(),
            "pipeline": self.pipeline,
            "calibrator": ConstantCalibrator(0.78),
            "run_config": {"dataset_metadata": {"sequence_lookback_days": 60, "sequence_grid_hours": 24}},
        }

    def test_ts_bundle_scores_active_markets_and_skips_missing_history(self):
        histories = {
            "m1": prepare_price_history([
                {"t": "2025-12-20T00:00:00Z", "p": 0.30},
                {"t": "2025-12-28T00:00:00Z", "p": 0.32},
                {"t": "2026-01-03T00:00:00Z", "p": 0.35},
                {"t": "2026-01-08T12:00:00Z", "p": 0.40},
            ]),
        }

        df = score_active_markets(
            self.bundle,
            active_markets=self.markets,
            price_histories=histories,
            snapshot_time=self.snapshot_time,
            top_k=10,
        )

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["id"], "m1")
        self.assertAlmostEqual(float(df.iloc[0]["price_yes"]), 0.40, places=6)
        self.assertAlmostEqual(float(df.iloc[0]["p_yes_calibrated"]), 0.78, places=6)
        self.assertEqual(df.iloc[0]["signal"], "STRONG BUY")
        self.assertEqual(len(self.pipeline.calls), 2)
        self.assertEqual(self.pipeline.calls[0]["days_to_end"], 1.0)


if __name__ == "__main__":
    unittest.main()
