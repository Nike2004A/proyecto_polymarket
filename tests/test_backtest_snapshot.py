import unittest

import numpy as np
import torch

from src.model.evaluate import backtest


class ConstantScoreModel:
    task = "classification"

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, numerical, category, text_embedding):
        return torch.tensor([0.8], dtype=torch.float32)


class SpyFeaturePipeline:
    def __init__(self):
        self.calls = []

    def transform_single(self, market, price_history=None, order_book=None):
        self.calls.append(
            {
                "market": market,
                "price_history": price_history,
                "order_book": order_book,
            }
        )
        return {
            "numerical": np.zeros(23, dtype=np.float32),
            "category_id": 0,
            "text_embedding": np.zeros(384, dtype=np.float32),
        }


class BacktestSnapshotTests(unittest.TestCase):
    def test_backtest_uses_snapshot_price_instead_of_final_resolution_price(self):
        market = {
            "id": "m1",
            "question": "Will X happen?",
            "outcomePrices": [1.0, 0.0],
            "endDate": "2026-01-10T00:00:00Z",
        }
        price_histories = {
            "m1": [
                {"t": "2026-01-03T00:00:00Z", "p": 0.40},
                {"t": "2026-01-04T00:00:00Z", "p": 0.45},
                {"t": "2026-01-10T00:00:00Z", "p": 0.99},
            ]
        }
        pipeline = SpyFeaturePipeline()
        model = ConstantScoreModel()

        trades_df, final_capital = backtest(
            model,
            historical_markets=[market],
            feature_pipeline=pipeline,
            price_histories=price_histories,
            initial_capital=1000.0,
            position_size=0.05,
            threshold=0.6,
            snapshot_offset_days=7,
        )

        self.assertEqual(len(trades_df), 1)
        self.assertAlmostEqual(trades_df.iloc[0]["price_yes"], 0.40, places=6)
        self.assertAlmostEqual(final_capital, 1075.0, places=6)
        self.assertEqual(len(pipeline.calls), 1)
        self.assertEqual(pipeline.calls[0]["market"]["outcomePrices"], [0.40, 0.60])
        self.assertEqual(pipeline.calls[0]["price_history"], price_histories["m1"])


if __name__ == "__main__":
    unittest.main()
