import unittest

import numpy as np
import torch

from src.scoring.scorer import score_active_markets


class ConstantScoreModel:
    task = "classification"

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, numerical, category, text_embedding):
        return torch.tensor([0.7], dtype=torch.float32)


class SpyFeaturePipeline:
    def __init__(self):
        self.calls = []

    def transform_single(self, market, price_history=None, order_book=None):
        self.calls.append(
            {
                "market_id": market["id"],
                "price_history": price_history,
                "order_book": order_book,
            }
        )
        return {
            "numerical": np.zeros(23, dtype=np.float32),
            "category_id": 0,
            "text_embedding": np.zeros(384, dtype=np.float32),
        }


class FakeClient:
    def __init__(self, markets):
        self.markets = markets
        self.history_calls = []
        self.book_calls = []

    def get_all_active_markets(self, max_markets=1000):
        return self.markets[:max_markets]

    def parse_market(self, market):
        return market

    def get_price_history(self, token_id):
        self.history_calls.append(token_id)
        return [{"t": 1, "p": 0.4}, {"t": 2, "p": 0.5}]

    def get_order_book(self, token_id):
        self.book_calls.append(token_id)
        return {"bids": [{"s": 10}], "asks": [{"s": 5}]}


class ScoringContextTests(unittest.TestCase):
    def setUp(self):
        self.market = {
            "id": "m1",
            "question": "Will X happen?",
            "clobTokenIds": ["token-1"],
            "outcomePrices": [0.42, 0.58],
            "volume24hr": 1500,
            "liquidity": 5000,
            "spread": 0.03,
            "slug": "will-x-happen",
            "endDate": "2026-01-10T00:00:00Z",
        }
        self.model = ConstantScoreModel()

    def test_scoring_uses_cached_history_and_order_book_when_present(self):
        client = FakeClient([self.market])
        pipeline = SpyFeaturePipeline()

        df = score_active_markets(
            self.model,
            client,
            pipeline,
            price_histories={"m1": [{"t": 1, "p": 0.4}]},
            order_books={"m1": {"bids": [{"s": 1}], "asks": [{"s": 2}]}},
            fetch_missing_context=False,
            max_markets=1,
        )

        self.assertEqual(len(df), 1)
        self.assertEqual(len(pipeline.calls), 1)
        self.assertEqual(pipeline.calls[0]["price_history"], [{"t": 1, "p": 0.4}])
        self.assertEqual(
            pipeline.calls[0]["order_book"],
            {"bids": [{"s": 1}], "asks": [{"s": 2}]},
        )
        self.assertEqual(client.history_calls, [])
        self.assertEqual(client.book_calls, [])

    def test_scoring_fetches_missing_context_when_cache_is_absent(self):
        client = FakeClient([self.market])
        pipeline = SpyFeaturePipeline()

        df = score_active_markets(
            self.model,
            client,
            pipeline,
            price_histories={},
            order_books={},
            fetch_missing_context=True,
            max_markets=1,
        )

        self.assertEqual(len(df), 1)
        self.assertEqual(len(pipeline.calls), 1)
        self.assertEqual(
            pipeline.calls[0]["price_history"],
            [{"t": 1, "p": 0.4}, {"t": 2, "p": 0.5}],
        )
        self.assertEqual(
            pipeline.calls[0]["order_book"],
            {"bids": [{"s": 10}], "asks": [{"s": 5}]},
        )
        self.assertEqual(client.history_calls, ["token-1"])
        self.assertEqual(client.book_calls, ["token-1"])


if __name__ == "__main__":
    unittest.main()
