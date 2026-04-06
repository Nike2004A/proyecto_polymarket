import unittest

import torch

from src.scoring.ts_scorer import score_active_markets_ts


class ConstantTSModel:
    task = "classification"

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, sequences, lengths):
        return torch.tensor([0.82], dtype=torch.float32)


class FakeClient:
    def __init__(self, markets):
        self.markets = markets
        self.history_calls = []

    def get_all_active_markets(self, max_markets=1000):
        return self.markets[:max_markets]

    def parse_market(self, market):
        return market

    def get_price_history(self, token_id):
        self.history_calls.append(token_id)
        if token_id == "token-1":
            return [
                {"t": 1, "p": 0.30},
                {"t": 2, "p": 0.32},
                {"t": 3, "p": 0.35},
                {"t": 4, "p": 0.34},
                {"t": 5, "p": 0.36},
            ]
        return [{"t": 1, "p": 0.5}]


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
                "clobTokenIds": ["token-2"],
                "outcomePrices": [0.61, 0.39],
                "volume24hr": 800,
                "liquidity": 2000,
                "spread": 0.04,
                "slug": "will-y-happen",
                "endDate": "2026-01-11T00:00:00Z",
            },
        ]
        self.model = ConstantTSModel()

    def test_ts_scorer_omits_markets_with_insufficient_history(self):
        client = FakeClient(self.markets)
        histories = {
            "m1": [
                {"t": 1, "p": 0.30},
                {"t": 2, "p": 0.32},
                {"t": 3, "p": 0.35},
                {"t": 4, "p": 0.34},
                {"t": 5, "p": 0.36},
            ],
            "m2": [
                {"t": 1, "p": 0.60},
                {"t": 2, "p": 0.61},
            ],
        }

        df = score_active_markets_ts(
            self.model,
            client,
            price_histories=histories,
            fetch_missing_history=False,
            top_k=10,
            max_markets=2,
            min_points=5,
        )

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["id"], "m1")
        self.assertEqual(client.history_calls, [])

    def test_ts_scorer_fetches_missing_history_when_needed(self):
        client = FakeClient(self.markets)

        df = score_active_markets_ts(
            self.model,
            client,
            price_histories={},
            fetch_missing_history=True,
            top_k=10,
            max_markets=2,
            min_points=5,
        )

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["id"], "m1")
        self.assertEqual(client.history_calls, ["token-1", "token-2"])


if __name__ == "__main__":
    unittest.main()
