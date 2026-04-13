import unittest

import torch

from src.features.numerical import NUM_NUMERICAL_FEATURES
from src.model.ts_architecture import PriceSequenceGRU


class PriceSequenceGRUTests(unittest.TestCase):
    def test_forward_returns_one_logit_per_sequence(self):
        model = PriceSequenceGRU(
            input_dim=3,
            static_num_features=NUM_NUMERICAL_FEATURES,
            num_categories=4,
            category_embed_dim=6,
            text_embed_dim=16,
            hidden_dim=16,
            static_hidden_dim=12,
            num_layers=1,
            dropout=0.2,
        )
        sequences = torch.randn(4, 8, 3)
        lengths = torch.LongTensor([8, 6, 5, 3])
        static_numerical = torch.randn(4, NUM_NUMERICAL_FEATURES)
        category_ids = torch.LongTensor([0, 1, 2, 3])
        text_embeddings = torch.randn(4, 16)

        scores = model(
            sequences,
            lengths,
            static_numerical,
            category_ids,
            text_embeddings,
        )

        self.assertEqual(tuple(scores.shape), (4,))
        self.assertTrue(torch.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
