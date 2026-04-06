import unittest

import torch

from src.model.ts_architecture import PriceSequenceGRU


class PriceSequenceGRUTests(unittest.TestCase):
    def test_forward_returns_one_score_per_sequence(self):
        model = PriceSequenceGRU(input_dim=3, hidden_dim=16, num_layers=1, dropout=0.2)
        sequences = torch.randn(4, 8, 3)
        lengths = torch.LongTensor([8, 6, 5, 3])

        scores = model(sequences, lengths)

        self.assertEqual(tuple(scores.shape), (4,))
        self.assertTrue(torch.all(scores >= 0))
        self.assertTrue(torch.all(scores <= 1))


if __name__ == "__main__":
    unittest.main()
