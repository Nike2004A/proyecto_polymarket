import math
import unittest

import numpy as np

from src.data.preprocessing import get_snapshot_cutoff_time
from src.features.ts_sequence import (
    build_live_price_sequence,
    build_market_price_sequence,
)
from src.model.ts_dataset import collate_ts_batch


class TimeSeriesSequenceTests(unittest.TestCase):
    def test_market_sequence_respects_snapshot_cutoff_and_left_padding(self):
        market = {
            "id": "m1",
            "endDate": "2026-01-10T00:00:00Z",
        }
        price_histories = {
            "m1": [
                {"t": "2026-01-01T00:00:00Z", "p": 0.20},
                {"t": "2026-01-02T12:00:00Z", "p": 0.35},
                {"t": "2026-01-03T00:00:00Z", "p": 0.40},
                {"t": "2026-01-04T00:00:00Z", "p": 0.90},
            ]
        }

        cutoff = get_snapshot_cutoff_time(market, snapshot_offset_days=7)
        self.assertEqual(str(cutoff), "2026-01-03 00:00:00+00:00")

        sequence, length, snapshot_price, _ = build_market_price_sequence(
            market,
            price_histories,
            snapshot_offset_days=7,
            seq_len=5,
            min_points=2,
        )

        self.assertEqual(length, 3)
        self.assertAlmostEqual(snapshot_price, 0.40, places=6)
        np.testing.assert_allclose(sequence[:2], 0.0)
        np.testing.assert_allclose(sequence[-3:, 0], np.array([0.20, 0.35, 0.40]))
        self.assertAlmostEqual(sequence[-2, 1], 0.15, places=6)
        expected_delta_time = math.log1p(36.0) / math.log1p(168.0)
        self.assertAlmostEqual(sequence[-2, 2], expected_delta_time, places=6)

    def test_sequence_builder_requires_minimum_points(self):
        sequence, length, snapshot_price = build_live_price_sequence(
            [{"t": "2026-01-01T00:00:00Z", "p": 0.5}],
            seq_len=4,
            min_points=2,
        )
        self.assertIsNone(sequence)
        self.assertIsNone(length)
        self.assertIsNone(snapshot_price)

    def test_collate_converts_left_padded_sequences_to_right_padded_batch(self):
        seq_a = np.zeros((5, 3), dtype=np.float32)
        seq_a[-3:] = np.array(
            [[0.2, 0.0, 0.0], [0.3, 0.1, 0.1], [0.4, 0.1, 0.1]],
            dtype=np.float32,
        )
        seq_b = np.zeros((5, 3), dtype=np.float32)
        seq_b[-2:] = np.array(
            [[0.5, 0.0, 0.0], [0.45, -0.05, 0.1]],
            dtype=np.float32,
        )

        batch = collate_ts_batch([
            {"sequence": torch_tensor(seq_a), "sequence_length": torch_long(3), "label": torch_float(1.0)},
            {"sequence": torch_tensor(seq_b), "sequence_length": torch_long(2), "label": torch_float(0.0)},
        ])

        np.testing.assert_allclose(
            batch["sequence"][0, :3].numpy(),
            seq_a[-3:],
        )
        np.testing.assert_allclose(
            batch["sequence"][1, :2].numpy(),
            seq_b[-2:],
        )
        np.testing.assert_allclose(batch["sequence"][0, 3:].numpy(), 0.0)
        np.testing.assert_allclose(batch["sequence"][1, 2:].numpy(), 0.0)


def torch_tensor(array):
    import torch

    return torch.FloatTensor(array)


def torch_long(value):
    import torch

    return torch.LongTensor([value]).squeeze(0)


def torch_float(value):
    import torch

    return torch.FloatTensor([value]).squeeze(0)


if __name__ == "__main__":
    unittest.main()
