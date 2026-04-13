import unittest

import numpy as np
import pandas as pd

from src.features.ts_sequence import build_grid_sequence


class TimeSeriesGridSequenceTests(unittest.TestCase):
    def test_grid_sequence_uses_ffill_and_observed_mask(self):
        history = [
            {"t": "2026-01-01T00:00:00Z", "p": 0.20},
            {"t": "2026-01-01T18:00:00Z", "p": 0.40},
            {"t": "2026-01-02T18:00:00Z", "p": 0.50},
        ]
        snapshot_time = pd.Timestamp("2026-01-03T00:00:00Z")

        sequence, length, snapshot_price = build_grid_sequence(
            history,
            snapshot_time=snapshot_time,
            lookback_days=2,
            step_hours=12,
        )

        self.assertEqual(length, 4)
        self.assertAlmostEqual(snapshot_price, 0.50, places=6)
        np.testing.assert_allclose(sequence[:, 0], np.array([0.20, 0.40, 0.40, 0.50], dtype=np.float32))
        np.testing.assert_allclose(sequence[:, 1], np.array([0.0, 0.20, 0.0, 0.10], dtype=np.float32))
        np.testing.assert_allclose(sequence[:, 2], np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
