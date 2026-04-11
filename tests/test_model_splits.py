import unittest

import numpy as np

from src.model.splits import build_dataset_split


class DatasetSplitTests(unittest.TestCase):
    def test_random_split_covers_all_indices_without_overlap(self):
        labels = np.array([0, 1] * 10, dtype=np.int64)

        split = build_dataset_split(
            n_samples=len(labels),
            labels=labels,
            val_split=0.2,
            test_split=0.2,
            strategy="random",
            seed=42,
        )

        merged = split.train_indices + split.val_indices + split.test_indices
        self.assertEqual(len(split.train_indices), 12)
        self.assertEqual(len(split.val_indices), 4)
        self.assertEqual(len(split.test_indices), 4)
        self.assertEqual(len(set(merged)), len(labels))
        self.assertEqual(sorted(merged), list(range(len(labels))))

    def test_temporal_split_respects_time_order(self):
        timestamps = np.array([50, 10, 40, 20, 30, 60], dtype=np.int64)

        split = build_dataset_split(
            n_samples=len(timestamps),
            labels=np.array([0, 1, 0, 1, 0, 1], dtype=np.int64),
            timestamps=timestamps,
            val_split=0.2,
            test_split=0.2,
            strategy="temporal",
            seed=123,
        )

        self.assertEqual(split.train_indices, [1, 3, 4, 2])
        self.assertEqual(split.val_indices, [0])
        self.assertEqual(split.test_indices, [5])
        self.assertEqual(
            split.temporal_cutoffs,
            {"val_start_timestamp": 50, "test_start_timestamp": 60},
        )


if __name__ == "__main__":
    unittest.main()
