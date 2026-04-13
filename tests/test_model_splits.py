import unittest

import numpy as np

from src.model.splits import build_dataset_split


class DatasetSplitTests(unittest.TestCase):
    def test_grouped_temporal_split_keeps_market_ids_together(self):
        timestamps = np.array([10, 10, 20, 20, 30, 30, 40, 40], dtype=np.int64)
        groups = np.array(["m1", "m1", "m2", "m2", "m3", "m3", "m4", "m4"], dtype=object)
        labels = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)

        split = build_dataset_split(
            n_samples=len(labels),
            labels=labels,
            timestamps=timestamps,
            groups=groups,
            val_split=0.25,
            test_split=0.25,
            strategy="temporal_grouped",
            seed=42,
        )

        train_groups = set(groups[split.train_indices])
        val_groups = set(groups[split.val_indices])
        test_groups = set(groups[split.test_indices])

        self.assertTrue(train_groups.isdisjoint(val_groups))
        self.assertTrue(train_groups.isdisjoint(test_groups))
        self.assertTrue(val_groups.isdisjoint(test_groups))
        self.assertEqual(split.temporal_cutoffs, {"val_start_timestamp": 30, "test_start_timestamp": 40})


if __name__ == "__main__":
    unittest.main()
