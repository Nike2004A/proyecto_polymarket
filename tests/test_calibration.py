import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.model.calibration import (
    ProbabilityCalibrator,
    fit_isotonic_calibrator,
    fit_platt_calibrator,
)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.logits = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
        self.labels = np.array([0, 0, 0, 1, 1], dtype=np.int64)

    def test_platt_and_isotonic_return_valid_probabilities(self):
        for calibrator in [
            fit_platt_calibrator(self.logits, self.labels),
            fit_isotonic_calibrator(self.logits, self.labels),
        ]:
            probs = calibrator.predict_proba(self.logits)
            self.assertEqual(probs.shape, self.logits.shape)
            self.assertTrue(np.all(probs >= 0.0))
            self.assertTrue(np.all(probs <= 1.0))

    def test_calibrator_save_and_load_preserves_predictions(self):
        calibrator = fit_platt_calibrator(self.logits, self.labels)
        expected = calibrator.predict_proba(self.logits)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration"
            calibrator.save(path)
            restored = ProbabilityCalibrator.load(path)

        actual = restored.predict_proba(self.logits)
        np.testing.assert_allclose(actual, expected, atol=1e-8)

    def test_probability_input_calibrator_roundtrips(self):
        probs = np.array([0.1, 0.2, 0.4, 0.8, 0.9], dtype=np.float64)
        calibrator = fit_isotonic_calibrator(probs, self.labels, input_kind="probability")
        expected = calibrator.predict_proba(probs)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration_probs"
            calibrator.save(path)
            restored = ProbabilityCalibrator.load(path)

        actual = restored.predict_proba(probs)
        np.testing.assert_allclose(actual, expected, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
