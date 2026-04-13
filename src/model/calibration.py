"""Probability calibration utilities for p_yes models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    """Wrapper around a fitted Platt or Isotonic calibrator."""

    def __init__(self, method: str, model, input_kind: str = "logit"):
        self.method = method
        self.model = model
        self.input_kind = input_kind

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if self.input_kind == "probability":
            probs = np.clip(values, 1e-6, 1 - 1e-6)
            if self.method == "identity":
                return probs
            if self.method == "platt":
                return self.model.predict_proba(probs.reshape(-1, 1))[:, 1]
            if self.method == "isotonic":
                return self.model.predict(probs)
            raise ValueError(f"Método de calibración no soportado: {self.method}")

        logits = values
        if self.method == "identity":
            return 1.0 / (1.0 + np.exp(-logits))
        if self.method == "platt":
            return self.model.predict_proba(logits.reshape(-1, 1))[:, 1]
        if self.method == "isotonic":
            return self.model.predict(logits)
        raise ValueError(f"Método de calibración no soportado: {self.method}")

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            joblib.dump(self.model, path / "calibrator.pkl")
        with open(path / "calibration_metadata.json", "w", encoding="utf-8") as f:
            json.dump({"method": self.method, "input_kind": self.input_kind}, f, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> "ProbabilityCalibrator":
        path = Path(directory)
        with open(path / "calibration_metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)
        model = None
        model_path = path / "calibrator.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
        return cls(metadata["method"], model, metadata.get("input_kind", "logit"))


def identity_calibrator(input_kind: str = "logit") -> ProbabilityCalibrator:
    return ProbabilityCalibrator("identity", None, input_kind=input_kind)


def fit_platt_calibrator(values: np.ndarray, labels: np.ndarray, input_kind: str = "logit") -> ProbabilityCalibrator:
    values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    model = LogisticRegression(max_iter=1000)
    model.fit(values, labels)
    return ProbabilityCalibrator("platt", model, input_kind=input_kind)


def fit_isotonic_calibrator(values: np.ndarray, labels: np.ndarray, input_kind: str = "logit") -> ProbabilityCalibrator:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(values, labels)
    return ProbabilityCalibrator("isotonic", model, input_kind=input_kind)
