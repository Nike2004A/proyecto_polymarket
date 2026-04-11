"""Helpers para métricas de clasificación y tuning de umbral."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
    digits: int = 4,
) -> dict:
    """Calcula métricas binarias a partir de scores continuos y un umbral."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    preds = (scores >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "pr_auc": average_precision_score(labels, scores)
        if len(np.unique(labels)) > 1
        else 0.0,
        "roc_auc": roc_auc_score(labels, scores)
        if len(np.unique(labels)) > 1
        else 0.0,
        "confusion_matrix": confusion_matrix(labels, preds),
        "classification_report": classification_report(
            labels,
            preds,
            target_names=["No Buy", "Buy"],
            digits=digits,
        ),
        "labels": labels,
        "scores": scores,
        "predictions": preds,
    }
    return metrics


def find_best_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    objective: str = "f1",
) -> dict:
    """Encuentra el mejor umbral sobre validación según el objetivo indicado."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)

    if len(np.unique(labels)) < 2:
        return {
            "threshold": 0.5,
            "objective": objective,
            "objective_value": 0.0,
        }

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return {
            "threshold": 0.5,
            "objective": objective,
            "objective_value": 0.0,
        }

    precision = precision[:-1]
    recall = recall[:-1]

    if objective == "f1":
        denom = precision + recall
        values = np.divide(
            2 * precision * recall,
            denom,
            out=np.zeros_like(denom),
            where=denom > 0,
        )
    elif objective == "precision":
        values = precision
    elif objective == "recall":
        values = recall
    else:
        raise ValueError(f"Objetivo de umbral no soportado: {objective}")

    best_idx = int(np.nanargmax(values))
    return {
        "threshold": float(thresholds[best_idx]),
        "objective": objective,
        "objective_value": float(values[best_idx]),
        "precision_at_threshold": float(precision[best_idx]),
        "recall_at_threshold": float(recall[best_idx]),
    }
