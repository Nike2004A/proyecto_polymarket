"""Probability, calibration, and EV-oriented evaluation helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-logits))


def compute_binary_classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.clip(np.asarray(scores, dtype=np.float64), 1e-6, 1 - 1e-6)
    predictions = (scores >= threshold).astype(np.int64)
    metrics = {
        "threshold": float(threshold),
        "accuracy": float((predictions == labels).mean()) if labels.size else 0.0,
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "predictions": predictions,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]),
    }
    if np.unique(labels).size > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        metrics["pr_auc"] = float(average_precision_score(labels, scores))
        metrics["auc_roc"] = metrics["roc_auc"]
    else:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0
        metrics["auc_roc"] = 0.0
    return metrics


def compute_probability_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1 - 1e-6)
    metrics = {
        "brier": float(brier_score_loss(labels, probs)),
        "log_loss": float(log_loss(labels, probs)),
        "positive_rate": float(labels.mean()) if labels.size else 0.0,
        "mean_probability": float(probs.mean()) if probs.size else 0.0,
        "ece": float(expected_calibration_error(labels, probs)),
    }
    if np.unique(labels).size > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
        metrics["pr_auc"] = float(average_precision_score(labels, probs))
    else:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0
    return metrics


def clip_probabilities_from_residual(
    predictions: np.ndarray,
    market_prices: np.ndarray,
) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=np.float64)
    market_prices = np.asarray(market_prices, dtype=np.float64)
    return np.clip(market_prices + predictions, 1e-6, 1 - 1e-6)


def compute_residual_metrics(
    labels: np.ndarray,
    market_prices: np.ndarray,
    residual_predictions: np.ndarray,
) -> dict:
    labels = np.asarray(labels, dtype=np.float64)
    market_prices = np.asarray(market_prices, dtype=np.float64)
    residual_predictions = np.asarray(residual_predictions, dtype=np.float64)
    target_residual = labels - market_prices
    return {
        "mae_residual": float(mean_absolute_error(target_residual, residual_predictions)),
        "mean_target_residual": float(target_residual.mean()) if target_residual.size else 0.0,
        "mean_predicted_residual": float(residual_predictions.mean()) if residual_predictions.size else 0.0,
    }


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    if probs.size == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        if right == 1.0:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        if not mask.any():
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.mean() * abs(bin_acc - bin_conf)
    return float(ece)


def compute_ev_metrics(
    labels: np.ndarray,
    market_prices: np.ndarray,
    calibrated_probs: np.ndarray,
    top_k: int = 20,
    roi_cap: float | None = None,
) -> dict:
    labels = np.asarray(labels, dtype=np.float64)
    market_prices = np.asarray(market_prices, dtype=np.float64)
    calibrated_probs = np.asarray(calibrated_probs, dtype=np.float64)

    ev_per_share = calibrated_probs - market_prices
    safe_prices = np.clip(market_prices, 1e-6, None)
    expected_roi = ev_per_share / safe_prices
    expected_roi_capped = np.clip(expected_roi, -roi_cap, roi_cap) if roi_cap is not None else expected_roi
    realized_pnl = labels - market_prices
    realized_roi = realized_pnl / safe_prices

    order = np.argsort(-ev_per_share)
    positive_order = [idx for idx in order.tolist() if ev_per_share[idx] > 0]
    top_order = np.asarray(positive_order[:top_k], dtype=np.int64)

    metrics = {
        "num_positive_ev": int((ev_per_share > 0).sum()),
        "mean_predicted_ev": float(ev_per_share.mean()),
        "mean_realized_pnl": float(realized_pnl.mean()),
        "mean_expected_roi": float(expected_roi.mean()),
        "mean_expected_roi_capped": float(expected_roi_capped.mean()),
    }

    if top_order.size == 0:
        metrics.update({
            "top_k": int(top_k),
            "top_k_count": 0,
            "top_k_avg_predicted_ev": 0.0,
            "top_k_avg_realized_pnl": 0.0,
            "top_k_avg_expected_roi": 0.0,
            "top_k_avg_expected_roi_capped": 0.0,
            "top_k_avg_realized_roi": 0.0,
            "top_k_hit_rate": 0.0,
        })
        return metrics

    metrics.update({
        "top_k": int(top_k),
        "top_k_count": int(top_order.size),
        "top_k_avg_predicted_ev": float(ev_per_share[top_order].mean()),
        "top_k_avg_realized_pnl": float(realized_pnl[top_order].mean()),
        "top_k_avg_expected_roi": float(expected_roi[top_order].mean()),
        "top_k_avg_expected_roi_capped": float(expected_roi_capped[top_order].mean()),
        "top_k_avg_realized_roi": float(realized_roi[top_order].mean()),
        "top_k_hit_rate": float(labels[top_order].mean()),
    })
    return metrics


def compute_bucketed_metrics(
    labels: np.ndarray,
    market_prices: np.ndarray,
    predicted_probs: np.ndarray,
    days_to_end: np.ndarray,
    top_k: int = 20,
    roi_cap: float | None = None,
) -> dict:
    labels = np.asarray(labels, dtype=np.float64)
    market_prices = np.asarray(market_prices, dtype=np.float64)
    predicted_probs = np.asarray(predicted_probs, dtype=np.float64)
    days_to_end = np.asarray(days_to_end, dtype=np.float64)

    bucket_masks = {
        "short_1_3d": (days_to_end >= 1.0) & (days_to_end <= 3.0),
        "medium_4_14d": (days_to_end > 3.0) & (days_to_end <= 14.0),
        "long_15plus": days_to_end > 14.0,
    }

    summary = {}
    for name, mask in bucket_masks.items():
        count = int(mask.sum())
        if count == 0:
            summary[name] = {"count": 0}
            continue
        summary[name] = {
            "count": count,
            "probability_metrics": compute_probability_metrics(labels[mask], predicted_probs[mask]),
            "market_baseline_metrics": compute_probability_metrics(labels[mask], market_prices[mask]),
            "ev_metrics": compute_ev_metrics(
                labels[mask],
                market_prices[mask],
                predicted_probs[mask],
                top_k=min(top_k, count),
                roi_cap=roi_cap,
            ),
        }
    return summary
