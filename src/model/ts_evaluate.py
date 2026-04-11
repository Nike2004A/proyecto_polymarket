"""Evaluación para el modelo GRU de series de tiempo."""

from __future__ import annotations

import logging

import numpy as np
import torch

from .ts_architecture import PriceSequenceGRU
from .metrics import compute_binary_classification_metrics

logger = logging.getLogger(__name__)


def evaluate_ts_model(
    model: PriceSequenceGRU,
    data_loader,
    threshold: float = 0.5,
    device: str | None = None,
) -> dict:
    """Evalúa el modelo TS y retorna métricas + scores crudos."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    all_scores = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            sequences = batch["sequence"].to(device)
            lengths = batch["sequence_length"].to(device)
            labels = batch["label"].to(device)
            scores = model(sequences, lengths)
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    logits_arr = np.asarray(all_scores, dtype=np.float32)
    labels_arr = np.asarray(all_labels, dtype=np.float32)
    probs_arr = 1.0 / (1.0 + np.exp(-logits_arr))
    results = compute_binary_classification_metrics(
        labels_arr.astype(int),
        probs_arr,
        threshold=threshold,
        digits=3,
    )
    results["logits"] = logits_arr
    results["auc_roc"] = results.pop("roc_auc")
    return results


def print_ts_evaluation(results: dict) -> None:
    logger.info("Threshold: %.4f", results["threshold"])
    logger.info("Accuracy: %.4f", results["accuracy"])
    logger.info("Precision: %.4f", results["precision"])
    logger.info("Recall: %.4f", results["recall"])
    logger.info("F1: %.4f", results["f1"])
    logger.info("AUC-ROC: %.4f", results["auc_roc"])
    logger.info("PR-AUC: %.4f", results["pr_auc"])
    print(results["classification_report"])
