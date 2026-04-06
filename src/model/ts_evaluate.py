"""Evaluación para el modelo GRU de series de tiempo."""

from __future__ import annotations

import logging

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from .ts_architecture import PriceSequenceGRU

logger = logging.getLogger(__name__)


def evaluate_ts_model(
    model: PriceSequenceGRU,
    data_loader,
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

    scores_arr = np.asarray(all_scores, dtype=np.float32)
    labels_arr = np.asarray(all_labels, dtype=np.float32)
    predicted = (scores_arr > 0.0).astype(int)  # logit > 0 ≡ sigmoid > 0.5
    labels_int = labels_arr.astype(int)

    if len(np.unique(labels_int)) < 2:
        auc = 0.0
        pr_auc = 0.0
    else:
        try:
            auc = roc_auc_score(labels_int, scores_arr)
        except ValueError:
            auc = 0.0
        try:
            pr_auc = average_precision_score(labels_int, scores_arr)
        except ValueError:
            pr_auc = 0.0
    if not np.isfinite(auc):
        auc = 0.0
    if not np.isfinite(pr_auc):
        pr_auc = 0.0

    return {
        "accuracy": accuracy_score(labels_int, predicted),
        "auc_roc": auc,
        "pr_auc": pr_auc,
        "confusion_matrix": confusion_matrix(labels_int, predicted),
        "classification_report": classification_report(
            labels_int,
            predicted,
            target_names=["No Buy", "Buy"],
            digits=3,
        ),
        "labels": labels_int,
        "scores": scores_arr,
    }


def print_ts_evaluation(results: dict) -> None:
    logger.info("Accuracy: %.4f", results["accuracy"])
    logger.info("AUC-ROC: %.4f", results["auc_roc"])
    logger.info("PR-AUC: %.4f", results["pr_auc"])
    print(results["classification_report"])
