"""Evaluation helpers for the tabular snapshot model."""

from __future__ import annotations

import numpy as np
import torch

from .metrics import (
    clip_probabilities_from_residual,
    compute_binary_classification_metrics,
    compute_bucketed_metrics,
    compute_ev_metrics,
    compute_probability_metrics,
    compute_residual_metrics,
    sigmoid,
)


def collect_tabular_outputs(model, dataloader, device: str | None = None) -> dict:
    """Collect logits, labels, and snapshot metadata from a tabular dataloader."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    logits = []
    labels = []
    prices = []
    days_to_end = []
    market_ids = []
    snapshot_times = []

    with torch.no_grad():
        for batch in dataloader:
            batch_logits = model(
                batch["numerical"].to(device),
                batch["category"].to(device),
                batch["text_emb"].to(device),
            )
            logits.extend(batch_logits.cpu().numpy())
            labels.extend(batch["label"].cpu().numpy())
            if "snapshot_price" in batch:
                prices.extend(batch["snapshot_price"].cpu().numpy())
            if "days_to_end" in batch:
                days_to_end.extend(batch["days_to_end"].cpu().numpy())
            if "market_id" in batch:
                market_ids.extend(batch["market_id"])
            if "snapshot_time" in batch:
                snapshot_times.extend(batch["snapshot_time"].cpu().numpy())

    return {
        "logits": np.asarray(logits, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.float32),
        "market_prices": np.asarray(prices, dtype=np.float32),
        "days_to_end": np.asarray(days_to_end, dtype=np.float32),
        "market_ids": np.asarray(market_ids, dtype=object),
        "snapshot_times": np.asarray(snapshot_times, dtype=np.int64),
    }


def evaluate_model(
    model,
    dataloader,
    calibrator=None,
    top_k: int = 20,
    prediction_mode: str = "classification",
    roi_cap: float | None = None,
    device: str | None = None,
    threshold: float | None = None,
) -> dict:
    outputs = collect_tabular_outputs(model, dataloader, device=device)
    if prediction_mode == "residual":
        raw_probs = clip_probabilities_from_residual(outputs["logits"], outputs["market_prices"])
        calibrated_probs = calibrator.predict_proba(raw_probs) if calibrator is not None else raw_probs
        residual_metrics = compute_residual_metrics(
            outputs["labels"],
            outputs["market_prices"],
            outputs["logits"],
        )
    else:
        raw_probs = sigmoid(outputs["logits"])
        calibrated_probs = calibrator.predict_proba(outputs["logits"]) if calibrator is not None else raw_probs
        residual_metrics = {}

    results = {
        "labels": outputs["labels"],
        "logits": outputs["logits"],
        "raw_probs": raw_probs,
        "calibrated_probs": calibrated_probs,
        "market_prices": outputs["market_prices"],
        "days_to_end": outputs["days_to_end"],
        "market_ids": outputs["market_ids"],
        "snapshot_times": outputs["snapshot_times"],
        "prediction_mode": prediction_mode,
        "raw_metrics": compute_probability_metrics(outputs["labels"], raw_probs),
        "calibrated_metrics": compute_probability_metrics(outputs["labels"], calibrated_probs),
        "market_baseline_metrics": compute_probability_metrics(outputs["labels"], outputs["market_prices"]),
        "residual_metrics": residual_metrics,
        "ev_metrics": compute_ev_metrics(
            outputs["labels"],
            outputs["market_prices"],
            calibrated_probs,
            top_k=top_k,
            roi_cap=roi_cap,
        ),
        "by_horizon": compute_bucketed_metrics(
            outputs["labels"],
            outputs["market_prices"],
            calibrated_probs,
            outputs["days_to_end"],
            top_k=top_k,
            roi_cap=roi_cap,
        ),
    }
    if threshold is not None:
        binary_metrics = compute_binary_classification_metrics(outputs["labels"], calibrated_probs, threshold=threshold)
        results.update(binary_metrics)
        results["scores"] = calibrated_probs
    return results
