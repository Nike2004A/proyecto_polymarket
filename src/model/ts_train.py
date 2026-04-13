"""Training loop for the fixed-grid sequence model."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import load_config
from .calibration import fit_isotonic_calibrator, fit_platt_calibrator, identity_calibrator
from .metrics import clip_probabilities_from_residual, compute_probability_metrics, sigmoid
from .ts_architecture import PriceSequenceGRU
from .ts_dataset import TimeSeriesMarketDataset, create_ts_train_val_test_dataloaders
from .ts_evaluate import collect_ts_outputs, evaluate_ts_model

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_ts_model(
    model: PriceSequenceGRU,
    train_loader,
    val_loader,
    epochs: int = 40,
    lr: float = 1e-3,
    patience: int = 6,
    device: str | None = None,
    save_dir: str = "data/models/price_sequence_gru",
    prediction_mode: str = "classification",
) -> dict:
    """Train the sequence model, selecting checkpoints by validation log loss."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    train_labels = np.concatenate([
        batch["label"].numpy()
        for batch in train_loader
    ])
    positives = float(train_labels.sum())
    negatives = float(len(train_labels) - positives)
    if prediction_mode == "residual":
        criterion = nn.SmoothL1Loss(beta=0.1)
    else:
        pos_weight = torch.tensor(
            [negatives / max(positives, 1.0)],
            device=device,
            dtype=torch.float32,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_brier": [],
        "val_log_loss": [],
        "val_auc": [],
        "val_pr_auc": [],
    }
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve_epochs = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            logits = model(
                batch["sequence"].to(device),
                batch["sequence_length"].to(device),
                batch["static_numerical"].to(device),
                batch["category"].to(device),
                batch["text_emb"].to(device),
            )
            targets = batch["target"].to(device) if prediction_mode == "residual" else batch["label"].to(device)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        val_logits = []
        val_labels = []
        val_prices = []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    batch["sequence"].to(device),
                    batch["sequence_length"].to(device),
                    batch["static_numerical"].to(device),
                    batch["category"].to(device),
                    batch["text_emb"].to(device),
                )
                targets = batch["target"].to(device) if prediction_mode == "residual" else batch["label"].to(device)
                val_losses.append(criterion(logits, targets).item())
                val_logits.extend(logits.cpu().numpy())
                val_labels.extend(batch["label"].cpu().numpy())
                if prediction_mode == "residual":
                    val_prices.extend(batch["snapshot_price"].cpu().numpy())

        if prediction_mode == "residual":
            val_probs = clip_probabilities_from_residual(
                np.asarray(val_logits, dtype=np.float32),
                np.asarray(val_prices, dtype=np.float32),
            )
        else:
            val_probs = sigmoid(np.asarray(val_logits, dtype=np.float32))
        val_metrics = compute_probability_metrics(np.asarray(val_labels, dtype=np.float32), val_probs)
        avg_train = float(np.mean(train_losses)) if train_losses else 0.0
        avg_val = float(np.mean(val_losses)) if val_losses else 0.0

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_brier"].append(val_metrics["brier"])
        history["val_log_loss"].append(val_metrics["log_loss"])
        history["val_auc"].append(val_metrics["roc_auc"])
        history["val_pr_auc"].append(val_metrics["pr_auc"])

        logger.info(
            "TS epoch %d/%d | train_loss=%.4f val_loss=%.4f brier=%.4f logloss=%.4f auc=%.4f pr_auc=%.4f",
            epoch + 1,
            epochs,
            avg_train,
            avg_val,
            val_metrics["brier"],
            val_metrics["log_loss"],
            val_metrics["roc_auc"],
            val_metrics["pr_auc"],
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch + 1
            no_improve_epochs = 0
            torch.save(model.state_dict(), save_path / "best_ts_gru_model.pt")
        else:
            no_improve_epochs += 1
            if patience and no_improve_epochs >= patience:
                logger.info("TS early stopping at epoch %d", epoch + 1)
                break

    torch.save(model.state_dict(), save_path / "last_ts_gru_model.pt")
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    with open(save_path / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return history


def train_ts_pipeline(
    cfg: dict,
    data_dir: str | None = None,
    save_dir: str | None = None,
) -> dict:
    """Train, calibrate, evaluate, and persist the sequence model bundle."""
    ts_data_cfg = cfg.get("ts_data", {})
    ts_model_cfg = cfg.get("ts_model", {})
    ts_training_cfg = cfg.get("ts_training", {})
    scoring_cfg = cfg.get("scoring", {})
    feature_cfg = cfg.get("features", {})

    data_dir = data_dir or ts_data_cfg.get("processed_dir", "data/processed_ts")
    save_dir = save_dir or ts_training_cfg.get("save_dir", "data/models/price_sequence_gru")
    epochs = ts_training_cfg.get("epochs", 40)
    lr = ts_training_cfg.get("learning_rate", 1e-3)
    batch_size = ts_training_cfg.get("batch_size", 128)
    patience = ts_training_cfg.get("patience", 6)
    seed = ts_training_cfg.get("seed", 42)
    val_split = ts_training_cfg.get("val_split", 0.15)
    test_split = ts_training_cfg.get("test_split", 0.15)
    split_strategy = ts_training_cfg.get("split_strategy", "temporal_grouped")
    top_k = scoring_cfg.get("top_k", 20)
    calibration_tolerance = float(feature_cfg.get("calibration_tolerance", 0.005))
    prediction_mode = "residual" if feature_cfg.get("target") == "residual_yes_minus_price" else "classification"
    roi_cap = scoring_cfg.get("expected_roi_cap")

    set_seed(seed)
    dataset = TimeSeriesMarketDataset.from_numpy_dir(data_dir)
    train_loader, val_loader, test_loader, split_metadata = create_ts_train_val_test_dataloaders(
        dataset,
        batch_size=batch_size,
        val_split=val_split,
        test_split=test_split,
        split_strategy=split_strategy,
        seed=seed,
    )

    model = PriceSequenceGRU(
        input_dim=int(dataset.sequences.shape[-1]),
        static_num_features=int(dataset.static_numerical.shape[-1]),
        num_categories=int(dataset.categories.max().item()) + 1 if len(dataset.categories) else 10,
        category_embed_dim=ts_model_cfg.get("category_embed_dim", 8),
        text_embed_dim=int(dataset.text_emb.shape[-1]),
        hidden_dim=ts_model_cfg.get("hidden_dim", 128),
        static_hidden_dim=ts_model_cfg.get("static_hidden_dim", 128),
        num_layers=ts_model_cfg.get("num_layers", 1),
        dropout=ts_model_cfg.get("dropout", 0.2),
        task="classification",
    )

    history = train_ts_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        patience=patience,
        save_dir=save_dir,
        prediction_mode=prediction_mode,
    )

    save_path = Path(save_dir)
    best_model = PriceSequenceGRU(
        input_dim=int(dataset.sequences.shape[-1]),
        static_num_features=int(dataset.static_numerical.shape[-1]),
        num_categories=int(dataset.categories.max().item()) + 1 if len(dataset.categories) else 10,
        category_embed_dim=ts_model_cfg.get("category_embed_dim", 8),
        text_embed_dim=int(dataset.text_emb.shape[-1]),
        hidden_dim=ts_model_cfg.get("hidden_dim", 128),
        static_hidden_dim=ts_model_cfg.get("static_hidden_dim", 128),
        num_layers=ts_model_cfg.get("num_layers", 1),
        dropout=ts_model_cfg.get("dropout", 0.2),
        task="classification",
    )
    best_model.load_state_dict(torch.load(save_path / "best_ts_gru_model.pt", map_location="cpu", weights_only=True))

    val_outputs = collect_ts_outputs(best_model, val_loader)
    if prediction_mode == "residual":
        raw_probs = clip_probabilities_from_residual(val_outputs["logits"], val_outputs["market_prices"])
        raw = identity_calibrator(input_kind="probability")
        platt = fit_platt_calibrator(raw_probs, val_outputs["labels"], input_kind="probability")
        isotonic = fit_isotonic_calibrator(raw_probs, val_outputs["labels"], input_kind="probability")
        candidate_calibrators = {
            "identity": raw,
            "platt": platt,
            "isotonic": isotonic,
        }
        calibration_scores = {}
        for name, calibrator in candidate_calibrators.items():
            probs = calibrator.predict_proba(raw_probs)
            calibration_scores[name] = compute_probability_metrics(val_outputs["labels"], probs)

        best_brier = min(metrics["brier"] for metrics in calibration_scores.values())
        preference = ["platt", "identity", "isotonic"]
        eligible = [
            name
            for name in preference
            if calibration_scores[name]["brier"] <= best_brier + calibration_tolerance
        ]
        best_calibration_name = eligible[0] if eligible else min(calibration_scores, key=lambda name: calibration_scores[name]["brier"])
        best_calibrator = candidate_calibrators[best_calibration_name]
        best_calibrator.save(save_path / "calibration")
    else:
        raw = identity_calibrator(input_kind="logit")
        platt = fit_platt_calibrator(val_outputs["logits"], val_outputs["labels"], input_kind="logit")
        isotonic = fit_isotonic_calibrator(val_outputs["logits"], val_outputs["labels"], input_kind="logit")
        candidate_calibrators = {
            "identity": raw,
            "platt": platt,
            "isotonic": isotonic,
        }
        calibration_scores = {}
        for name, calibrator in candidate_calibrators.items():
            probs = calibrator.predict_proba(val_outputs["logits"])
            calibration_scores[name] = compute_probability_metrics(val_outputs["labels"], probs)

        best_brier = min(metrics["brier"] for metrics in calibration_scores.values())
        preference = ["platt", "identity", "isotonic"]
        eligible = [
            name
            for name in preference
            if calibration_scores[name]["brier"] <= best_brier + calibration_tolerance
        ]
        best_calibration_name = eligible[0] if eligible else min(calibration_scores, key=lambda name: calibration_scores[name]["brier"])
        best_calibrator = candidate_calibrators[best_calibration_name]
        best_calibrator.save(save_path / "calibration")

    val_results = evaluate_ts_model(
        best_model,
        val_loader,
        calibrator=best_calibrator,
        top_k=top_k,
        prediction_mode=prediction_mode,
        roi_cap=roi_cap,
    )
    test_results = evaluate_ts_model(
        best_model,
        test_loader,
        calibrator=best_calibrator,
        top_k=top_k,
        prediction_mode=prediction_mode,
        roi_cap=roi_cap,
    ) if test_loader else {}

    run_config = {
        "data_dir": data_dir,
        "save_dir": save_dir,
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "patience": patience,
        "seed": seed,
        "prediction_mode": prediction_mode,
        "split": split_metadata,
        "dataset_metadata": dataset.metadata,
        "model": {
            "input_dim": int(dataset.sequences.shape[-1]),
            "static_num_features": int(dataset.static_numerical.shape[-1]),
            "num_categories": int(dataset.categories.max().item()) + 1 if len(dataset.categories) else 10,
            "category_embed_dim": ts_model_cfg.get("category_embed_dim", 8),
            "hidden_dim": ts_model_cfg.get("hidden_dim", 128),
            "static_hidden_dim": ts_model_cfg.get("static_hidden_dim", 128),
            "num_layers": ts_model_cfg.get("num_layers", 1),
            "dropout": ts_model_cfg.get("dropout", 0.2),
        },
        "calibration": {
            "selected": best_calibration_name,
            "tolerance": calibration_tolerance,
            "candidates": calibration_scores,
        },
        "selection_metric": cfg.get("model_selection", {}).get("primary_model_selection_metric", "topk_ev"),
    }
    with open(save_path / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    metrics_payload = {
        "validation": _serialize_results(val_results),
        "test": _serialize_results(test_results),
        "market_baseline_beaten": {
            "brier": (
                test_results["calibrated_metrics"]["brier"] < test_results["market_baseline_metrics"]["brier"]
                if test_results
                else False
            ),
            "log_loss": (
                test_results["calibrated_metrics"]["log_loss"] < test_results["market_baseline_metrics"]["log_loss"]
                if test_results
                else False
            ),
            "top_k_realized_pnl": (
                test_results["ev_metrics"]["top_k_avg_realized_pnl"] > 0.0
                if test_results
                else False
            ),
        },
    }
    with open(save_path / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    logger.info(
        "TS calibration selected=%s | test_brier=%.4f market_brier=%.4f topk_realized_pnl=%.4f",
        best_calibration_name,
        test_results["calibrated_metrics"]["brier"] if test_results else 0.0,
        test_results["market_baseline_metrics"]["brier"] if test_results else 0.0,
        test_results["ev_metrics"]["top_k_avg_realized_pnl"] if test_results else 0.0,
    )

    return {
        "model_name": "price_sequence_gru",
        "save_dir": save_dir,
        "history": history,
        "validation": val_results,
        "test": test_results,
        "calibration_method": best_calibration_name,
    }


def _serialize_results(results: dict) -> dict:
    if not results:
        return {}
    return {
        "raw_metrics": results["raw_metrics"],
        "calibrated_metrics": results["calibrated_metrics"],
        "market_baseline_metrics": results["market_baseline_metrics"],
        "residual_metrics": results.get("residual_metrics", {}),
        "ev_metrics": results["ev_metrics"],
        "by_horizon": results.get("by_horizon", {}),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train fixed-grid GRU p_yes model")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_ts_pipeline(cfg, data_dir=args.data_dir, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
