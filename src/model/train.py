"""Training orchestration for tabular + sequence p_yes models."""

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
from .architecture import MarketValueNet
from .calibration import fit_isotonic_calibrator, fit_platt_calibrator, identity_calibrator
from .catboost_train import train_catboost_pipeline
from .dataset import PolymarketDataset, create_train_val_test_dataloaders
from .evaluate import collect_tabular_outputs, evaluate_model
from .gbdt_train import train_gbdt_pipeline
from .metrics import clip_probabilities_from_residual, compute_probability_metrics, sigmoid
from .ts_train import train_ts_pipeline

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    model: MarketValueNet,
    train_loader,
    val_loader,
    epochs: int = 30,
    lr: float = 1e-3,
    patience: int = 6,
    device: str | None = None,
    save_dir: str = "data/models/market_value_baseline",
    prediction_mode: str = "classification",
) -> dict:
    """Train the tabular model with BCEWithLogitsLoss and val log loss selection."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    train_labels = np.concatenate([batch["label"].numpy() for batch in train_loader])
    positives = float(train_labels.sum())
    negatives = float(len(train_labels) - positives)

    if prediction_mode == "residual":
        criterion = nn.SmoothL1Loss(beta=0.1)
    else:
        pos_weight = torch.tensor([negatives / max(positives, 1.0)], device=device, dtype=torch.float32)
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
                batch["numerical"].to(device),
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
                    batch["numerical"].to(device),
                    batch["category"].to(device),
                    batch["text_emb"].to(device),
                )
                targets = batch["target"].to(device) if prediction_mode == "residual" else batch["label"].to(device)
                val_losses.append(criterion(logits, targets).item())
                val_logits.extend(logits.cpu().numpy())
                val_labels.extend(batch["label"].cpu().numpy())
                if prediction_mode == "residual":
                    if "snapshot_price" not in batch:
                        raise ValueError("Residual mode requiere snapshot_price en validación.")
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
            "TAB epoch %d/%d | train_loss=%.4f val_loss=%.4f brier=%.4f logloss=%.4f auc=%.4f pr_auc=%.4f",
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
            torch.save(model.state_dict(), save_path / "best_market_model.pt")
        else:
            no_improve_epochs += 1
            if patience and no_improve_epochs >= patience:
                logger.info("Tabular early stopping at epoch %d", epoch + 1)
                break

    torch.save(model.state_dict(), save_path / "last_market_model.pt")
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    with open(save_path / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return history


def train_tabular_pipeline(
    cfg: dict,
    data_dir: str | None = None,
    save_dir: str | None = None,
) -> dict:
    """Train, calibrate, and evaluate the tabular model bundle."""
    training_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    scoring_cfg = cfg.get("scoring", {})
    feature_cfg = cfg.get("features", {})

    data_dir = data_dir or data_cfg.get("processed_dir", "data/processed")
    save_dir = save_dir or training_cfg.get("save_dir", "data/models/market_value_baseline")
    epochs = training_cfg.get("epochs", 30)
    lr = training_cfg.get("learning_rate", 1e-3)
    batch_size = training_cfg.get("batch_size", 128)
    patience = training_cfg.get("patience", 6)
    seed = training_cfg.get("seed", 42)
    val_split = training_cfg.get("val_split", 0.15)
    test_split = training_cfg.get("test_split", 0.15)
    split_strategy = training_cfg.get("split_strategy", "temporal_grouped")
    top_k = scoring_cfg.get("top_k", 20)
    calibration_tolerance = float(feature_cfg.get("calibration_tolerance", 0.005))
    prediction_mode = "residual" if feature_cfg.get("target") == "residual_yes_minus_price" else "classification"
    roi_cap = scoring_cfg.get("expected_roi_cap")

    set_seed(seed)
    dataset = PolymarketDataset.from_numpy_dir(data_dir)
    train_loader, val_loader, test_loader, split_metadata = create_train_val_test_dataloaders(
        dataset,
        batch_size=batch_size,
        val_split=val_split,
        test_split=test_split,
        split_strategy=split_strategy,
        seed=seed,
    )

    model = MarketValueNet(
        num_numerical_features=int(dataset.numerical.shape[-1]),
        num_categories=int(dataset.categories.max().item()) + 1 if len(dataset.categories) else model_cfg.get("num_categories", 10),
        category_embed_dim=model_cfg.get("category_embed_dim", 8),
        text_embed_dim=int(dataset.text_emb.shape[-1]),
        hidden_dims=model_cfg.get("hidden_dims", [256, 128, 64]),
        dropout=model_cfg.get("dropout", 0.2),
        task="classification",
    )

    history = train_model(
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
    best_model = MarketValueNet(
        num_numerical_features=int(dataset.numerical.shape[-1]),
        num_categories=int(dataset.categories.max().item()) + 1 if len(dataset.categories) else model_cfg.get("num_categories", 10),
        category_embed_dim=model_cfg.get("category_embed_dim", 8),
        text_embed_dim=int(dataset.text_emb.shape[-1]),
        hidden_dims=model_cfg.get("hidden_dims", [256, 128, 64]),
        dropout=model_cfg.get("dropout", 0.2),
        task="classification",
    )
    best_model.load_state_dict(torch.load(save_path / "best_market_model.pt", map_location="cpu", weights_only=True))

    val_outputs = collect_tabular_outputs(best_model, val_loader)
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

    val_results = evaluate_model(
        best_model,
        val_loader,
        calibrator=best_calibrator,
        top_k=top_k,
        prediction_mode=prediction_mode,
        roi_cap=roi_cap,
    )
    test_results = evaluate_model(
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
        "model": {
            "num_numerical_features": int(dataset.numerical.shape[-1]),
            "num_categories": int(dataset.categories.max().item()) + 1 if len(dataset.categories) else model_cfg.get("num_categories", 10),
            "category_embed_dim": model_cfg.get("category_embed_dim", 8),
            "hidden_dims": model_cfg.get("hidden_dims", [256, 128, 64]),
            "dropout": model_cfg.get("dropout", 0.2),
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
        "TAB calibration selected=%s | test_brier=%.4f market_brier=%.4f topk_realized_pnl=%.4f",
        best_calibration_name,
        test_results["calibrated_metrics"]["brier"] if test_results else 0.0,
        test_results["market_baseline_metrics"]["brier"] if test_results else 0.0,
        test_results["ev_metrics"]["top_k_avg_realized_pnl"] if test_results else 0.0,
    )

    return {
        "model_name": "market_value_baseline",
        "save_dir": save_dir,
        "history": history,
        "validation": val_results,
        "test": test_results,
        "calibration_method": best_calibration_name,
    }


def save_model_comparison(*summaries: dict | None, output_dir: str | Path) -> dict:
    """Compare trained models and persist primary-model selection metadata."""
    models = [summary for summary in summaries if summary]
    if not models:
        raise ValueError("No hay modelos para comparar.")

    def ranking_key(summary: dict):
        test = summary["test"]
        return (
            float(test["ev_metrics"]["top_k_avg_realized_pnl"]),
            -float(test["calibrated_metrics"]["brier"]),
            -float(test["calibrated_metrics"]["log_loss"]),
        )

    def beats_market(summary: dict) -> bool:
        test = summary["test"]
        top_k_positive = float(test["ev_metrics"]["top_k_avg_realized_pnl"]) > 0.0
        beats_brier = float(test["calibrated_metrics"]["brier"]) < float(test["market_baseline_metrics"]["brier"])
        beats_log_loss = float(test["calibrated_metrics"]["log_loss"]) < float(test["market_baseline_metrics"]["log_loss"])
        return top_k_positive and (beats_brier or beats_log_loss)

    eligible_models = [summary for summary in models if beats_market(summary)]
    primary_pool = eligible_models if eligible_models else models
    primary = max(primary_pool, key=ranking_key)
    comparison = {
        "models": {
            summary["model_name"]: {
                "save_dir": summary["save_dir"],
                "calibration_method": summary["calibration_method"],
                "beats_market": beats_market(summary),
                "test": _serialize_results(summary["test"]),
            }
            for summary in models
        },
        "primary_model": {
            "name": primary["model_name"],
            "save_dir": primary["save_dir"],
            "selection_metric": "eligible_if(top_k_pnl>0 and beats_market_prob) -> top_k_avg_realized_pnl -> brier -> log_loss",
        },
    }
    out_path = Path(output_dir) / "registry"
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    with open(out_path / "primary_model.json", "w", encoding="utf-8") as f:
        json.dump(comparison["primary_model"], f, indent=2)
    return comparison


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

    parser = argparse.ArgumentParser(description="Train snapshot-based p_yes models")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--only", choices=["tabular", "ts", "gbdt", "catboost", "both", "all"], default="all")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--ts-data-dir", default=None)
    parser.add_argument("--ts-save-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    tabular_summary = None
    ts_summary = None
    gbdt_summary = None
    catboost_summary = None
    if args.only in {"tabular", "both", "all"}:
        tabular_summary = train_tabular_pipeline(cfg, data_dir=args.data_dir, save_dir=args.save_dir)
    if args.only in {"ts", "both", "all"}:
        ts_summary = train_ts_pipeline(cfg, data_dir=args.ts_data_dir, save_dir=args.ts_save_dir)
    if args.only in {"gbdt", "all"}:
        gbdt_summary = train_gbdt_pipeline(cfg, data_dir=args.data_dir)
    if args.only in {"catboost", "all"}:
        catboost_summary = train_catboost_pipeline(cfg, data_dir=args.data_dir)

    if args.only in {"both", "all"} and any(summary is not None for summary in [tabular_summary, ts_summary, gbdt_summary, catboost_summary]):
        models_root = Path(cfg.get("training", {}).get("save_dir", "data/models/market_value_baseline")).parent
        comparison = save_model_comparison(tabular_summary, ts_summary, gbdt_summary, catboost_summary, output_dir=models_root)
        logger.info("Primary model selected: %s", comparison["primary_model"]["name"])


if __name__ == "__main__":
    main()
