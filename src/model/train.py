"""Training loop para MarketValueNet."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

from .architecture import MarketValueNet
from .dataset import PolymarketDataset, create_train_val_test_dataloaders
from .evaluate import evaluate_model
from .metrics import find_best_threshold

logger = logging.getLogger(__name__)


def _to_json_serializable(value):
    if isinstance(value, dict):
        return {k: _to_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def train_model(
    model: MarketValueNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int | None = None,
    device: str | None = None,
    save_dir: str = "data/models/market_value_baseline",
) -> dict:
    """
    Entrena el modelo MarketValueNet.

    Returns:
        Diccionario con historial de entrenamiento.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if model.task == "classification":
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve_epochs = 0
    effective_patience = patience if patience is not None and patience > 0 else None
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_auc": [],
        "val_pr_auc": [],
    }

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        train_losses = []
        for batch in train_loader:
            num = batch["numerical"].to(device)
            cat = batch["category"].to(device)
            txt = batch["text_emb"].to(device)
            lbl = batch["label"].to(device)

            pred = model(num, cat, txt)
            loss = criterion(pred, lbl)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validate ──
        model.eval()
        val_losses, correct, total = [], 0, 0
        all_val_scores, all_val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                num = batch["numerical"].to(device)
                cat = batch["category"].to(device)
                txt = batch["text_emb"].to(device)
                lbl = batch["label"].to(device)

                pred = model(num, cat, txt)
                val_losses.append(criterion(pred, lbl).item())

                if model.task == "classification":
                    predicted = (pred > 0.5).float()
                    correct += (predicted == lbl).sum().item()
                    total += lbl.size(0)
                    all_val_scores.extend(pred.cpu().numpy())
                    all_val_labels.extend(lbl.cpu().numpy())

        scheduler.step()

        avg_train = float(np.mean(train_losses)) if train_losses else 0.0
        avg_val = float(np.mean(val_losses)) if val_losses else 0.0
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        metric_improved = False
        if model.task == "classification":
            acc = correct / total if total > 0 else 0
            # AUC-ROC es más informativa que accuracy para datos desbalanceados
            try:
                auc = roc_auc_score(all_val_labels, all_val_scores)
            except ValueError:
                auc = 0.0
            try:
                pr_auc = average_precision_score(all_val_labels, all_val_scores)
            except ValueError:
                pr_auc = 0.0
            history["val_accuracy"].append(acc)
            history["val_auc"].append(auc)
            history["val_pr_auc"].append(pr_auc)
            logger.info(
                "Epoch %d/%d | Train: %.4f | Val Loss: %.4f | Acc: %.3f | AUC: %.3f | PR-AUC: %.3f",
                epoch + 1, epochs, avg_train, avg_val, acc, auc, pr_auc,
            )
        else:
            logger.info(
                "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f",
                epoch + 1, epochs, avg_train, avg_val,
            )

        # Guardar mejor modelo por AUC (más relevante que val_loss con datos desbalanceados)
        if model.task == "classification":
            if auc > best_val_auc:
                best_val_auc = auc
                best_epoch = epoch + 1
                no_improve_epochs = 0
                metric_improved = True
                torch.save(model.state_dict(), save_path / "best_market_model.pt")
            else:
                no_improve_epochs += 1
        elif avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch + 1
            no_improve_epochs = 0
            metric_improved = True
            torch.save(model.state_dict(), save_path / "best_market_model.pt")
        else:
            no_improve_epochs += 1

        if effective_patience is not None and not metric_improved and no_improve_epochs >= effective_patience:
            logger.info(
                "Early stopping en época %d (patience=%d).",
                epoch + 1,
                effective_patience,
            )
            break

    # Guardar último modelo e historial
    torch.save(model.state_dict(), save_path / "last_market_model.pt")
    history["best_epoch"] = best_epoch
    if model.task == "classification":
        history["best_val_auc"] = best_val_auc
    else:
        history["best_val_loss"] = best_val_loss
    with open(save_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Entrenamiento completado. Mejor val AUC: %.4f", best_val_auc)
    logger.info("Modelos guardados en %s/", save_path)

    return history


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train MarketValueNet")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--val-split", type=float, default=None)
    parser.add_argument("--test-split", type=float, default=None)
    parser.add_argument("--split-strategy", choices=["random", "temporal"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--task", choices=["classification", "regression"], default=None)
    args = parser.parse_args()

    # Cargar config.yaml como base, CLI args sobrescriben
    from ..config import load_config
    cfg = load_config(args.config)
    training_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    data_dir = args.data_dir or data_cfg.get("processed_dir", "data/processed")
    save_dir = args.save_dir or training_cfg.get("save_dir", "data/models/market_value_baseline")
    epochs = args.epochs or training_cfg.get("epochs", 50)
    lr = args.lr or training_cfg.get("learning_rate", 1e-3)
    batch_size = args.batch_size or training_cfg.get("batch_size", 64)
    val_split = args.val_split if args.val_split is not None else training_cfg.get("val_split", 0.15)
    test_split = args.test_split if args.test_split is not None else training_cfg.get("test_split", 0.15)
    split_strategy = args.split_strategy or training_cfg.get("split_strategy", "random")
    seed = args.seed if args.seed is not None else training_cfg.get("seed", 42)
    patience = args.patience if args.patience is not None else training_cfg.get("patience", 0)
    task = args.task or model_cfg.get("task", "classification")

    logger.info("Config cargada desde %s", args.config)
    logger.info(
        "  data_dir=%s, save_dir=%s, epochs=%d, lr=%s, batch_size=%d, "
        "task=%s, split_strategy=%s, val_split=%.3f, test_split=%.3f, seed=%d, patience=%s",
        data_dir,
        save_dir,
        epochs,
        lr,
        batch_size,
        task,
        split_strategy,
        val_split,
        test_split,
        seed,
        patience,
    )

    # Cargar datos
    logger.info("Cargando dataset...")
    dataset = PolymarketDataset.from_numpy_dir(data_dir)
    logger.info("  Samples: %d", len(dataset))
    logger.info("  Positivos: %d", int(dataset.labels.sum()))
    logger.info("  Negativos: %d", len(dataset) - int(dataset.labels.sum()))
    if split_strategy == "temporal" and dataset.timestamps is None:
        raise ValueError(
            "split_strategy='temporal' requiere end_dates.npy en el dataset."
        )

    train_loader, val_loader, test_loader, split_metadata = create_train_val_test_dataloaders(
        dataset,
        batch_size=batch_size,
        val_split=val_split,
        test_split=test_split,
        use_weighted_sampler=training_cfg.get("use_weighted_sampler", True),
        split_strategy=split_strategy,
        seed=seed,
    )

    # Crear modelo
    model = MarketValueNet(
        num_numerical_features=dataset.numerical.shape[1],
        num_categories=model_cfg.get("num_categories", 10),
        category_embed_dim=model_cfg.get("category_embed_dim", 8),
        text_embed_dim=dataset.text_emb.shape[1],
        hidden_dims=model_cfg.get("hidden_dims", [256, 128, 64]),
        dropout=model_cfg.get("dropout", 0.3),
        task=task,
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Modelo creado: %d parámetros", total_params)

    # Entrenar
    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        patience=patience,
        save_dir=save_dir,
    )

    save_path = Path(save_dir)
    run_config = {
        "data_dir": data_dir,
        "save_dir": save_dir,
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "task": task,
        "seed": seed,
        "patience": patience,
        "split": split_metadata,
        "model": {
            "num_categories": model_cfg.get("num_categories", 10),
            "category_embed_dim": model_cfg.get("category_embed_dim", 8),
            "hidden_dims": model_cfg.get("hidden_dims", [256, 128, 64]),
            "dropout": model_cfg.get("dropout", 0.3),
        },
    }
    with open(save_path / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    if test_loader is not None:
        best_model = MarketValueNet(
            num_numerical_features=dataset.numerical.shape[1],
            num_categories=model_cfg.get("num_categories", 10),
            category_embed_dim=model_cfg.get("category_embed_dim", 8),
            text_embed_dim=dataset.text_emb.shape[1],
            hidden_dims=model_cfg.get("hidden_dims", [256, 128, 64]),
            dropout=model_cfg.get("dropout", 0.3),
            task=task,
        )
        best_model.load_state_dict(
            torch.load(save_path / "best_market_model.pt", map_location="cpu", weights_only=True)
        )
        val_results = evaluate_model(best_model, val_loader, threshold=0.5)
        threshold_tuning = find_best_threshold(
            val_results["labels"],
            val_results["scores"],
            objective="f1",
        )
        test_default = evaluate_model(best_model, test_loader, threshold=0.5)
        test_tuned = evaluate_model(
            best_model,
            test_loader,
            threshold=threshold_tuning["threshold"],
        )
        serializable_test_metrics = {
            "default_threshold": {
                key: value
                for key, value in test_default.items()
                if key not in {"scores", "labels", "predictions", "confusion_matrix"}
            },
            "tuned_threshold": {
                key: value
                for key, value in test_tuned.items()
                if key not in {"scores", "labels", "predictions", "confusion_matrix"}
            },
            "threshold_tuning": threshold_tuning,
        }
        serializable_test_metrics["default_threshold"]["confusion_matrix"] = test_default["confusion_matrix"].tolist()
        serializable_test_metrics["tuned_threshold"]["confusion_matrix"] = test_tuned["confusion_matrix"].tolist()
        serializable_test_metrics = _to_json_serializable(serializable_test_metrics)
        with open(save_path / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(serializable_test_metrics, f, indent=2)

        logger.info(
            "Test default thr=0.50 | Acc: %.3f | Precision: %.3f | Recall: %.3f | F1: %.3f | AUC: %.3f | PR-AUC: %.3f",
            serializable_test_metrics["default_threshold"].get("accuracy", 0.0),
            serializable_test_metrics["default_threshold"].get("precision", 0.0),
            serializable_test_metrics["default_threshold"].get("recall", 0.0),
            serializable_test_metrics["default_threshold"].get("f1", 0.0),
            serializable_test_metrics["default_threshold"].get("roc_auc", 0.0),
            serializable_test_metrics["default_threshold"].get("pr_auc", 0.0),
        )
        logger.info(
            "Test tuned thr=%.3f | Acc: %.3f | Precision: %.3f | Recall: %.3f | F1: %.3f | AUC: %.3f | PR-AUC: %.3f",
            serializable_test_metrics["tuned_threshold"].get("threshold", 0.5),
            serializable_test_metrics["tuned_threshold"].get("accuracy", 0.0),
            serializable_test_metrics["tuned_threshold"].get("precision", 0.0),
            serializable_test_metrics["tuned_threshold"].get("recall", 0.0),
            serializable_test_metrics["tuned_threshold"].get("f1", 0.0),
            serializable_test_metrics["tuned_threshold"].get("roc_auc", 0.0),
            serializable_test_metrics["tuned_threshold"].get("pr_auc", 0.0),
        )


if __name__ == "__main__":
    main()
