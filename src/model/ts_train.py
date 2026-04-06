"""Training loop reproducible para PriceSequenceGRU."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from ..config import load_config
from .ts_architecture import PriceSequenceGRU
from .ts_dataset import TimeSeriesMarketDataset, create_ts_dataloaders

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_ts_model(
    model: PriceSequenceGRU,
    train_loader,
    val_loader,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 5,
    seed: int = 42,
    device: str | None = None,
    save_dir: str = "data/models/ts_gru",
    run_config: dict | None = None,
) -> dict:
    """Entrena el modelo GRU con checkpoint por mejor val AUC."""
    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_auc": [],
        "val_pr_auc": [],
    }
    best_val_auc = float("-inf")
    no_improve_epochs = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            sequences = batch["sequence"].to(device)
            lengths = batch["sequence_length"].to(device)
            labels = batch["label"].to(device)

            scores = model(sequences, lengths)
            loss = criterion(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch["sequence"].to(device)
                lengths = batch["sequence_length"].to(device)
                labels = batch["label"].to(device)

                scores = model(sequences, lengths)
                val_losses.append(criterion(scores, labels).item())
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_scores_arr = np.asarray(all_scores, dtype=np.float32)
        all_labels_arr = np.asarray(all_labels, dtype=np.float32)
        predicted = (all_scores_arr > 0.0).astype(int)  # logit > 0 ≡ sigmoid > 0.5
        labels_int = all_labels_arr.astype(int)

        if len(np.unique(labels_int)) < 2:
            val_auc = 0.0
            val_pr_auc = 0.0
        else:
            try:
                val_auc = roc_auc_score(labels_int, all_scores_arr)
            except ValueError:
                val_auc = 0.0
            try:
                val_pr_auc = average_precision_score(labels_int, all_scores_arr)
            except ValueError:
                val_pr_auc = 0.0
        if not np.isfinite(val_auc):
            val_auc = 0.0
        if not np.isfinite(val_pr_auc):
            val_pr_auc = 0.0

        val_acc = float((predicted == labels_int).mean()) if len(labels_int) else 0.0
        avg_train = float(np.mean(train_losses)) if train_losses else 0.0
        avg_val = float(np.mean(val_losses)) if val_losses else 0.0

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_accuracy"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["val_pr_auc"].append(val_pr_auc)

        logger.info(
            "Epoch %d/%d | Train: %.4f | Val: %.4f | Acc: %.3f | AUC: %.3f | PR-AUC: %.3f",
            epoch + 1,
            epochs,
            avg_train,
            avg_val,
            val_acc,
            val_auc,
            val_pr_auc,
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            no_improve_epochs = 0
            torch.save(model.state_dict(), save_path / "best_ts_gru_model.pt")
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            logger.info(
                "Early stopping en época %d (patience=%d).",
                epoch + 1,
                patience,
            )
            break

    torch.save(model.state_dict(), save_path / "last_ts_gru_model.pt")
    with open(save_path / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(save_path / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config or {}, f, indent=2)

    if best_val_auc == float("-inf"):
        best_val_auc = 0.0
    logger.info("Entrenamiento TS completado. Mejor val AUC: %.4f", best_val_auc)
    return history


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train PriceSequenceGRU")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ts_data_cfg = cfg.get("ts_data", {})
    ts_model_cfg = cfg.get("ts_model", {})
    ts_training_cfg = cfg.get("ts_training", {})

    def _get(arg, cfg_key, default):
        return arg if arg is not None else cfg_key if cfg_key is not None else default

    data_dir = args.data_dir or ts_data_cfg.get("processed_dir", "data/processed_ts")
    save_dir = args.save_dir or ts_training_cfg.get("save_dir", "data/models/ts_gru")
    epochs = _get(args.epochs, ts_training_cfg.get("epochs"), 50)
    lr = _get(args.lr, ts_training_cfg.get("learning_rate"), 1e-3)
    batch_size = _get(args.batch_size, ts_training_cfg.get("batch_size"), 64)
    patience = _get(args.patience, ts_training_cfg.get("patience"), 5)
    seed = _get(args.seed, ts_training_cfg.get("seed"), 42)

    dataset = TimeSeriesMarketDataset.from_numpy_dir(data_dir)
    train_loader, val_loader = create_ts_dataloaders(
        dataset,
        batch_size=batch_size,
        temporal_split=True,
        use_weighted_sampler=True,
        seed=seed,
    )

    model = PriceSequenceGRU(
        input_dim=int(dataset.sequences.shape[-1]),
        hidden_dim=ts_model_cfg.get("hidden_dim", 64),
        num_layers=ts_model_cfg.get("num_layers", 1),
        dropout=ts_model_cfg.get("dropout", 0.2),
        task="classification",
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Modelo TS creado: params=%d | input_dim=%d",
        total_params,
        dataset.sequences.shape[-1],
    )

    run_config = {
        "data_dir": data_dir,
        "save_dir": save_dir,
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "patience": patience,
        "seed": seed,
        "dataset_metadata": dataset.metadata,
        "model": {
            "input_dim": int(dataset.sequences.shape[-1]),
            "hidden_dim": ts_model_cfg.get("hidden_dim", 64),
            "num_layers": ts_model_cfg.get("num_layers", 1),
            "dropout": ts_model_cfg.get("dropout", 0.2),
        },
    }
    train_ts_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        patience=patience,
        seed=seed,
        save_dir=save_dir,
        run_config=run_config,
    )


if __name__ == "__main__":
    main()
