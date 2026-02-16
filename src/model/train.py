"""Training loop para MarketValueNet."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .architecture import MarketValueNet
from .dataset import PolymarketDataset, create_dataloaders

logger = logging.getLogger(__name__)


def train_model(
    model: MarketValueNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str | None = None,
    save_dir: str = "data/models",
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

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

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

        scheduler.step()

        avg_train = float(np.mean(train_losses)) if train_losses else 0.0
        avg_val = float(np.mean(val_losses)) if val_losses else 0.0
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        if model.task == "classification":
            acc = correct / total if total > 0 else 0
            history["val_accuracy"].append(acc)
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {avg_train:.4f} | "
                f"Val Loss: {avg_val:.4f} | "
                f"Val Acc: {acc:.4f}"
            )
        else:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {avg_train:.4f} | "
                f"Val Loss: {avg_val:.4f}"
            )

        # Save best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path / "best_market_model.pt")

    # Guardar último modelo e historial
    torch.save(model.state_dict(), save_path / "last_market_model.pt")
    with open(save_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nEntrenamiento completado. Mejor val loss: {best_val_loss:.4f}")
    print(f"Modelos guardados en {save_path}/")

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
    parser.add_argument("--task", choices=["classification", "regression"], default=None)
    args = parser.parse_args()

    # Cargar config.yaml como base, CLI args sobrescriben
    from ..config import load_config
    cfg = load_config(args.config)
    training_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    data_dir = args.data_dir or data_cfg.get("processed_dir", "data/processed")
    save_dir = args.save_dir or training_cfg.get("save_dir", "data/models")
    epochs = args.epochs or training_cfg.get("epochs", 50)
    lr = args.lr or training_cfg.get("learning_rate", 1e-3)
    batch_size = args.batch_size or training_cfg.get("batch_size", 64)
    task = args.task or model_cfg.get("task", "classification")

    logger.info("Config cargada desde %s", args.config)
    logger.info("  data_dir=%s, save_dir=%s, epochs=%d, lr=%s, batch_size=%d, task=%s",
                data_dir, save_dir, epochs, lr, batch_size, task)

    # Cargar datos
    logger.info("Cargando dataset...")
    dataset = PolymarketDataset.from_numpy_dir(data_dir)
    logger.info("  Samples: %d", len(dataset))
    logger.info("  Positivos: %d", int(dataset.labels.sum()))
    logger.info("  Negativos: %d", len(dataset) - int(dataset.labels.sum()))
    if dataset.timestamps is not None:
        logger.info("  Timestamps disponibles: temporal split habilitado.")
    else:
        logger.warning("  No hay timestamps: se usará random split.")

    train_loader, val_loader = create_dataloaders(
        dataset, batch_size=batch_size, temporal_split=True
    )

    # Crear modelo
    model = MarketValueNet(
        num_numerical_features=dataset.numerical.shape[1],
        num_categories=model_cfg.get("num_categories", 20),
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
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
