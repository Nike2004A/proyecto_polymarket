"""Training loop para MarketValueNet."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .architecture import MarketValueNet
from .dataset import PolymarketDataset, create_dataloaders


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
    parser = argparse.ArgumentParser(description="Train MarketValueNet")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--save-dir", default="data/models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    args = parser.parse_args()

    # Cargar datos
    print("Cargando dataset...")
    dataset = PolymarketDataset.from_numpy_dir(args.data_dir)
    print(f"  Samples: {len(dataset)}")
    print(f"  Positivos: {int(dataset.labels.sum())}")
    print(f"  Negativos: {len(dataset) - int(dataset.labels.sum())}")

    train_loader, val_loader = create_dataloaders(
        dataset, batch_size=args.batch_size
    )

    # Crear modelo
    model = MarketValueNet(
        num_numerical_features=dataset.numerical.shape[1],
        task=args.task,
    )
    print(f"\nModelo: {model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros totales: {total_params:,}")

    # Entrenar
    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
