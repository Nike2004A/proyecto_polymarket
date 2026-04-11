"""Dataset y DataLoaders para el entrenamiento del modelo."""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler

from .splits import build_dataset_split

logger = logging.getLogger(__name__)


class PolymarketDataset(Dataset):
    """Dataset de mercados de Polymarket para entrenamiento."""

    def __init__(
        self,
        numerical_features: np.ndarray,
        category_ids: np.ndarray,
        text_embeddings: np.ndarray,
        labels: np.ndarray,
        timestamps: np.ndarray | None = None,
    ):
        self.numerical = torch.FloatTensor(numerical_features)
        self.categories = torch.LongTensor(category_ids)
        self.text_emb = torch.FloatTensor(text_embeddings)
        self.labels = torch.FloatTensor(labels)
        # Timestamps (unix epoch) for temporal split — None si no disponible
        self.timestamps = timestamps

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "numerical": self.numerical[idx],
            "category": self.categories[idx],
            "text_emb": self.text_emb[idx],
            "label": self.labels[idx],
        }

    @classmethod
    def from_numpy_dir(cls, directory: str) -> "PolymarketDataset":
        """Carga un dataset desde archivos .npy en un directorio."""
        path = Path(directory)

        timestamps = None
        end_dates_path = path / "end_dates.npy"
        if end_dates_path.exists():
            timestamps = np.load(end_dates_path)

        return cls(
            numerical_features=np.load(path / "numerical_features.npy"),
            category_ids=np.load(path / "category_ids.npy"),
            text_embeddings=np.load(path / "text_embeddings.npy"),
            labels=np.load(path / "labels.npy"),
            timestamps=timestamps,
        )


def create_dataloaders(
    dataset: PolymarketDataset,
    batch_size: int = 64,
    val_split: float = 0.2,
    use_weighted_sampler: bool = True,
    num_workers: int = 0,
    temporal_split: bool = True,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Divide en train/val y crea DataLoaders.

    Args:
        dataset: PolymarketDataset.
        batch_size: Tamaño de batch.
        val_split: Fracción para validación.
        use_weighted_sampler: Si usar muestreo ponderado para balancear clases.
        num_workers: Número de workers para carga de datos.
        temporal_split: Si True y hay timestamps, divide temporalmente
            (train = mercados más antiguos, val = más recientes) en lugar
            de random split. Esto evita data leakage temporal.
        seed: Semilla para random split y sampler.

    Returns:
        (train_loader, val_loader)
    """
    split_strategy = "temporal" if temporal_split else "random"
    train_loader, val_loader, _, _ = create_train_val_test_dataloaders(
        dataset,
        batch_size=batch_size,
        val_split=val_split,
        test_split=0.0,
        use_weighted_sampler=use_weighted_sampler,
        num_workers=num_workers,
        split_strategy=split_strategy,
        seed=seed,
    )
    return train_loader, val_loader


def create_train_val_test_dataloaders(
    dataset: PolymarketDataset,
    batch_size: int = 64,
    val_split: float = 0.15,
    test_split: float = 0.15,
    use_weighted_sampler: bool = True,
    num_workers: int = 0,
    split_strategy: str = "random",
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader | None, dict]:
    """Crea dataloaders train/val/test reproducibles."""
    split = build_dataset_split(
        n_samples=len(dataset),
        labels=dataset.labels.numpy(),
        timestamps=dataset.timestamps,
        val_split=val_split,
        test_split=test_split,
        strategy=split_strategy,
        seed=seed,
    )

    if split.strategy == "temporal":
        logger.info(
            "Temporal split: train=%d, val=%d, test=%d",
            len(split.train_indices),
            len(split.val_indices),
            len(split.test_indices),
        )
    else:
        logger.info(
            "Random split: train=%d, val=%d, test=%d (seed=%d)",
            len(split.train_indices),
            len(split.val_indices),
            len(split.test_indices),
            seed,
        )

    train_ds = Subset(dataset, split.train_indices)
    val_ds = Subset(dataset, split.val_indices)

    train_sampler = None
    shuffle = True
    if use_weighted_sampler:
        train_labels = dataset.labels[split.train_indices]
        class_counts = torch.bincount(train_labels.long())
        if len(class_counts) >= 2 and class_counts.min() > 0:
            class_weights = 1.0 / class_counts.float()
            sample_weights = class_weights[train_labels.long()]
            generator = torch.Generator().manual_seed(seed)
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_ds),
                replacement=True,
                generator=generator,
            )
            shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle if train_sampler is None else False,
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = None
    if split.test_indices:
        test_loader = DataLoader(
            Subset(dataset, split.test_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    metadata = split.to_metadata()
    metadata["class_balance"] = {
        "train_positive_rate": float(dataset.labels[split.train_indices].float().mean().item()),
        "val_positive_rate": float(dataset.labels[split.val_indices].float().mean().item()),
        "test_positive_rate": (
            float(dataset.labels[split.test_indices].float().mean().item())
            if split.test_indices
            else None
        ),
    }
    return train_loader, val_loader, test_loader, metadata
