"""Dataset y DataLoaders para el entrenamiento del modelo."""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler

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

    Returns:
        (train_loader, val_loader)
    """
    n = len(dataset)
    n_val = int(n * val_split)
    n_train = n - n_val

    if temporal_split and dataset.timestamps is not None:
        # Temporal split: ordenar por timestamp, train = primeros, val = últimos
        sorted_indices = np.argsort(dataset.timestamps)
        train_indices = sorted_indices[:n_train].tolist()
        val_indices = sorted_indices[n_train:].tolist()
        logger.info(
            "Temporal split: train=%d (más antiguos), val=%d (más recientes)",
            len(train_indices), len(val_indices),
        )
    else:
        if temporal_split:
            logger.warning(
                "temporal_split=True pero no hay timestamps disponibles. "
                "Usando random split. Para temporal split, regenera features "
                "con el pipeline actualizado que guarda end_dates.npy."
            )
        # Random split (fallback)
        generator = torch.Generator().manual_seed(42)
        all_indices = torch.randperm(n, generator=generator).tolist()
        train_indices = all_indices[:n_train]
        val_indices = all_indices[n_train:]

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)

    # Weighted sampler para balancear clases
    train_sampler = None
    shuffle = True
    if use_weighted_sampler:
        train_labels = dataset.labels[train_indices]
        class_counts = torch.bincount(train_labels.long())
        if len(class_counts) >= 2 and class_counts.min() > 0:
            class_weights = 1.0 / class_counts.float()
            sample_weights = class_weights[train_labels.long()]
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_ds),
                replacement=True,
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

    return train_loader, val_loader
