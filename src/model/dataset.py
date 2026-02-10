"""Dataset y DataLoaders para el entrenamiento del modelo."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler


class PolymarketDataset(Dataset):
    """Dataset de mercados de Polymarket para entrenamiento."""

    def __init__(
        self,
        numerical_features: np.ndarray,
        category_ids: np.ndarray,
        text_embeddings: np.ndarray,
        labels: np.ndarray,
    ):
        self.numerical = torch.FloatTensor(numerical_features)
        self.categories = torch.LongTensor(category_ids)
        self.text_emb = torch.FloatTensor(text_embeddings)
        self.labels = torch.FloatTensor(labels)

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
        return cls(
            numerical_features=np.load(path / "numerical_features.npy"),
            category_ids=np.load(path / "category_ids.npy"),
            text_embeddings=np.load(path / "text_embeddings.npy"),
            labels=np.load(path / "labels.npy"),
        )


def create_dataloaders(
    dataset: PolymarketDataset,
    batch_size: int = 64,
    val_split: float = 0.2,
    use_weighted_sampler: bool = True,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Divide en train/val y crea DataLoaders.

    Args:
        dataset: PolymarketDataset.
        batch_size: Tamaño de batch.
        val_split: Fracción para validación.
        use_weighted_sampler: Si usar muestreo ponderado para balancear clases.
        num_workers: Número de workers para carga de datos.

    Returns:
        (train_loader, val_loader)
    """
    n = len(dataset)
    n_val = int(n * val_split)
    n_train = n - n_val

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Weighted sampler para balancear clases
    train_sampler = None
    shuffle = True
    if use_weighted_sampler:
        train_labels = dataset.labels[train_ds.indices]
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
