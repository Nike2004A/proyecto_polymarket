"""Dataset y dataloaders para el modelo de series de tiempo."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

logger = logging.getLogger(__name__)


class TimeSeriesMarketDataset(Dataset):
    """Dataset TS construido a partir de secuencias de precio pre-snapshot."""

    def __init__(
        self,
        sequences: np.ndarray,
        sequence_lengths: np.ndarray,
        labels: np.ndarray,
        timestamps: np.ndarray | None = None,
        market_ids: np.ndarray | None = None,
        snapshot_prices: np.ndarray | None = None,
        metadata: dict | None = None,
    ):
        self.sequences = torch.FloatTensor(sequences)
        self.sequence_lengths = torch.LongTensor(sequence_lengths)
        self.labels = torch.FloatTensor(labels)
        self.timestamps = timestamps
        self.market_ids = market_ids
        self.snapshot_prices = snapshot_prices
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "sequence": self.sequences[idx],
            "sequence_length": self.sequence_lengths[idx],
            "label": self.labels[idx],
        }
        if self.market_ids is not None:
            item["market_id"] = self.market_ids[idx]
        if self.snapshot_prices is not None:
            item["snapshot_price"] = float(self.snapshot_prices[idx])
        return item

    @classmethod
    def from_numpy_dir(cls, directory: str) -> "TimeSeriesMarketDataset":
        path = Path(directory)
        metadata = {}
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)

        timestamps = None
        end_dates_path = path / "end_dates.npy"
        if end_dates_path.exists():
            timestamps = np.load(end_dates_path)

        market_ids = None
        market_ids_path = path / "market_ids.npy"
        if market_ids_path.exists():
            market_ids = np.load(market_ids_path, allow_pickle=True)

        snapshot_prices = None
        snapshot_prices_path = path / "snapshot_prices.npy"
        if snapshot_prices_path.exists():
            snapshot_prices = np.load(snapshot_prices_path)

        return cls(
            sequences=np.load(path / "sequences.npy"),
            sequence_lengths=np.load(path / "sequence_lengths.npy"),
            labels=np.load(path / "labels.npy"),
            timestamps=timestamps,
            market_ids=market_ids,
            snapshot_prices=snapshot_prices,
            metadata=metadata,
        )


def collate_ts_batch(batch: list[dict]) -> dict[str, torch.Tensor | list]:
    """
    Convierte secuencias left-padded almacenadas en right-padded para GRU packed.
    """
    sequences = torch.stack([item["sequence"] for item in batch])
    lengths = torch.stack([item["sequence_length"] for item in batch]).long()
    labels = torch.stack([item["label"] for item in batch]).float()

    right_padded = torch.zeros_like(sequences)
    for i, length in enumerate(lengths.tolist()):
        if length <= 0:
            continue
        right_padded[i, :length] = sequences[i, -length:]

    collated: dict[str, torch.Tensor | list] = {
        "sequence": right_padded,
        "sequence_length": lengths,
        "label": labels,
    }

    if "market_id" in batch[0]:
        collated["market_id"] = [item["market_id"] for item in batch]
    if "snapshot_price" in batch[0]:
        collated["snapshot_price"] = torch.tensor(
            [item["snapshot_price"] for item in batch],
            dtype=torch.float32,
        )
    return collated


def create_ts_dataloaders(
    dataset: TimeSeriesMarketDataset,
    batch_size: int = 64,
    val_split: float = 0.2,
    use_weighted_sampler: bool = True,
    num_workers: int = 0,
    temporal_split: bool = True,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Crea dataloaders train/val para el dataset TS."""
    n = len(dataset)
    n_val = int(n * val_split)
    n_train = n - n_val

    if temporal_split and dataset.timestamps is not None:
        sorted_indices = np.argsort(dataset.timestamps)
        train_indices = sorted_indices[:n_train].tolist()
        val_indices = sorted_indices[n_train:].tolist()
        logger.info(
            "Temporal split TS: train=%d (más antiguos), val=%d (más recientes)",
            len(train_indices),
            len(val_indices),
        )
    else:
        if temporal_split:
            raise ValueError(
                "temporal_split=True pero el dataset TS no tiene end_dates.npy."
            )
        generator = torch.Generator().manual_seed(seed)
        all_indices = torch.randperm(n, generator=generator).tolist()
        train_indices = all_indices[:n_train]
        val_indices = all_indices[n_train:]

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)

    train_sampler = None
    if use_weighted_sampler:
        train_labels = dataset.labels[train_indices].to(torch.long)
        class_counts = torch.bincount(train_labels)
        if len(class_counts) >= 2 and class_counts.min() > 0:
            class_weights = 1.0 / class_counts.float()
            sample_weights = class_weights[train_labels]
            generator = torch.Generator().manual_seed(seed)
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_ds),
                replacement=True,
                generator=generator,
            )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_ts_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_ts_batch,
    )
    return train_loader, val_loader
