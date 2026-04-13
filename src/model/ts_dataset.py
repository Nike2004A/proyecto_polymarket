"""Dataset and dataloaders for the fixed-grid sequence model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .splits import build_dataset_split


class TimeSeriesMarketDataset(Dataset):
    """Sequence dataset with a static branch aligned to the tabular snapshots."""

    def __init__(
        self,
        sequences: np.ndarray,
        sequence_lengths: np.ndarray,
        static_numerical: np.ndarray,
        category_ids: np.ndarray,
        text_embeddings: np.ndarray,
        labels: np.ndarray,
        targets: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
        groups: np.ndarray | None = None,
        market_ids: np.ndarray | None = None,
        snapshot_prices: np.ndarray | None = None,
        snapshot_times: np.ndarray | None = None,
        days_to_end: np.ndarray | None = None,
        metadata: dict | None = None,
    ):
        self.sequences = torch.FloatTensor(sequences)
        self.sequence_lengths = torch.LongTensor(sequence_lengths)
        self.static_numerical = torch.FloatTensor(static_numerical)
        self.categories = torch.LongTensor(category_ids)
        self.text_emb = torch.FloatTensor(text_embeddings)
        self.labels = torch.FloatTensor(labels)
        self.targets = torch.FloatTensor(targets if targets is not None else labels)
        self.timestamps = timestamps
        self.groups = groups
        self.market_ids = market_ids
        self.snapshot_prices = snapshot_prices
        self.snapshot_times = snapshot_times
        self.days_to_end = days_to_end
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "sequence": self.sequences[idx],
            "sequence_length": self.sequence_lengths[idx],
            "static_numerical": self.static_numerical[idx],
            "category": self.categories[idx],
            "text_emb": self.text_emb[idx],
            "label": self.labels[idx],
            "target": self.targets[idx],
        }
        if self.market_ids is not None:
            item["market_id"] = str(self.market_ids[idx])
        if self.snapshot_prices is not None:
            item["snapshot_price"] = float(self.snapshot_prices[idx])
        if self.snapshot_times is not None:
            item["snapshot_time"] = int(self.snapshot_times[idx])
        if self.days_to_end is not None:
            item["days_to_end"] = float(self.days_to_end[idx])
        return item

    @classmethod
    def from_numpy_dir(cls, directory: str) -> "TimeSeriesMarketDataset":
        path = Path(directory)
        metadata = {}
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)

        return cls(
            sequences=np.load(path / "sequences.npy"),
            sequence_lengths=np.load(path / "sequence_lengths.npy"),
            static_numerical=np.load(path / "static_numerical.npy"),
            category_ids=np.load(path / "category_ids.npy"),
            text_embeddings=np.load(path / "text_embeddings.npy"),
            labels=np.load(path / "labels.npy"),
            targets=_maybe_load(path / "targets.npy"),
            timestamps=_maybe_load(path / "end_dates.npy"),
            groups=_maybe_load(path / "market_ids.npy", allow_pickle=True),
            market_ids=_maybe_load(path / "market_ids.npy", allow_pickle=True),
            snapshot_prices=_maybe_load(path / "snapshot_prices.npy"),
            snapshot_times=_maybe_load(path / "snapshot_times.npy"),
            days_to_end=_maybe_load(path / "days_to_end.npy"),
            metadata=metadata,
        )


def create_ts_train_val_test_dataloaders(
    dataset: TimeSeriesMarketDataset,
    batch_size: int = 128,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 0,
    split_strategy: str = "temporal_grouped",
    seed: int = 42,
    use_weighted_sampler: bool | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader | None, dict]:
    split = build_dataset_split(
        n_samples=len(dataset),
        labels=dataset.labels.numpy(),
        timestamps=dataset.timestamps,
        groups=dataset.groups,
        val_split=val_split,
        test_split=test_split,
        strategy=split_strategy,
        seed=seed,
    )

    train_loader = DataLoader(
        Subset(dataset, split.train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        Subset(dataset, split.val_indices),
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


def _maybe_load(path: Path, allow_pickle: bool = False):
    if not path.exists():
        return None
    return np.load(path, allow_pickle=allow_pickle)
