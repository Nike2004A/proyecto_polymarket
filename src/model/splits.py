"""Utilities for reproducible dataset splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplit:
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]
    strategy: str
    seed: int
    val_split: float
    test_split: float
    temporal_cutoffs: dict[str, int | None] | None = None

    def to_metadata(self) -> dict:
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "val_split": self.val_split,
            "test_split": self.test_split,
            "train_size": len(self.train_indices),
            "val_size": len(self.val_indices),
            "test_size": len(self.test_indices),
            "temporal_cutoffs": self.temporal_cutoffs,
        }


def build_dataset_split(
    *,
    n_samples: int,
    labels: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    val_split: float = 0.15,
    test_split: float = 0.15,
    strategy: str = "random",
    seed: int = 42,
) -> DatasetSplit:
    """Build train/val/test indices with optional grouped temporal logic."""
    _validate_split_args(n_samples=n_samples, val_split=val_split, test_split=test_split)

    if strategy == "temporal_grouped":
        return _build_temporal_grouped_split(
            n_samples=n_samples,
            timestamps=timestamps,
            groups=groups,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
        )
    if strategy == "temporal":
        if groups is not None:
            return _build_temporal_grouped_split(
                n_samples=n_samples,
                timestamps=timestamps,
                groups=groups,
                val_split=val_split,
                test_split=test_split,
                seed=seed,
            )
        return _build_temporal_split(
            n_samples=n_samples,
            timestamps=timestamps,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
        )
    if strategy == "random":
        return _build_random_split(
            n_samples=n_samples,
            labels=labels,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
        )
    raise ValueError(f"split strategy no soportada: {strategy}")


def _validate_split_args(*, n_samples: int, val_split: float, test_split: float) -> None:
    if n_samples <= 2:
        raise ValueError("Se requieren al menos 3 muestras para train/val/test.")
    if not 0 < val_split < 1:
        raise ValueError("val_split debe estar en (0, 1).")
    if not 0 <= test_split < 1:
        raise ValueError("test_split debe estar en [0, 1).")
    if val_split + test_split >= 1:
        raise ValueError("val_split + test_split debe ser menor a 1.")


def _prepare_stratify_labels(labels: np.ndarray | None) -> np.ndarray | None:
    if labels is None:
        return None
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or counts.min() < 2:
        return None
    return labels


def _build_random_split(
    *,
    n_samples: int,
    labels: np.ndarray | None,
    val_split: float,
    test_split: float,
    seed: int,
) -> DatasetSplit:
    indices = np.arange(n_samples)
    stratify_labels = _prepare_stratify_labels(labels)

    if test_split > 0:
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=test_split,
            random_state=seed,
            shuffle=True,
            stratify=stratify_labels,
        )
    else:
        train_val_indices = indices
        test_indices = np.array([], dtype=int)

    val_ratio = val_split / (1.0 - test_split)
    stratify_train_val = None
    if labels is not None:
        stratify_train_val = _prepare_stratify_labels(np.asarray(labels)[train_val_indices])

    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_ratio,
        random_state=seed + 1,
        shuffle=True,
        stratify=stratify_train_val,
    )

    return DatasetSplit(
        train_indices=np.asarray(train_indices, dtype=int).tolist(),
        val_indices=np.asarray(val_indices, dtype=int).tolist(),
        test_indices=np.asarray(test_indices, dtype=int).tolist(),
        strategy="random",
        seed=seed,
        val_split=val_split,
        test_split=test_split,
        temporal_cutoffs=None,
    )


def _build_temporal_split(
    *,
    n_samples: int,
    timestamps: np.ndarray | None,
    val_split: float,
    test_split: float,
    seed: int,
) -> DatasetSplit:
    if timestamps is None:
        raise ValueError("split_strategy='temporal' requiere timestamps.")

    ordered_indices = np.argsort(np.asarray(timestamps))
    n_val = int(n_samples * val_split)
    n_test = int(n_samples * test_split)
    n_train = n_samples - n_val - n_test

    train_indices = ordered_indices[:n_train]
    val_indices = ordered_indices[n_train:n_train + n_val]
    test_indices = ordered_indices[n_train + n_val:]

    cutoff_val = int(np.asarray(timestamps)[val_indices[0]]) if len(val_indices) else None
    cutoff_test = int(np.asarray(timestamps)[test_indices[0]]) if len(test_indices) else None
    return DatasetSplit(
        train_indices=train_indices.tolist(),
        val_indices=val_indices.tolist(),
        test_indices=test_indices.tolist(),
        strategy="temporal",
        seed=seed,
        val_split=val_split,
        test_split=test_split,
        temporal_cutoffs={
            "val_start_timestamp": cutoff_val,
            "test_start_timestamp": cutoff_test,
        },
    )


def _build_temporal_grouped_split(
    *,
    n_samples: int,
    timestamps: np.ndarray | None,
    groups: np.ndarray | None,
    val_split: float,
    test_split: float,
    seed: int,
) -> DatasetSplit:
    if timestamps is None or groups is None:
        raise ValueError("split_strategy='temporal_grouped' requiere timestamps y groups.")

    timestamps = np.asarray(timestamps)
    groups = np.asarray(groups)
    n_val_target = int(n_samples * val_split)
    n_test_target = int(n_samples * test_split)
    n_train_target = n_samples - n_val_target - n_test_target

    unique_groups = np.unique(groups)
    group_rows = []
    for group in unique_groups:
        mask = groups == group
        group_rows.append({
            "group": group,
            "timestamp": int(np.max(timestamps[mask])),
            "indices": np.flatnonzero(mask),
            "count": int(mask.sum()),
        })
    group_rows.sort(key=lambda row: row["timestamp"])

    train_groups: list[dict] = []
    val_groups: list[dict] = []
    test_groups: list[dict] = []
    assigned = 0

    for row in group_rows:
        if assigned < n_train_target:
            train_groups.append(row)
        elif assigned < n_train_target + n_val_target:
            val_groups.append(row)
        else:
            test_groups.append(row)
        assigned += row["count"]

    if not val_groups and test_groups:
        val_groups.append(test_groups.pop(0))
    if not test_groups and val_groups:
        test_groups.append(val_groups.pop(-1))

    train_indices = np.concatenate([row["indices"] for row in train_groups]) if train_groups else np.array([], dtype=int)
    val_indices = np.concatenate([row["indices"] for row in val_groups]) if val_groups else np.array([], dtype=int)
    test_indices = np.concatenate([row["indices"] for row in test_groups]) if test_groups else np.array([], dtype=int)

    cutoff_val = int(np.min(timestamps[val_indices])) if val_indices.size else None
    cutoff_test = int(np.min(timestamps[test_indices])) if test_indices.size else None

    return DatasetSplit(
        train_indices=np.sort(train_indices).astype(int).tolist(),
        val_indices=np.sort(val_indices).astype(int).tolist(),
        test_indices=np.sort(test_indices).astype(int).tolist(),
        strategy="temporal_grouped",
        seed=seed,
        val_split=val_split,
        test_split=test_split,
        temporal_cutoffs={
            "val_start_timestamp": cutoff_val,
            "test_start_timestamp": cutoff_test,
        },
    )
