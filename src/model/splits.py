"""Utilities para dividir datasets en train/val/test."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplit:
    """Índices de train/val/test junto con metadata útil para reproducibilidad."""

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
    val_split: float = 0.15,
    test_split: float = 0.15,
    strategy: str = "random",
    seed: int = 42,
) -> DatasetSplit:
    """Construye índices reproducibles para train/val/test."""
    _validate_split_args(n_samples=n_samples, val_split=val_split, test_split=test_split)

    if strategy == "temporal":
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


def _validate_split_args(
    *,
    n_samples: int,
    val_split: float,
    test_split: float,
) -> None:
    if n_samples <= 2:
        raise ValueError("Se requieren al menos 3 muestras para train/val/test.")
    if not 0 < val_split < 1:
        raise ValueError("val_split debe estar en (0, 1).")
    if not 0 <= test_split < 1:
        raise ValueError("test_split debe estar en [0, 1).")
    if val_split + test_split >= 1:
        raise ValueError("val_split + test_split debe ser menor a 1.")

    n_val = int(n_samples * val_split)
    n_test = int(n_samples * test_split)
    n_train = n_samples - n_val - n_test
    if n_val <= 0:
        raise ValueError("val_split deja 0 muestras de validación; súbelo un poco.")
    if test_split > 0 and n_test <= 0:
        raise ValueError("test_split deja 0 muestras de test; súbelo un poco.")
    if n_train <= 0:
        raise ValueError("Los splits dejan 0 muestras de entrenamiento.")


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
        train_val_indices, test_indices = _split_once(
            indices=indices,
            test_size=test_split,
            seed=seed,
            stratify_labels=stratify_labels,
        )
    else:
        train_val_indices = indices
        test_indices = np.array([], dtype=int)

    val_ratio_within_train_val = val_split / (1.0 - test_split)
    stratify_train_val = None
    if stratify_labels is not None:
        stratify_train_val = np.asarray(labels)[train_val_indices]
        stratify_train_val = _prepare_stratify_labels(stratify_train_val)

    train_indices, val_indices = _split_once(
        indices=train_val_indices,
        test_size=val_ratio_within_train_val,
        seed=seed + 1,
        stratify_labels=stratify_train_val,
    )

    return DatasetSplit(
        train_indices=train_indices.tolist(),
        val_indices=val_indices.tolist(),
        test_indices=test_indices.tolist(),
        strategy="random",
        seed=seed,
        val_split=val_split,
        test_split=test_split,
        temporal_cutoffs=None,
    )


def _prepare_stratify_labels(labels: np.ndarray | None) -> np.ndarray | None:
    if labels is None:
        return None
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or counts.min() < 2:
        return None
    return labels


def _split_once(
    *,
    indices: np.ndarray,
    test_size: float,
    seed: int,
    stratify_labels: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        train_idx, holdout_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify_labels,
        )
    except ValueError:
        train_idx, holdout_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )
    return np.asarray(train_idx, dtype=int), np.asarray(holdout_idx, dtype=int)


def _build_temporal_split(
    *,
    n_samples: int,
    timestamps: np.ndarray | None,
    val_split: float,
    test_split: float,
    seed: int,
) -> DatasetSplit:
    if timestamps is None:
        raise ValueError(
            "split_strategy='temporal' requiere timestamps disponibles."
        )

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
