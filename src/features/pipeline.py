"""Snapshot-based dataset builder and feature pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..config import load_config
from ..data.snapshots import build_snapshot_samples, prepare_history_map
from .categorical import CategoryEncoder
from .numerical import (
    NUMERICAL_FEATURE_NAMES,
    NUM_NUMERICAL_FEATURES,
    extract_numerical_features,
    extract_numerical_features_batch,
)
from .text import DummyTextEncoder, TextEncoder
from .ts_sequence import build_grid_sequence, SEQUENCE_FEATURE_NAMES

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Scaler + encoders for snapshot-based tabular features."""

    _shared_text_encoder: TextEncoder | None = None

    def __init__(
        self,
        text_encoder: TextEncoder | DummyTextEncoder | None = None,
        category_encoder: CategoryEncoder | None = None,
        scaler: StandardScaler | None = None,
        use_dummy_text: bool = False,
    ):
        if text_encoder is not None:
            self.text_encoder = text_encoder
        elif use_dummy_text:
            self.text_encoder = DummyTextEncoder()
        else:
            if FeaturePipeline._shared_text_encoder is None:
                FeaturePipeline._shared_text_encoder = TextEncoder()
            self.text_encoder = FeaturePipeline._shared_text_encoder

        self.category_encoder = category_encoder or CategoryEncoder()
        self.scaler = scaler or StandardScaler()
        self._fitted = False
        self._text_cache: dict[str, np.ndarray] = {}

    @property
    def text_embed_dim(self) -> int:
        return self.text_encoder.embed_dim

    @property
    def num_numerical_features(self) -> int:
        return NUM_NUMERICAL_FEATURES

    @property
    def num_categories(self) -> int:
        return self.category_encoder.num_categories

    def transform_single(
        self,
        market: dict,
        price_history=None,
        snapshot_time: pd.Timestamp | None = None,
        days_to_end: float | None = None,
        order_book=None,
    ) -> dict:
        """Transform a single live or historical market snapshot."""
        numerical = extract_numerical_features(
            market,
            price_history=price_history,
            snapshot_time=snapshot_time,
            days_to_end=days_to_end,
        )
        if self._fitted:
            numerical = self.scaler.transform(numerical.reshape(1, -1)).squeeze(0)

        category_id = self.category_encoder.encode(market)
        text_embedding = self._encode_market_text(market, key=str(market.get("id", market.get("question", ""))))
        return {
            "numerical": numerical.astype(np.float32),
            "category_id": int(category_id),
            "text_embedding": text_embedding.astype(np.float32),
        }

    def fit_transform_batch(self, samples: list[dict]) -> dict:
        """Fit scaler/encoders on a snapshot sample batch and transform it."""
        markets = [sample["market"] for sample in samples]
        numerical = extract_numerical_features_batch(samples)
        self.scaler.fit(numerical)
        numerical_scaled = self.scaler.transform(numerical).astype(np.float32)
        self._fitted = True

        category_ids = self.category_encoder.encode_batch(markets)
        text_embeddings = self._encode_samples_text(samples)
        return {
            "numerical": numerical_scaled,
            "category_ids": category_ids.astype(np.int64),
            "text_embeddings": text_embeddings.astype(np.float32),
        }

    def transform_batch(self, samples: list[dict]) -> dict:
        """Transform a batch without refitting the scaler."""
        markets = [sample["market"] for sample in samples]
        numerical = extract_numerical_features_batch(samples)
        if self._fitted:
            numerical = self.scaler.transform(numerical).astype(np.float32)
        category_ids = self.category_encoder.encode_batch(markets)
        text_embeddings = self._encode_samples_text(samples)
        return {
            "numerical": numerical.astype(np.float32),
            "category_ids": category_ids.astype(np.int64),
            "text_embeddings": text_embeddings.astype(np.float32),
        }

    def save(self, directory: str) -> None:
        import joblib

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path / "scaler.pkl")
        self.category_encoder.save(str(path / "category_encoder.json"))
        metadata = {
            "fitted": self._fitted,
            "num_numerical": NUM_NUMERICAL_FEATURES,
            "num_categories": self.category_encoder.num_categories,
            "text_embed_dim": self.text_embed_dim,
            "feature_names": NUMERICAL_FEATURE_NAMES,
        }
        with open(path / "pipeline_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, directory: str, use_dummy_text: bool = False) -> "FeaturePipeline":
        import joblib

        path = Path(directory)
        scaler = joblib.load(path / "scaler.pkl")
        category_encoder = CategoryEncoder.load(str(path / "category_encoder.json"))
        pipeline = cls(
            category_encoder=category_encoder,
            scaler=scaler,
            use_dummy_text=use_dummy_text,
        )
        pipeline._fitted = True
        return pipeline

    def warm_text_cache(self, markets: list[dict]) -> None:
        """Batch-encode market questions to avoid per-market live scoring overhead."""
        missing_keys: list[str] = []
        missing_texts: list[str] = []
        for market in markets:
            key = str(market.get("id", market.get("question", "")))
            if key in self._text_cache:
                continue
            missing_keys.append(key)
            missing_texts.append(market.get("question", ""))

        if not missing_keys:
            return

        embeddings = self.text_encoder.encode_batch(
            missing_texts,
            show_progress_bar=False,
        )
        for key, embedding in zip(missing_keys, embeddings):
            self._text_cache[key] = np.asarray(embedding, dtype=np.float32)

    def _encode_samples_text(self, samples: list[dict]) -> np.ndarray:
        missing_keys: list[str] = []
        missing_texts: list[str] = []
        ordered_keys: list[str] = []

        for sample in samples:
            market = sample["market"]
            key = str(sample["market_id"])
            ordered_keys.append(key)
            if key in self._text_cache:
                continue
            missing_keys.append(key)
            missing_texts.append(market.get("question", ""))

        if missing_keys:
            batch_embeddings = self.text_encoder.encode_batch(
                missing_texts,
                show_progress_bar=True,
            )
            for key, embedding in zip(missing_keys, batch_embeddings):
                self._text_cache[key] = np.asarray(embedding, dtype=np.float32)

        return np.stack([self._text_cache[key] for key in ordered_keys]).astype(np.float32)

    def _encode_market_text(self, market: dict, key: str) -> np.ndarray:
        if key not in self._text_cache:
            self._text_cache[key] = self.text_encoder.encode(market.get("question", "")).astype(np.float32)
        return self._text_cache[key]


def build_snapshot_datasets(
    resolved_markets: list[dict],
    price_histories: dict,
    horizons_days: list[int],
    lookback_days: int,
    step_hours: int,
    use_dummy_text: bool = False,
) -> tuple[dict, dict, dict]:
    """Build aligned tabular and sequence datasets from cached raw inputs."""
    prepared_histories = prepare_history_map(price_histories)
    samples: list[dict] = []
    skipped = {
        "missing_history": 0,
        "no_valid_horizons": 0,
        "no_sequence": 0,
        "ambiguous": 0,
    }

    for market in resolved_markets:
        market_id = str(market.get("id", ""))
        history = prepared_histories.get(market_id)
        if history is None or history.is_empty:
            skipped["missing_history"] += 1
            continue

        market_samples = build_snapshot_samples(
            market,
            history,
            horizons_days=horizons_days,
            min_history_points=2,
        )
        if not market_samples:
            skipped["no_valid_horizons"] += 1
            continue

        for sample in market_samples:
            sequence, length, snapshot_price = build_grid_sequence(
                sample["history"],
                snapshot_time=sample["snapshot_time"],
                lookback_days=lookback_days,
                step_hours=step_hours,
            )
            if sequence is None or length is None or snapshot_price is None:
                skipped["no_sequence"] += 1
                continue
            sample["sequence"] = sequence
            sample["sequence_length"] = int(length)
            sample["snapshot_price_yes"] = float(snapshot_price)
            samples.append(sample)

    if not samples:
        raise ValueError("No se pudieron construir muestras snapshot válidas.")

    pipeline = FeaturePipeline(use_dummy_text=use_dummy_text)
    tabular = pipeline.fit_transform_batch(samples)
    sequence_lengths = np.asarray([sample["sequence_length"] for sample in samples], dtype=np.int64)
    sequences = np.stack([sample["sequence"] for sample in samples]).astype(np.float32)
    labels = np.asarray([sample["label_yes"] for sample in samples], dtype=np.float32)
    targets = np.asarray([sample["target_residual"] for sample in samples], dtype=np.float32)
    market_ids = np.asarray([sample["market_id"] for sample in samples], dtype=str)
    snapshot_times = np.asarray([sample["snapshot_ts"] for sample in samples], dtype=np.int64)
    end_dates = np.asarray([sample["end_ts"] for sample in samples], dtype=np.int64)
    days_to_end = np.asarray([sample["days_to_end"] for sample in samples], dtype=np.int64)
    snapshot_prices = np.asarray([sample["snapshot_price_yes"] for sample in samples], dtype=np.float32)

    tabular_dataset = {
        "numerical": tabular["numerical"],
        "category_ids": tabular["category_ids"],
        "text_embeddings": tabular["text_embeddings"],
        "labels": labels,
        "targets": targets,
        "market_ids": market_ids,
        "snapshot_times": snapshot_times,
        "end_dates": end_dates,
        "days_to_end": days_to_end,
        "snapshot_prices": snapshot_prices,
    }
    ts_dataset = {
        "sequences": sequences,
        "sequence_lengths": sequence_lengths,
        "static_numerical": tabular["numerical"],
        "category_ids": tabular["category_ids"],
        "text_embeddings": tabular["text_embeddings"],
        "labels": labels,
        "targets": targets,
        "market_ids": market_ids,
        "snapshot_times": snapshot_times,
        "end_dates": end_dates,
        "days_to_end": days_to_end,
        "snapshot_prices": snapshot_prices,
    }

    metadata = {
        "num_samples": int(labels.size),
        "num_markets": int(len({sample["market_id"] for sample in samples})),
        "snapshot_horizons_days": [int(x) for x in horizons_days],
        "sequence_lookback_days": int(lookback_days),
        "sequence_grid_hours": int(step_hours),
        "tabular_feature_names": NUMERICAL_FEATURE_NAMES,
        "sequence_feature_names": SEQUENCE_FEATURE_NAMES,
        "skipped": skipped,
        "market_positive_rate": float(labels.mean()),
        "target_name": "residual_yes_minus_price",
        "benchmark": "market_price_yes",
    }

    return tabular_dataset, ts_dataset, {"metadata": metadata, "pipeline": pipeline}


def save_snapshot_datasets(
    tabular_dataset: dict,
    ts_dataset: dict,
    metadata: dict,
    pipeline: FeaturePipeline,
    tabular_dir: str | Path,
    ts_dir: str | Path,
) -> None:
    """Persist aligned tabular and sequence datasets to disk."""
    tab_dir = Path(tabular_dir)
    seq_dir = Path(ts_dir)
    tab_dir.mkdir(parents=True, exist_ok=True)
    seq_dir.mkdir(parents=True, exist_ok=True)

    np.save(tab_dir / "numerical_features.npy", tabular_dataset["numerical"])
    np.save(tab_dir / "category_ids.npy", tabular_dataset["category_ids"])
    np.save(tab_dir / "text_embeddings.npy", tabular_dataset["text_embeddings"])
    np.save(tab_dir / "labels.npy", tabular_dataset["labels"])
    np.save(tab_dir / "targets.npy", tabular_dataset["targets"])
    np.save(tab_dir / "market_ids.npy", tabular_dataset["market_ids"])
    np.save(tab_dir / "snapshot_times.npy", tabular_dataset["snapshot_times"])
    np.save(tab_dir / "end_dates.npy", tabular_dataset["end_dates"])
    np.save(tab_dir / "days_to_end.npy", tabular_dataset["days_to_end"])
    np.save(tab_dir / "snapshot_prices.npy", tabular_dataset["snapshot_prices"])
    pipeline.save(str(tab_dir / "pipeline"))

    np.save(seq_dir / "sequences.npy", ts_dataset["sequences"])
    np.save(seq_dir / "sequence_lengths.npy", ts_dataset["sequence_lengths"])
    np.save(seq_dir / "static_numerical.npy", ts_dataset["static_numerical"])
    np.save(seq_dir / "category_ids.npy", ts_dataset["category_ids"])
    np.save(seq_dir / "text_embeddings.npy", ts_dataset["text_embeddings"])
    np.save(seq_dir / "labels.npy", ts_dataset["labels"])
    np.save(seq_dir / "targets.npy", ts_dataset["targets"])
    np.save(seq_dir / "market_ids.npy", ts_dataset["market_ids"])
    np.save(seq_dir / "snapshot_times.npy", ts_dataset["snapshot_times"])
    np.save(seq_dir / "end_dates.npy", ts_dataset["end_dates"])
    np.save(seq_dir / "days_to_end.npy", ts_dataset["days_to_end"])
    np.save(seq_dir / "snapshot_prices.npy", ts_dataset["snapshot_prices"])

    with open(tab_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(seq_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def build_and_save_from_raw(
    input_dir: str | Path,
    tabular_output_dir: str | Path,
    ts_output_dir: str | Path,
    horizons_days: list[int],
    lookback_days: int,
    step_hours: int,
    use_dummy_text: bool = False,
) -> dict:
    """Load cached raw JSONs, build aligned datasets, and save them."""
    input_path = Path(input_dir)
    with open(input_path / "resolved_markets.json", encoding="utf-8") as f:
        resolved_markets = json.load(f)
    with open(input_path / "price_histories.json", encoding="utf-8") as f:
        price_histories = json.load(f)

    tabular_dataset, ts_dataset, aux = build_snapshot_datasets(
        resolved_markets=resolved_markets,
        price_histories=price_histories,
        horizons_days=horizons_days,
        lookback_days=lookback_days,
        step_hours=step_hours,
        use_dummy_text=use_dummy_text,
    )
    save_snapshot_datasets(
        tabular_dataset,
        ts_dataset,
        metadata=aux["metadata"],
        pipeline=aux["pipeline"],
        tabular_dir=tabular_output_dir,
        ts_dir=ts_output_dir,
    )
    return aux["metadata"]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Build snapshot-based tabular + TS datasets")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ts-output-dir", default=None)
    parser.add_argument("--dummy-text", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    feature_cfg = cfg.get("features", {})
    ts_cfg = cfg.get("ts_data", {})

    input_dir = args.input_dir or data_cfg.get("raw_dir", "data/raw")
    output_dir = args.output_dir or data_cfg.get("processed_dir", "data/processed")
    ts_output_dir = args.ts_output_dir or ts_cfg.get("processed_dir", "data/processed_ts")
    horizons_days = feature_cfg.get("snapshot_horizons_days", [1, 3, 7, 14, 30])
    lookback_days = ts_cfg.get("sequence_lookback_days", 30)
    step_hours = ts_cfg.get("sequence_grid_hours", 12)
    use_dummy_text = bool(args.dummy_text or feature_cfg.get("use_dummy_text", False))

    logger.info(
        "Building snapshot datasets from %s -> %s / %s | horizons=%s lookback=%sd step=%sh",
        input_dir,
        output_dir,
        ts_output_dir,
        horizons_days,
        lookback_days,
        step_hours,
    )

    metadata = build_and_save_from_raw(
        input_dir=input_dir,
        tabular_output_dir=output_dir,
        ts_output_dir=ts_output_dir,
        horizons_days=horizons_days,
        lookback_days=lookback_days,
        step_hours=step_hours,
        use_dummy_text=use_dummy_text,
    )
    logger.info(
        "Snapshot datasets saved | samples=%d markets=%d positive_rate=%.3f skipped=%s",
        metadata["num_samples"],
        metadata["num_markets"],
        metadata["market_positive_rate"],
        metadata["skipped"],
    )


if __name__ == "__main__":
    main()
