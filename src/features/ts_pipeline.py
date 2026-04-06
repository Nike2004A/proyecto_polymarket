"""Pipeline reproducible para construir el dataset de series de tiempo."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from ..config import load_config
from ..data.preprocessing import compute_label, get_market_end_time
from .ts_sequence import (
    SEQUENCE_FEATURE_NAMES,
    build_market_price_sequence,
)

logger = logging.getLogger(__name__)


def build_ts_dataset(
    resolved_markets: list[dict],
    price_histories: dict,
    seq_len: int = 64,
    min_points: int = 5,
    snapshot_offset_days: int = 7,
    use_adaptive_cutoff: bool = True,
    limit: int | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Construye el dataset TS completo a partir de mercados resueltos."""
    sequences = []
    lengths = []
    labels = []
    end_dates = []
    market_ids = []
    snapshot_prices = []

    skipped = {
        "no_sequence": 0,
        "ambiguous": 0,
        "missing_end_date": 0,
    }

    source_markets = resolved_markets[:limit] if limit is not None else resolved_markets
    for market in source_markets:
        sequence, length, snapshot_price, _ = build_market_price_sequence(
            market,
            price_histories=price_histories,
            snapshot_offset_days=snapshot_offset_days,
            seq_len=seq_len,
            min_points=min_points,
            use_adaptive_cutoff=use_adaptive_cutoff,
        )
        if sequence is None or length is None or snapshot_price is None:
            skipped["no_sequence"] += 1
            continue

        label = compute_label(market, snapshot_price)
        if label == -1:
            skipped["ambiguous"] += 1
            continue

        end_time = get_market_end_time(market)
        if end_time is None:
            skipped["missing_end_date"] += 1
            continue

        sequences.append(sequence)
        lengths.append(length)
        labels.append(label)
        end_dates.append(end_time.timestamp())
        market_ids.append(str(market.get("id", "")))
        snapshot_prices.append(snapshot_price)

    if not sequences:
        raise ValueError("No se pudieron construir secuencias válidas.")

    dataset = {
        "sequences": np.asarray(sequences, dtype=np.float32),
        "sequence_lengths": np.asarray(lengths, dtype=np.int64),
        "labels": np.asarray(labels, dtype=np.float32),
        "end_dates": np.asarray(end_dates, dtype=np.float64),
        "market_ids": np.asarray(market_ids, dtype=str),
        "snapshot_prices": np.asarray(snapshot_prices, dtype=np.float32),
    }
    metadata = {
        "num_samples": int(len(sequences)),
        "seq_len": int(seq_len),
        "min_points": int(min_points),
        "num_features": len(SEQUENCE_FEATURE_NAMES),
        "feature_names": SEQUENCE_FEATURE_NAMES,
        "snapshot_offset_days": int(snapshot_offset_days),
        "adaptive_cutoff": bool(use_adaptive_cutoff),
        "skipped": skipped,
        "retention_rate": float(len(sequences) / len(source_markets)),
    }
    return dataset, metadata


def save_ts_dataset(
    dataset: dict[str, np.ndarray],
    metadata: dict,
    output_dir: str | Path,
) -> None:
    """Guarda el dataset TS y su metadata en disco."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "sequences.npy", dataset["sequences"])
    np.save(out / "sequence_lengths.npy", dataset["sequence_lengths"])
    np.save(out / "labels.npy", dataset["labels"])
    np.save(out / "end_dates.npy", dataset["end_dates"])
    np.save(out / "market_ids.npy", dataset["market_ids"])
    np.save(out / "snapshot_prices.npy", dataset["snapshot_prices"])

    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _load_raw_inputs(input_dir: Path) -> tuple[list[dict], dict]:
    with open(input_dir / "resolved_markets.json", encoding="utf-8") as f:
        resolved_markets = json.load(f)
    with open(input_dir / "price_histories.json", encoding="utf-8") as f:
        price_histories = json.load(f)
    return resolved_markets, price_histories


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Build time-series dataset")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--min-points", type=int, default=None)
    parser.add_argument("--snapshot-offset-days", type=int, default=None)
    parser.add_argument(
        "--no-adaptive-cutoff", action="store_true",
        help="Desactivar fallback adaptivo para mercados de vida corta",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    ts_cfg = cfg.get("ts_data", {})

    def _get(arg, cfg_val, default):
        return arg if arg is not None else cfg_val if cfg_val is not None else default

    input_dir = Path(args.input_dir or data_cfg.get("raw_dir", "data/raw"))
    output_dir = Path(args.output_dir or ts_cfg.get("processed_dir", "data/processed_ts"))
    seq_len = _get(args.seq_len, ts_cfg.get("seq_len"), 64)
    min_points = _get(args.min_points, ts_cfg.get("min_points"), 5)
    snapshot_offset_days = _get(
        args.snapshot_offset_days, ts_cfg.get("snapshot_offset_days"), 7
    )
    # --no-adaptive-cutoff toma precedencia; si no se pasa, leer config (default True)
    use_adaptive_cutoff = not args.no_adaptive_cutoff and ts_cfg.get("adaptive_cutoff", True)

    logger.info(
        "Construyendo dataset TS desde %s -> %s "
        "(seq_len=%d, min_points=%d, snapshot_offset=%d, adaptive_cutoff=%s)",
        input_dir,
        output_dir,
        seq_len,
        min_points,
        snapshot_offset_days,
        use_adaptive_cutoff,
    )

    resolved_markets, price_histories = _load_raw_inputs(input_dir)
    dataset, metadata = build_ts_dataset(
        resolved_markets,
        price_histories,
        seq_len=seq_len,
        min_points=min_points,
        snapshot_offset_days=snapshot_offset_days,
        use_adaptive_cutoff=use_adaptive_cutoff,
        limit=args.limit,
    )
    save_ts_dataset(dataset, metadata, output_dir)

    logger.info("Dataset TS guardado en %s", output_dir)
    logger.info(
        "Muestras válidas: %d | retention: %.1f%% | skipped=%s",
        metadata["num_samples"],
        100 * metadata["retention_rate"],
        metadata["skipped"],
    )


if __name__ == "__main__":
    main()
