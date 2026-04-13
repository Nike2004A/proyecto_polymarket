"""Explicit scorer for the fixed-grid sequence model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ..config import load_config
from .common import load_active_context, load_ts_bundle
from .scorer import score_active_markets, summarize_scored_markets

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Score active markets with the GRU bundle")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--models-root", default=None)
    parser.add_argument("--pipeline-dir", default=None)
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    scoring_cfg = cfg.get("scoring", {})
    training_cfg = cfg.get("training", {})

    raw_dir = args.raw_dir or data_cfg.get("raw_dir", "data/raw")
    models_root = args.models_root or str(Path(training_cfg.get("save_dir", "data/models/market_value_baseline")).parent)
    pipeline_dir = args.pipeline_dir or str(Path(data_cfg.get("processed_dir", "data/processed")) / "pipeline")
    top_k = args.top or scoring_cfg.get("top_k", 20)

    active_markets, price_histories = load_active_context(raw_dir)
    bundle = load_ts_bundle(Path(models_root) / "price_sequence_gru", pipeline_dir)
    df = score_active_markets(
        bundle,
        active_markets=active_markets,
        price_histories=price_histories,
        top_k=top_k,
    )
    summary = summarize_scored_markets(df)

    output_dir = Path(args.output or Path(models_root) / "price_sequence_gru")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "live_scores.csv", index=False)
    with open(output_dir / "live_scoring_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("GRU scored rows=%d signals=%s", len(df), summary.get("signals", {}))
    if not df.empty:
        print(df.head(top_k).to_string(index=False))


if __name__ == "__main__":
    main()
