"""Compatibility wrapper for the unified snapshot dataset builder."""

from __future__ import annotations

import argparse
import logging

from ..config import load_config
from .pipeline import build_and_save_from_raw

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Build snapshot-based TS dataset")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dummy-text", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    feature_cfg = cfg.get("features", {})
    ts_cfg = cfg.get("ts_data", {})

    input_dir = args.input_dir or data_cfg.get("raw_dir", "data/raw")
    output_dir = args.output_dir or ts_cfg.get("processed_dir", "data/processed_ts")
    build_and_save_from_raw(
        input_dir=input_dir,
        tabular_output_dir=data_cfg.get("processed_dir", "data/processed"),
        ts_output_dir=output_dir,
        horizons_days=feature_cfg.get("snapshot_horizons_days", [1, 3, 7, 14, 30]),
        lookback_days=ts_cfg.get("sequence_lookback_days", 30),
        step_hours=ts_cfg.get("sequence_grid_hours", 12),
        use_dummy_text=bool(args.dummy_text or feature_cfg.get("use_dummy_text", False)),
    )
    logger.info("TS dataset saved to %s", output_dir)


if __name__ == "__main__":
    main()
