#!/usr/bin/env python3
"""Minimal runner for the rebuilt snapshot-based pipeline."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run")

ALL_STEPS = ["features", "train", "score"]


def _run(cmd: list[str]) -> None:
    log.info("$ %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)


def step_features(cfg: str, force: bool) -> None:
    sentinel = ROOT / "data" / "processed" / "numerical_features.npy"
    if sentinel.exists() and not force:
        log.info("SKIP features — %s ya existe", sentinel.relative_to(ROOT))
        return
    _run([sys.executable, "-m", "src.features.pipeline", "--config", cfg])


def step_train(cfg: str, force: bool) -> None:
    sentinel = ROOT / "data" / "models" / "registry" / "primary_model.json"
    if sentinel.exists() and not force:
        log.info("SKIP train — %s ya existe", sentinel.relative_to(ROOT))
        return
    _run([sys.executable, "-m", "src.model.train", "--config", cfg, "--only", "all"])


def step_score(cfg: str, force: bool) -> None:
    _run([sys.executable, "-m", "src.scoring.scorer", "--config", cfg, "--all-models"])


STEP_FN = {
    "features": step_features,
    "train": step_train,
    "score": step_score,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument("--steps", default="features,train,score")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = [step.strip() for step in args.steps.split(",") if step.strip()]
    invalid = [step for step in steps if step not in STEP_FN]
    if invalid:
        raise SystemExit(f"Pasos inválidos: {invalid}. Disponibles: {ALL_STEPS}")

    log.info("Pipeline: %s", " -> ".join(steps))
    for step in steps:
        log.info("==== %s ====", step.upper())
        STEP_FN[step](args.config, args.force)


if __name__ == "__main__":
    main()
