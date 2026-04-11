#!/usr/bin/env python3
"""
run.py — Pipeline completo de Polymarket ML.

Ejecuta todos los pasos desde la descarga de datos hasta los notebooks de análisis.
Cada paso verifica si sus artefactos ya existen y los salta a menos que se use --force.

Uso:
    python run.py                              # todo el pipeline
    python run.py --steps fetch,features       # solo descarga + features
    python run.py --steps train --force        # reentrenar aunque ya existan modelos
    python run.py --skip fetch                 # todo menos la descarga
    python run.py --steps notebooks            # solo ejecutar notebooks
    python run.py --only baseline              # solo el flujo baseline (features + train)
    python run.py --only gru                   # solo el flujo GRU (features_ts + train_ts)

Pasos disponibles (en orden):
    fetch         Descarga datos desde la API de Polymarket
    features      Genera data/processed/ (baseline Wide & Deep)
    features_ts   Genera data/processed_ts/ (GRU)
    train         Entrena MarketValueNet
    train_ts      Entrena PriceSequenceGRU
    notebooks     Ejecuta todos los notebooks en orden (produce outputs + figuras)
    score         Scoring en vivo con ambos modelos
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
NOTEBOOKS_DIR = ROOT / "notebooks"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run")

# Orden de ejecución de notebooks (sin playground ni exploración pura)
NOTEBOOK_ORDER = [
    "03_1_ts_dataset_validation.ipynb",
    "03_processed_dataset_eda.ipynb",
    "04_model_training.ipynb",
    "04_1_ts_model_training.ipynb",
    "05_live_scoring.ipynb",
    "05_1_ts_live_scoring.ipynb",
    "05_2_model_comparison.ipynb",
]

ALL_STEPS = ["fetch", "features", "features_ts", "train", "train_ts", "notebooks", "score"]

ONLY_BASELINE = ["features", "train"]
ONLY_GRU = ["features_ts", "train_ts"]


def _run(cmd: list[str], cwd: Path = ROOT) -> None:
    log.info("$ %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        log.error("Comando falló con código %d", result.returncode)
        sys.exit(result.returncode)


def _skip(reason: str) -> None:
    log.info("  SKIP — %s (usa --force para reejecutar)", reason)


def step_fetch(cfg: str, force: bool) -> None:
    sentinel = ROOT / "data" / "raw" / "resolved_markets.json"
    if sentinel.exists() and not force:
        _skip(f"{sentinel.relative_to(ROOT)} ya existe")
        return
    _run([sys.executable, "-m", "src.data.fetcher", "--mode", "full"])


def step_features(cfg: str, force: bool) -> None:
    sentinel = ROOT / "data" / "processed" / "numerical_features.npy"
    if sentinel.exists() and not force:
        _skip(f"{sentinel.relative_to(ROOT)} ya existe")
        return
    _run([sys.executable, "-m", "src.features.pipeline", "--config", cfg])


def step_features_ts(cfg: str, force: bool) -> None:
    sentinel = ROOT / "data" / "processed_ts" / "sequences.npy"
    if sentinel.exists() and not force:
        _skip(f"{sentinel.relative_to(ROOT)} ya existe")
        return
    _run([sys.executable, "-m", "src.features.ts_pipeline", "--config", cfg])


def step_train(cfg: str, force: bool) -> None:
    sentinel = ROOT / "data" / "models" / "market_value_baseline" / "best_market_model.pt"
    if sentinel.exists() and not force:
        _skip(f"{sentinel.relative_to(ROOT)} ya existe")
        return
    _run([sys.executable, "-m", "src.model.train", "--config", cfg])


def step_train_ts(cfg: str, force: bool) -> None:
    sentinel = ROOT / "data" / "models" / "price_sequence_gru" / "best_ts_gru_model.pt"
    if sentinel.exists() and not force:
        _skip(f"{sentinel.relative_to(ROOT)} ya existe")
        return
    _run([sys.executable, "-m", "src.model.ts_train", "--config", cfg])


def step_notebooks(cfg: str, force: bool) -> None:
    for nb_name in NOTEBOOK_ORDER:
        nb_path = NOTEBOOKS_DIR / nb_name
        if not nb_path.exists():
            log.warning("  Notebook no encontrado, saltando: %s", nb_name)
            continue
        log.info("Ejecutando notebook: %s", nb_name)
        _run([
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=600",
            str(nb_path),
        ])


def step_score(cfg: str, force: bool) -> None:
    _run([sys.executable, "-m", "src.scoring.scorer", "--config", cfg])
    _run([sys.executable, "-m", "src.scoring.ts_scorer", "--config", cfg])


STEP_FN = {
    "fetch":       step_fetch,
    "features":    step_features,
    "features_ts": step_features_ts,
    "train":       step_train,
    "train_ts":    step_train_ts,
    "notebooks":   step_notebooks,
    "score":       step_score,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        default=str(ROOT / "config" / "config.yaml"),
        help="Ruta al archivo de configuración (default: config/config.yaml)",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--steps",
        metavar="STEP[,STEP...]",
        help=f"Pasos a ejecutar separados por coma. Disponibles: {', '.join(ALL_STEPS)}",
    )
    group.add_argument(
        "--skip",
        metavar="STEP[,STEP...]",
        help="Pasos a omitir (ejecuta todos los demás)",
    )
    group.add_argument(
        "--only",
        choices=["baseline", "gru"],
        help="Atajo: 'baseline' = features+train, 'gru' = features_ts+train_ts",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Reejecutar pasos aunque sus artefactos ya existan",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.only == "baseline":
        steps = ONLY_BASELINE
    elif args.only == "gru":
        steps = ONLY_GRU
    elif args.steps:
        steps = [s.strip() for s in args.steps.split(",")]
        invalid = [s for s in steps if s not in STEP_FN]
        if invalid:
            log.error("Pasos inválidos: %s. Disponibles: %s", invalid, ALL_STEPS)
            sys.exit(1)
    elif args.skip:
        skip_set = {s.strip() for s in args.skip.split(",")}
        steps = [s for s in ALL_STEPS if s not in skip_set]
    else:
        steps = ALL_STEPS

    log.info("Pipeline: %s", " → ".join(steps))
    log.info("Config:   %s", args.config)
    log.info("Force:    %s", args.force)

    for step in steps:
        log.info("=" * 50)
        log.info("PASO: %s", step.upper())
        log.info("=" * 50)
        STEP_FN[step](args.config, args.force)

    log.info("=" * 50)
    log.info("Pipeline completado: %d pasos", len(steps))


if __name__ == "__main__":
    main()
