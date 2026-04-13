"""Shared helpers for loading trained bundles and active-market context."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from ..data.snapshots import PreparedHistory, latest_price_before, prepare_history_map
from ..features.pipeline import FeaturePipeline
from ..model.architecture import MarketValueNet
from ..model.calibration import ProbabilityCalibrator
from ..model.metrics import sigmoid
from ..model.ts_architecture import PriceSequenceGRU


def get_models_registry_dir(models_root: str | Path) -> Path:
    return Path(models_root) / "registry"


def load_active_context(raw_dir: str | Path) -> tuple[list[dict], dict[str, PreparedHistory]]:
    raw_path = Path(raw_dir)
    with open(raw_path / "active_markets.json", encoding="utf-8") as f:
        active_markets = json.load(f)
    with open(raw_path / "price_histories.json", encoding="utf-8") as f:
        histories = json.load(f)
    return active_markets, prepare_history_map(histories)


def resolve_market_price(market: dict, history: PreparedHistory | None, snapshot_time: pd.Timestamp) -> float:
    if history is not None:
        price = latest_price_before(history, snapshot_time)
        if price is not None:
            return float(price)

    outcome_prices = market.get("outcomePrices", [])
    if isinstance(outcome_prices, str):
        try:
            outcome_prices = json.loads(outcome_prices)
        except (ValueError, TypeError):
            outcome_prices = []
    if isinstance(outcome_prices, list) and outcome_prices:
        try:
            return float(outcome_prices[0])
        except (ValueError, TypeError):
            pass

    try:
        return float(market.get("lastTradePrice") or 0.5)
    except (ValueError, TypeError):
        return 0.5


def resolve_days_to_end(market: dict, snapshot_time: pd.Timestamp) -> float:
    end_date = market.get("endDate")
    try:
        end_time = pd.to_datetime(end_date, utc=True)
    except (ValueError, TypeError):
        return 0.0
    return max(0.0, (end_time - snapshot_time).total_seconds() / 86400.0)


def load_primary_model_info(models_root: str | Path) -> dict | None:
    root = Path(models_root)
    path = get_models_registry_dir(root) / "primary_model.json"
    if not path.exists():
        path = root / "primary_model.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_tabular_bundle(model_dir: str | Path, pipeline_dir: str | Path) -> dict:
    model_path = Path(model_dir)
    with open(model_path / "run_config.json", encoding="utf-8") as f:
        run_config = json.load(f)

    pipeline = FeaturePipeline.load(str(pipeline_dir))
    model = MarketValueNet(
        num_numerical_features=run_config["model"]["num_numerical_features"],
        num_categories=run_config["model"]["num_categories"],
        category_embed_dim=run_config["model"]["category_embed_dim"],
        text_embed_dim=pipeline.text_embed_dim,
        hidden_dims=run_config["model"]["hidden_dims"],
        dropout=run_config["model"]["dropout"],
        task="classification",
    )
    model.load_state_dict(torch.load(model_path / "best_market_model.pt", map_location="cpu", weights_only=True))
    calibration_dir = model_path / "calibration"
    calibrator = ProbabilityCalibrator.load(calibration_dir) if calibration_dir.exists() else None
    return {
        "model_name": "market_value_baseline",
        "model": model,
        "pipeline": pipeline,
        "calibrator": calibrator,
        "prediction_mode": run_config.get("prediction_mode", "classification"),
        "run_config": run_config,
    }


def load_ts_bundle(model_dir: str | Path, pipeline_dir: str | Path) -> dict:
    model_path = Path(model_dir)
    with open(model_path / "run_config.json", encoding="utf-8") as f:
        run_config = json.load(f)

    pipeline = FeaturePipeline.load(str(pipeline_dir))
    model = PriceSequenceGRU(
        input_dim=run_config["model"]["input_dim"],
        static_num_features=run_config["model"]["static_num_features"],
        num_categories=run_config["model"].get("num_categories", 10),
        category_embed_dim=run_config["model"].get("category_embed_dim", 8),
        text_embed_dim=pipeline.text_embed_dim,
        hidden_dim=run_config["model"]["hidden_dim"],
        static_hidden_dim=run_config["model"]["static_hidden_dim"],
        num_layers=run_config["model"]["num_layers"],
        dropout=run_config["model"]["dropout"],
        task="classification",
    )
    model.load_state_dict(torch.load(model_path / "best_ts_gru_model.pt", map_location="cpu", weights_only=True))
    calibration_dir = model_path / "calibration"
    calibrator = ProbabilityCalibrator.load(calibration_dir) if calibration_dir.exists() else None
    return {
        "model_name": "price_sequence_gru",
        "model": model,
        "pipeline": pipeline,
        "calibrator": calibrator,
        "prediction_mode": run_config.get("prediction_mode", "classification"),
        "run_config": run_config,
    }


def load_gbdt_bundle(model_dir: str | Path, pipeline_dir: str | Path) -> dict:
    model_path = Path(model_dir)
    with open(model_path / "run_config.json", encoding="utf-8") as f:
        run_config = json.load(f)

    pipeline = FeaturePipeline.load(str(pipeline_dir))
    calibration_dir = model_path / "calibration"
    calibrator = ProbabilityCalibrator.load(calibration_dir) if calibration_dir.exists() else None
    return {
        "model_name": "hist_gradient_boosting",
        "model": joblib.load(model_path / "gbdt_model.pkl"),
        "category_ohe": joblib.load(model_path / "category_encoder.pkl"),
        "text_pca": joblib.load(model_path / "text_pca.pkl"),
        "pipeline": pipeline,
        "calibrator": calibrator,
        "prediction_mode": run_config.get("prediction_mode", "classification_probability"),
        "run_config": run_config,
    }

def score_model_output_to_probs(
    model_output: float,
    price_yes: float,
    prediction_mode: str,
    calibrator: ProbabilityCalibrator | None = None,
) -> tuple[float, float]:
    if prediction_mode == "residual":
        raw_prob = float(np.clip(price_yes + model_output, 1e-6, 1 - 1e-6))
        calibrated_prob = float(calibrator.predict_proba([raw_prob])[0]) if calibrator is not None else raw_prob
        return raw_prob, calibrated_prob

    raw_prob = float(sigmoid([model_output])[0])
    calibrated_prob = float(calibrator.predict_proba([model_output])[0]) if calibrator is not None else raw_prob
    return raw_prob, calibrated_prob
