"""CatBoost residual model over snapshot features."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor, Pool

from ..config import load_config
from .calibration import fit_isotonic_calibrator, fit_platt_calibrator, identity_calibrator
from .dataset import PolymarketDataset
from .metrics import (
    clip_probabilities_from_residual,
    compute_bucketed_metrics,
    compute_ev_metrics,
    compute_probability_metrics,
    compute_residual_metrics,
)
from .splits import build_dataset_split

logger = logging.getLogger(__name__)


DEFAULT_CATBOOST_CANDIDATES = [
    {
        "iterations": 500,
        "depth": 4,
        "learning_rate": 0.03,
        "l2_leaf_reg": 8.0,
        "min_data_in_leaf": 40,
        "bagging_temperature": 0.5,
        "random_strength": 1.5,
    },
    {
        "iterations": 700,
        "depth": 5,
        "learning_rate": 0.04,
        "l2_leaf_reg": 10.0,
        "min_data_in_leaf": 60,
        "bagging_temperature": 1.0,
        "random_strength": 2.0,
    },
    {
        "iterations": 700,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 10.0,
        "min_data_in_leaf": 50,
        "bagging_temperature": 1.0,
        "random_strength": 1.5,
    },
    {
        "iterations": 900,
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 12.0,
        "min_data_in_leaf": 80,
        "bagging_temperature": 1.0,
        "random_strength": 2.0,
    },
    {
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.07,
        "l2_leaf_reg": 8.0,
        "min_data_in_leaf": 70,
        "bagging_temperature": 0.0,
        "random_strength": 1.0,
    },
    {
        "iterations": 800,
        "depth": 7,
        "learning_rate": 0.03,
        "l2_leaf_reg": 14.0,
        "min_data_in_leaf": 100,
        "bagging_temperature": 1.0,
        "random_strength": 3.0,
    },
]


def _build_feature_matrix(
    numerical: np.ndarray,
    categories: np.ndarray,
    text_embeddings: np.ndarray,
) -> np.ndarray:
    num_dim = numerical.shape[1]
    text_dim = text_embeddings.shape[1]
    matrix = np.empty((len(numerical), num_dim + 1 + text_dim), dtype=object)
    matrix[:, :num_dim] = numerical.astype(np.float32)
    matrix[:, num_dim] = categories.astype(str)
    matrix[:, num_dim + 1 :] = text_embeddings.astype(np.float32)
    return matrix


def _make_pool(
    numerical: np.ndarray,
    categories: np.ndarray,
    text_embeddings: np.ndarray,
    targets: np.ndarray,
) -> Pool:
    num_dim = numerical.shape[1]
    features = _build_feature_matrix(numerical, categories, text_embeddings)
    return Pool(features, label=targets, cat_features=[num_dim])


def _select_best_calibrator(
    raw_probs: np.ndarray,
    labels: np.ndarray,
    market_prices: np.ndarray,
    days_to_end: np.ndarray,
    top_k: int,
    roi_cap: float | None,
    calibration_tolerance: float,
) -> tuple[str, dict, dict, dict]:
    candidate_calibrators = {
        "identity": identity_calibrator(input_kind="probability"),
        "platt": fit_platt_calibrator(raw_probs, labels, input_kind="probability"),
        "isotonic": fit_isotonic_calibrator(raw_probs, labels, input_kind="probability"),
    }
    calibration_scores = {}
    calibration_results = {}
    for name, calibrator in candidate_calibrators.items():
        probs = calibrator.predict_proba(raw_probs)
        calibration_scores[name] = compute_probability_metrics(labels, probs)
        calibration_results[name] = {
            "raw_metrics": compute_probability_metrics(labels, raw_probs),
            "calibrated_metrics": calibration_scores[name],
            "market_baseline_metrics": compute_probability_metrics(labels, market_prices),
            "residual_metrics": {},
            "ev_metrics": compute_ev_metrics(labels, market_prices, probs, top_k=top_k, roi_cap=roi_cap),
            "by_horizon": compute_bucketed_metrics(labels, market_prices, probs, days_to_end, top_k=top_k, roi_cap=roi_cap),
        }
    best_brier = min(metrics["brier"] for metrics in calibration_scores.values())
    eligible = [
        name
        for name in candidate_calibrators
        if calibration_scores[name]["brier"] <= best_brier + calibration_tolerance
    ]
    pool = eligible if eligible else list(candidate_calibrators)
    best_name = max(pool, key=lambda name: _ranking_key(calibration_results[name]))
    return best_name, candidate_calibrators[best_name], calibration_scores, calibration_results[best_name]


def _ranking_key(results: dict) -> tuple[float, float, float]:
    return (
        float(results["ev_metrics"]["top_k_avg_realized_pnl"]),
        -float(results["calibrated_metrics"]["brier"]),
        -float(results["calibrated_metrics"]["log_loss"]),
    )


def train_catboost_pipeline(
    cfg: dict,
    data_dir: str | None = None,
    save_dir: str | None = None,
) -> dict:
    training_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    feature_cfg = cfg.get("features", {})
    scoring_cfg = cfg.get("scoring", {})

    data_dir = data_dir or data_cfg.get("processed_dir", "data/processed")
    save_dir = save_dir or "data/models/catboost_residual"
    seed = int(training_cfg.get("seed", 42))
    val_split = float(training_cfg.get("val_split", 0.15))
    test_split = float(training_cfg.get("test_split", 0.15))
    split_strategy = training_cfg.get("split_strategy", "temporal_grouped")
    top_k = int(scoring_cfg.get("top_k", 20))
    roi_cap = scoring_cfg.get("expected_roi_cap")
    calibration_tolerance = float(feature_cfg.get("calibration_tolerance", 0.005))
    thread_count = int(training_cfg.get("catboost_thread_count", 4))
    catboost_cfg = cfg.get("catboost_model", {})
    candidate_configs = catboost_cfg.get("candidate_configs", DEFAULT_CATBOOST_CANDIDATES)

    dataset = PolymarketDataset.from_numpy_dir(data_dir)
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

    train_idx = np.asarray(split.train_indices, dtype=np.int64)
    val_idx = np.asarray(split.val_indices, dtype=np.int64)
    test_idx = np.asarray(split.test_indices, dtype=np.int64)

    numerical = dataset.numerical.numpy()
    categories = dataset.categories.numpy()
    text_embeddings = dataset.text_emb.numpy()
    labels = dataset.labels.numpy().astype(np.int64)
    market_prices = np.asarray(dataset.snapshot_prices, dtype=np.float32)
    residual_targets = labels.astype(np.float32) - market_prices
    days_to_end = np.asarray(dataset.days_to_end, dtype=np.float32)

    train_pool = _make_pool(
        numerical[train_idx],
        categories[train_idx],
        text_embeddings[train_idx],
        residual_targets[train_idx],
    )
    val_pool = _make_pool(
        numerical[val_idx],
        categories[val_idx],
        text_embeddings[val_idx],
        residual_targets[val_idx],
    )
    test_pool = _make_pool(
        numerical[test_idx],
        categories[test_idx],
        text_embeddings[test_idx],
        residual_targets[test_idx],
    )

    search_results: list[dict] = []
    best_artifact: dict | None = None
    for model_params in candidate_configs:
        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=thread_count,
            **model_params,
        )
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        val_residual_preds = np.asarray(model.predict(val_pool), dtype=np.float32)
        val_raw_probs = clip_probabilities_from_residual(val_residual_preds, market_prices[val_idx])
        calibration_name, calibrator, calibration_scores, val_results = _select_best_calibrator(
            val_raw_probs,
            labels[val_idx],
            market_prices[val_idx],
            days_to_end[val_idx],
            top_k=top_k,
            roi_cap=roi_cap,
            calibration_tolerance=calibration_tolerance,
        )
        val_probs = calibrator.predict_proba(val_raw_probs)
        val_results["residual_metrics"] = compute_residual_metrics(labels[val_idx], market_prices[val_idx], val_residual_preds)
        artifact = {
            "model": model,
            "model_params": model_params,
            "calibration_name": calibration_name,
            "calibrator": calibrator,
            "calibration_scores": calibration_scores,
            "val_raw_probs": val_raw_probs,
            "val_probs": val_probs,
            "val_residual_preds": val_residual_preds,
            "val_results": val_results,
        }
        search_results.append(
            {
                "model_params": model_params,
                "best_iteration": int(model.get_best_iteration()),
                "selected_calibration": calibration_name,
                "validation": {
                    "brier": float(val_results["calibrated_metrics"]["brier"]),
                    "log_loss": float(val_results["calibrated_metrics"]["log_loss"]),
                    "top_k_avg_realized_pnl": float(val_results["ev_metrics"]["top_k_avg_realized_pnl"]),
                    "top_k_hit_rate": float(val_results["ev_metrics"]["top_k_hit_rate"]),
                },
            }
        )
        if best_artifact is None or _ranking_key(val_results) > _ranking_key(best_artifact["val_results"]):
            best_artifact = artifact

    if best_artifact is None:
        raise ValueError("No se pudo entrenar ningún candidato CatBoost.")

    model = best_artifact["model"]
    best_model_params = best_artifact["model_params"]
    best_calibration_name = best_artifact["calibration_name"]
    best_calibrator = best_artifact["calibrator"]
    calibration_scores = best_artifact["calibration_scores"]
    val_results = best_artifact["val_results"]

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_model(save_path / "catboost_model.cbm")
    best_calibrator.save(save_path / "calibration")
    with open(save_path / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({"candidates": search_results}, f, indent=2)

    test_residual_preds = np.asarray(model.predict(test_pool), dtype=np.float32)
    test_raw_probs = clip_probabilities_from_residual(test_residual_preds, market_prices[test_idx])
    test_probs = best_calibrator.predict_proba(test_raw_probs)
    test_results = {
        "raw_metrics": compute_probability_metrics(labels[test_idx], test_raw_probs),
        "calibrated_metrics": compute_probability_metrics(labels[test_idx], test_probs),
        "market_baseline_metrics": compute_probability_metrics(labels[test_idx], market_prices[test_idx]),
        "residual_metrics": compute_residual_metrics(labels[test_idx], market_prices[test_idx], test_residual_preds),
        "ev_metrics": compute_ev_metrics(labels[test_idx], market_prices[test_idx], test_probs, top_k=top_k, roi_cap=roi_cap),
        "by_horizon": compute_bucketed_metrics(labels[test_idx], market_prices[test_idx], test_probs, days_to_end[test_idx], top_k=top_k, roi_cap=roi_cap),
    }

    run_config = {
        "data_dir": data_dir,
        "save_dir": save_dir,
        "prediction_mode": "residual",
        "split": {
            "val_split": val_split,
            "test_split": test_split,
            "strategy": split_strategy,
            "seed": seed,
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx)),
            "test_size": int(len(test_idx)),
            **split.to_metadata(),
            "class_balance": {
                "train_positive_rate": float(labels[train_idx].mean()),
                "val_positive_rate": float(labels[val_idx].mean()),
                "test_positive_rate": float(labels[test_idx].mean()),
            },
        },
        "model": {
            "type": "CatBoostRegressor",
            **best_model_params,
            "num_numerical_features": int(numerical.shape[1]),
            "text_embed_dim": int(text_embeddings.shape[1]),
            "num_categories": int(len(np.unique(categories))),
            "thread_count": thread_count,
            "best_iteration": int(model.get_best_iteration()),
        },
        "calibration": {
            "selected": best_calibration_name,
            "tolerance": calibration_tolerance,
            "candidates": calibration_scores,
        },
        "search_results": search_results,
        "selection_metric": cfg.get("model_selection", {}).get("primary_model_selection_metric", "topk_ev"),
    }
    with open(save_path / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    metrics_payload = {
        "validation": val_results,
        "test": test_results,
        "market_baseline_beaten": {
            "brier": test_results["calibrated_metrics"]["brier"] < test_results["market_baseline_metrics"]["brier"],
            "log_loss": test_results["calibrated_metrics"]["log_loss"] < test_results["market_baseline_metrics"]["log_loss"],
            "top_k_realized_pnl": test_results["ev_metrics"]["top_k_avg_realized_pnl"] > 0.0,
        },
    }
    with open(save_path / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    logger.info(
        "CATBOOST residual calibration=%s | test_brier=%.4f market_brier=%.4f topk_realized_pnl=%.4f",
        best_calibration_name,
        test_results["calibrated_metrics"]["brier"],
        test_results["market_baseline_metrics"]["brier"],
        test_results["ev_metrics"]["top_k_avg_realized_pnl"],
    )

    return {
        "model_name": "catboost_residual",
        "save_dir": save_dir,
        "history": {},
        "validation": val_results,
        "test": test_results,
        "calibration_method": best_calibration_name,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = load_config("config/config.yaml")
    train_catboost_pipeline(cfg)


if __name__ == "__main__":
    main()
