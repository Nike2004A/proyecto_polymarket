"""Gradient-boosted tabular model over snapshot features."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder

from ..config import load_config
from .calibration import fit_isotonic_calibrator, fit_platt_calibrator, identity_calibrator
from .dataset import PolymarketDataset
from .metrics import compute_bucketed_metrics, compute_ev_metrics, compute_probability_metrics
from .splits import build_dataset_split

logger = logging.getLogger(__name__)


DEFAULT_GBDT_CANDIDATES = [
    {
        "learning_rate": 0.03,
        "max_depth": 4,
        "max_iter": 400,
        "min_samples_leaf": 30,
        "l2_regularization": 0.0,
    },
    {
        "learning_rate": 0.03,
        "max_depth": 6,
        "max_iter": 500,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
    },
    {
        "learning_rate": 0.05,
        "max_depth": 6,
        "max_iter": 300,
        "min_samples_leaf": 40,
        "l2_regularization": 0.1,
    },
    {
        "learning_rate": 0.05,
        "max_depth": 8,
        "max_iter": 400,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
    },
    {
        "learning_rate": 0.07,
        "max_depth": 6,
        "max_iter": 250,
        "min_samples_leaf": 50,
        "l2_regularization": 0.2,
    },
    {
        "learning_rate": 0.1,
        "max_depth": 4,
        "max_iter": 250,
        "min_samples_leaf": 40,
        "l2_regularization": 0.1,
    },
]


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _build_matrix(
    numerical: np.ndarray,
    categories: np.ndarray,
    text_embeddings: np.ndarray,
    category_encoder: OneHotEncoder,
    text_pca: PCA,
) -> np.ndarray:
    category_features = category_encoder.transform(categories.reshape(-1, 1))
    text_features = text_pca.transform(text_embeddings)
    return np.concatenate([numerical, category_features, text_features], axis=1).astype(np.float32)


def _select_best_calibrator(
    raw_probs: np.ndarray,
    labels: np.ndarray,
    calibration_tolerance: float,
) -> tuple[str, dict, dict]:
    candidate_calibrators = {
        "identity": identity_calibrator(input_kind="probability"),
        "platt": fit_platt_calibrator(raw_probs, labels, input_kind="probability"),
        "isotonic": fit_isotonic_calibrator(raw_probs, labels, input_kind="probability"),
    }
    calibration_scores = {
        name: compute_probability_metrics(labels, calibrator.predict_proba(raw_probs))
        for name, calibrator in candidate_calibrators.items()
    }
    best_brier = min(metrics["brier"] for metrics in calibration_scores.values())
    preference = ["identity", "platt", "isotonic"]
    eligible = [
        name
        for name in preference
        if calibration_scores[name]["brier"] <= best_brier + calibration_tolerance
    ]
    best_name = eligible[0] if eligible else min(calibration_scores, key=lambda name: calibration_scores[name]["brier"])
    return best_name, candidate_calibrators[best_name], calibration_scores


def _ranking_key(results: dict) -> tuple[float, float, float]:
    return (
        float(results["ev_metrics"]["top_k_avg_realized_pnl"]),
        -float(results["calibrated_metrics"]["brier"]),
        -float(results["calibrated_metrics"]["log_loss"]),
    )


def train_gbdt_pipeline(
    cfg: dict,
    data_dir: str | None = None,
    save_dir: str | None = None,
) -> dict:
    training_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    feature_cfg = cfg.get("features", {})
    scoring_cfg = cfg.get("scoring", {})

    data_dir = data_dir or data_cfg.get("processed_dir", "data/processed")
    save_dir = save_dir or "data/models/hist_gradient_boosting"
    seed = int(training_cfg.get("seed", 42))
    val_split = float(training_cfg.get("val_split", 0.15))
    test_split = float(training_cfg.get("test_split", 0.15))
    split_strategy = training_cfg.get("split_strategy", "temporal_grouped")
    top_k = int(scoring_cfg.get("top_k", 20))
    roi_cap = scoring_cfg.get("expected_roi_cap")
    calibration_tolerance = float(feature_cfg.get("calibration_tolerance", 0.005))
    gbdt_cfg = cfg.get("gbdt_model", {})
    candidate_configs = gbdt_cfg.get("candidate_configs", DEFAULT_GBDT_CANDIDATES)

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
    labels = dataset.labels.numpy()
    market_prices = np.asarray(dataset.snapshot_prices, dtype=np.float32)
    days_to_end = np.asarray(dataset.days_to_end, dtype=np.float32)

    category_encoder = _make_one_hot_encoder()
    category_encoder.fit(categories[train_idx].reshape(-1, 1))

    text_dim = text_embeddings.shape[1]
    pca_dim = min(16, text_dim, max(2, len(train_idx) - 1))
    text_pca = PCA(n_components=pca_dim, random_state=seed)
    text_pca.fit(text_embeddings[train_idx])

    X_train = _build_matrix(numerical[train_idx], categories[train_idx], text_embeddings[train_idx], category_encoder, text_pca)
    X_val = _build_matrix(numerical[val_idx], categories[val_idx], text_embeddings[val_idx], category_encoder, text_pca)
    X_test = _build_matrix(numerical[test_idx], categories[test_idx], text_embeddings[test_idx], category_encoder, text_pca)

    search_results: list[dict] = []
    best_artifact: dict | None = None
    for model_params in candidate_configs:
        model = HistGradientBoostingClassifier(random_state=seed, **model_params)
        model.fit(X_train, labels[train_idx].astype(np.int64))

        val_raw_probs = model.predict_proba(X_val)[:, 1]
        calibration_name, calibrator, calibration_scores = _select_best_calibrator(
            val_raw_probs,
            labels[val_idx],
            calibration_tolerance=calibration_tolerance,
        )
        val_probs = calibrator.predict_proba(val_raw_probs)
        val_results = {
            "raw_metrics": compute_probability_metrics(labels[val_idx], val_raw_probs),
            "calibrated_metrics": compute_probability_metrics(labels[val_idx], val_probs),
            "market_baseline_metrics": compute_probability_metrics(labels[val_idx], market_prices[val_idx]),
            "residual_metrics": {},
            "ev_metrics": compute_ev_metrics(labels[val_idx], market_prices[val_idx], val_probs, top_k=top_k, roi_cap=roi_cap),
            "by_horizon": compute_bucketed_metrics(labels[val_idx], market_prices[val_idx], val_probs, days_to_end[val_idx], top_k=top_k, roi_cap=roi_cap),
        }
        artifact = {
            "model": model,
            "model_params": model_params,
            "calibration_name": calibration_name,
            "calibrator": calibrator,
            "calibration_scores": calibration_scores,
            "val_raw_probs": val_raw_probs,
            "val_probs": val_probs,
            "val_results": val_results,
        }
        search_results.append({
            "model_params": model_params,
            "selected_calibration": calibration_name,
            "validation": {
                "brier": float(val_results["calibrated_metrics"]["brier"]),
                "log_loss": float(val_results["calibrated_metrics"]["log_loss"]),
                "top_k_avg_realized_pnl": float(val_results["ev_metrics"]["top_k_avg_realized_pnl"]),
                "top_k_hit_rate": float(val_results["ev_metrics"]["top_k_hit_rate"]),
            },
        })
        if best_artifact is None or _ranking_key(val_results) > _ranking_key(best_artifact["val_results"]):
            best_artifact = artifact

    if best_artifact is None:
        raise ValueError("No se pudo entrenar ningún candidato GBDT.")

    model = best_artifact["model"]
    best_model_params = best_artifact["model_params"]
    best_calibration_name = best_artifact["calibration_name"]
    best_calibrator = best_artifact["calibrator"]
    calibration_scores = best_artifact["calibration_scores"]
    val_raw_probs = best_artifact["val_raw_probs"]
    val_probs = best_artifact["val_probs"]

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path / "gbdt_model.pkl")
    joblib.dump(category_encoder, save_path / "category_encoder.pkl")
    joblib.dump(text_pca, save_path / "text_pca.pkl")
    best_calibrator.save(save_path / "calibration")
    with open(save_path / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({"candidates": search_results}, f, indent=2)

    test_raw_probs = model.predict_proba(X_test)[:, 1]
    test_probs = best_calibrator.predict_proba(test_raw_probs)

    val_results = best_artifact["val_results"]
    test_results = {
        "raw_metrics": compute_probability_metrics(labels[test_idx], test_raw_probs),
        "calibrated_metrics": compute_probability_metrics(labels[test_idx], test_probs),
        "market_baseline_metrics": compute_probability_metrics(labels[test_idx], market_prices[test_idx]),
        "residual_metrics": {},
        "ev_metrics": compute_ev_metrics(labels[test_idx], market_prices[test_idx], test_probs, top_k=top_k, roi_cap=roi_cap),
        "by_horizon": compute_bucketed_metrics(labels[test_idx], market_prices[test_idx], test_probs, days_to_end[test_idx], top_k=top_k, roi_cap=roi_cap),
    }

    run_config = {
        "data_dir": data_dir,
        "save_dir": save_dir,
        "prediction_mode": "classification_probability",
        "split": {
            **split.to_metadata(),
            "class_balance": {
                "train_positive_rate": float(labels[train_idx].mean()),
                "val_positive_rate": float(labels[val_idx].mean()),
                "test_positive_rate": float(labels[test_idx].mean()),
            },
        },
        "model": {
            "type": "HistGradientBoostingClassifier",
            **best_model_params,
            "category_feature_dim": int(category_encoder.transform(categories[:1].reshape(-1, 1)).shape[1]),
            "text_pca_dim": int(pca_dim),
            "num_numerical_features": int(numerical.shape[1]),
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
        "GBDT calibration selected=%s | test_brier=%.4f market_brier=%.4f topk_realized_pnl=%.4f",
        best_calibration_name,
        test_results["calibrated_metrics"]["brier"],
        test_results["market_baseline_metrics"]["brier"],
        test_results["ev_metrics"]["top_k_avg_realized_pnl"],
    )

    return {
        "model_name": "hist_gradient_boosting",
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
    train_gbdt_pipeline(cfg)


if __name__ == "__main__":
    main()
