"""Primary scorer for snapshot-based p_yes models."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import load_config
from .common import (
    get_models_registry_dir,
    load_active_context,
    load_catboost_bundle,
    load_gbdt_bundle,
    load_lstm_bundle,
    load_primary_model_info,
    load_tabular_bundle,
    load_ts_bundle,
    resolve_days_to_end,
    resolve_market_price,
    score_model_output_to_probs,
)
from .signals import generate_signals

logger = logging.getLogger(__name__)


SIGNAL_CONFIG_KEYS = {
    "buy_ev_threshold",
    "strong_buy_ev_threshold",
    "buy_roi_threshold",
    "strong_buy_roi_threshold",
    "expected_roi_cap",
    "min_price_yes",
    "max_price_yes",
    "max_days_to_end",
    "min_liquidity",
    "min_volume_24h",
    "max_spread",
}


def score_active_markets(
    bundle: dict,
    active_markets: list[dict],
    price_histories: dict,
    snapshot_time: pd.Timestamp | None = None,
    device: str | None = None,
    top_k: int | None = None,
    signal_config: dict | None = None,
) -> pd.DataFrame:
    """Score active markets with either the tabular or sequence model bundle."""
    if snapshot_time is None:
        snapshot_time = pd.Timestamp.now(tz="UTC")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = bundle["model_name"]
    model = bundle["model"]
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    pipeline = bundle["pipeline"]
    calibrator = bundle["calibrator"]
    prediction_mode = bundle.get("prediction_mode", "classification")
    pipeline.warm_text_cache(active_markets)

    ts_cfg = bundle["run_config"].get("dataset_metadata", {}) if model_name in {"price_sequence_gru", "price_sequence_lstm"} else {}
    lookback_days = ts_cfg.get("sequence_lookback_days", 30)
    step_hours = ts_cfg.get("sequence_grid_hours", 12)

    rows = []
    for market in active_markets:
        market_id = str(market.get("id", ""))
        history = price_histories.get(market_id)
        days_to_end = resolve_days_to_end(market, snapshot_time)
        if days_to_end <= 0.0:
            continue
        price_yes = resolve_market_price(market, history, snapshot_time)

        try:
            features = pipeline.transform_single(
                market,
                price_history=history,
                snapshot_time=snapshot_time,
                days_to_end=days_to_end,
            )

            if model_name == "market_value_baseline":
                with torch.no_grad():
                    model_output = float(
                        model(
                            torch.FloatTensor(features["numerical"]).unsqueeze(0).to(device),
                            torch.LongTensor([features["category_id"]]).to(device),
                            torch.FloatTensor(features["text_embedding"]).unsqueeze(0).to(device),
                        ).item()
                    )
                raw_prob, calibrated_prob = score_model_output_to_probs(
                    model_output,
                    price_yes=price_yes,
                    prediction_mode=prediction_mode,
                    calibrator=calibrator,
                )
            elif model_name == "hist_gradient_boosting":
                category_ohe = bundle["category_ohe"]
                text_pca = bundle["text_pca"]
                feature_vector = np.concatenate([
                    features["numerical"],
                    category_ohe.transform(np.asarray([[features["category_id"]]], dtype=np.int64)).reshape(-1),
                    text_pca.transform(features["text_embedding"].reshape(1, -1)).reshape(-1),
                ]).astype(np.float32)
                raw_prob = float(model.predict_proba(feature_vector.reshape(1, -1))[0, 1])
                calibrated_prob = float(calibrator.predict_proba([raw_prob])[0]) if calibrator is not None else raw_prob
            elif model_name == "catboost_residual":
                run_model_cfg = bundle["run_config"]["model"]
                num_dim = int(run_model_cfg["num_numerical_features"])
                text_dim = int(run_model_cfg["text_embed_dim"])
                feature_vector = np.empty((1, num_dim + 1 + text_dim), dtype=object)
                feature_vector[0, :num_dim] = features["numerical"].astype(np.float32)
                feature_vector[0, num_dim] = str(features["category_id"])
                feature_vector[0, num_dim + 1 :] = features["text_embedding"].astype(np.float32)
                model_output = float(model.predict(feature_vector)[0])
                raw_prob, calibrated_prob = score_model_output_to_probs(
                    model_output,
                    price_yes=price_yes,
                    prediction_mode=prediction_mode,
                    calibrator=calibrator,
                )
            else:
                from ..features.ts_sequence import build_live_price_sequence

                sequence, length, _ = build_live_price_sequence(
                    history,
                    snapshot_time=snapshot_time,
                    lookback_days=lookback_days,
                    step_hours=step_hours,
                )
                if sequence is None or length is None:
                    continue
                with torch.no_grad():
                    model_output = float(
                        model(
                            torch.FloatTensor(sequence).unsqueeze(0).to(device),
                            torch.LongTensor([length]).to(device),
                            torch.FloatTensor(features["numerical"]).unsqueeze(0).to(device),
                            torch.LongTensor([features["category_id"]]).to(device),
                            torch.FloatTensor(features["text_embedding"]).unsqueeze(0).to(device),
                        ).item()
                    )
                raw_prob, calibrated_prob = score_model_output_to_probs(
                    model_output,
                    price_yes=price_yes,
                    prediction_mode=prediction_mode,
                    calibrator=calibrator,
                )
            ev_per_share = float(calibrated_prob - price_yes)
            expected_roi = float(ev_per_share / max(price_yes, 1e-6))
            rows.append({
                "model_name": model_name,
                "id": market_id,
                "question": market.get("question", ""),
                "slug": market.get("slug", ""),
                "price_yes": float(price_yes),
                "p_yes_raw": float(raw_prob),
                "p_yes_calibrated": float(calibrated_prob),
                "ev_per_share": ev_per_share,
                "expected_roi": expected_roi,
                "days_to_end": float(days_to_end),
                "liquidity": float(market.get("liquidity", 0) or 0),
                "volume_24h": float(market.get("volume24hr", 0) or 0),
                "spread": float(market.get("spread", 0) or 0),
                "end_date": market.get("endDate", ""),
            })
        except Exception as exc:
            logger.debug("Score omitió market=%s model=%s: %s", market_id, model_name, exc)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    signal_kwargs = {
        key: value
        for key, value in (signal_config or {}).items()
        if key in SIGNAL_CONFIG_KEYS
    }
    df = generate_signals(df, **signal_kwargs)
    df = df.sort_values(["ev_per_share", "p_yes_calibrated"], ascending=False).reset_index(drop=True)
    if top_k is not None:
        df = df.head(top_k).reset_index(drop=True)
    return df


def summarize_scored_markets(scored_df: pd.DataFrame) -> dict:
    """Return a compact distribution summary for active-market scoring."""
    if scored_df.empty:
        return {"count": 0}

    summary = {
        "count": int(len(scored_df)),
        "signals": scored_df["signal"].value_counts().to_dict(),
    }
    for column in ["p_yes_calibrated", "ev_per_share", "expected_roi", "expected_roi_capped"]:
        summary[column] = {
            "p10": float(scored_df[column].quantile(0.10)),
            "p50": float(scored_df[column].quantile(0.50)),
            "p90": float(scored_df[column].quantile(0.90)),
            "min": float(scored_df[column].min()),
            "max": float(scored_df[column].max()),
        }
    return summary


def preview_scored_markets(scored_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Prefer actionable rows in CLI previews so the output matches the signal policy."""
    if scored_df.empty:
        return scored_df

    actionable = scored_df[scored_df["signal"] != "HOLD"].copy()
    if not actionable.empty:
        return actionable.head(top_k).reset_index(drop=True)
    return scored_df.head(top_k).reset_index(drop=True)


def load_bundle_by_name(name: str, models_root: str | Path, pipeline_dir: str | Path) -> dict:
    models_path = Path(models_root)
    if name == "market_value_baseline":
        return load_tabular_bundle(models_path / "market_value_baseline", pipeline_dir)
    if name == "price_sequence_gru":
        return load_ts_bundle(models_path / "price_sequence_gru", pipeline_dir)
    if name == "price_sequence_lstm":
        return load_lstm_bundle(models_path / "price_sequence_lstm", pipeline_dir)
    if name == "hist_gradient_boosting":
        return load_gbdt_bundle(models_path / "hist_gradient_boosting", pipeline_dir)
    if name == "catboost_residual":
        return load_catboost_bundle(models_path / "catboost_residual", pipeline_dir)
    raise ValueError(f"Modelo no soportado: {name}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Score active markets with the selected primary model")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--models-root", default=None)
    parser.add_argument("--pipeline-dir", default=None)
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--all-models", action="store_true")
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

    model_names: list[str]
    if args.all_models:
        candidate_model_names = [
            "market_value_baseline",
            "price_sequence_gru",
            "price_sequence_lstm",
            "hist_gradient_boosting",
            "catboost_residual",
        ]
        model_names = [
            name
            for name in candidate_model_names
            if (Path(models_root) / name / "run_config.json").exists()
        ]
    else:
        primary = load_primary_model_info(models_root)
        model_names = [primary["name"]] if primary else ["hist_gradient_boosting"]

    dfs = []
    summaries = {}
    for model_name in model_names:
        bundle = load_bundle_by_name(model_name, models_root, pipeline_dir)
        df = score_active_markets(
            bundle,
            active_markets=active_markets,
            price_histories=price_histories,
            top_k=top_k if not args.all_models else None,
            signal_config=scoring_cfg,
        )
        summaries[model_name] = summarize_scored_markets(df)
        dfs.append(df)
        logger.info(
            "Scored %s | rows=%d signals=%s",
            model_name,
            len(df),
            summaries[model_name].get("signals", {}),
        )

    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    output_dir = get_models_registry_dir(Path(args.output or models_root))
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / "live_scores.csv", index=False)
    with open(output_dir / "live_scoring_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    if not combined.empty:
        preview_frames = [
            preview_scored_markets(df, top_k if top_k is not None else len(df))
            for df in dfs
            if not df.empty
        ]
        preview = pd.concat(preview_frames, ignore_index=True) if preview_frames else combined.head(top_k)
        print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
