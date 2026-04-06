"""Evaluación del modelo y backtesting."""

import json
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from .architecture import MarketValueNet
from ..data.preprocessing import (
    infer_resolution_from_market,
    build_snapshot_market,
)
from ..features.pipeline import FeaturePipeline

logger = logging.getLogger(__name__)


def evaluate_model(
    model: MarketValueNet,
    dataloader: DataLoader,
    device: str | None = None,
) -> dict:
    """
    Evalúa el modelo en un DataLoader.

    Returns:
        Diccionario con métricas de evaluación.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for batch in dataloader:
            num = batch["numerical"].to(device)
            cat = batch["category"].to(device)
            txt = batch["text_emb"].to(device)
            lbl = batch["label"]

            scores = model(num, cat, txt).cpu()
            all_scores.extend(scores.numpy())
            all_labels.extend(lbl.numpy())

            if model.task == "classification":
                preds = (scores > 0.5).float()
                all_preds.extend(preds.numpy())

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    results = {"scores": all_scores, "labels": all_labels}

    if model.task == "classification":
        all_preds = np.array(all_preds)
        results.update({
            "predictions": all_preds,
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "recall": recall_score(all_labels, all_preds, zero_division=0),
            "f1": f1_score(all_labels, all_preds, zero_division=0),
            "roc_auc": roc_auc_score(all_labels, all_scores)
            if len(np.unique(all_labels)) > 1
            else 0.0,
            "confusion_matrix": confusion_matrix(all_labels, all_preds),
            "classification_report": classification_report(
                all_labels, all_preds, target_names=["No Buy", "Buy"]
            ),
        })
    else:
        mse = float(np.mean((all_scores - all_labels) ** 2))
        mae = float(np.mean(np.abs(all_scores - all_labels)))
        results.update({"mse": mse, "mae": mae})

    return results


def print_evaluation(results: dict) -> None:
    """Imprime resultados de evaluación."""
    if "accuracy" in results:
        print("=== Resultados de Clasificación ===")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1 Score:  {results['f1']:.4f}")
        print(f"  ROC AUC:   {results['roc_auc']:.4f}")
        print(f"\n{results['classification_report']}")
    else:
        print("=== Resultados de Regresión ===")
        print(f"  MSE: {results['mse']:.6f}")
        print(f"  MAE: {results['mae']:.6f}")


def backtest(
    model: MarketValueNet,
    historical_markets: list[dict],
    feature_pipeline: FeaturePipeline,
    price_histories: dict | None = None,
    initial_capital: float = 1000.0,
    position_size: float = 0.05,
    threshold: float = 0.6,
    snapshot_offset_days: int = 7,
    use_adaptive_cutoff: bool = True,
    device: str | None = None,
) -> tuple[pd.DataFrame, float]:
    """
    Simula trading con el modelo sobre datos históricos.

    Args:
        model: Modelo entrenado.
        historical_markets: Lista de mercados resueltos.
        feature_pipeline: Pipeline de features.
        price_histories: Historiales keyed por market_id para reconstruir
            snapshots anti-leakage.
        initial_capital: Capital inicial.
        position_size: Fracción del capital por trade.
        threshold: Umbral mínimo del modelo para comprar.
        snapshot_offset_days: Días antes del endDate para el snapshot.
        device: Dispositivo de cómputo.

    Returns:
        (DataFrame de trades, capital final)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    capital = initial_capital
    trades = []

    if price_histories is None:
        logger.warning(
            "Backtest sin price_histories: se omitirán mercados sin snapshot válido."
        )

    for market in historical_markets:
        market_id = market.get("id", "?")
        try:
            snapshot_market, snapshot_price = build_snapshot_market(
                market,
                price_histories=price_histories,
                snapshot_offset_days=snapshot_offset_days,
                use_adaptive_cutoff=use_adaptive_cutoff,
            )
            if snapshot_market is None or snapshot_price is None:
                continue

            history = None
            if price_histories is not None:
                history = price_histories.get(str(market_id))

            features = feature_pipeline.transform_single(
                snapshot_market,
                price_history=history,
            )
            num_tensor = (
                torch.FloatTensor(features["numerical"]).unsqueeze(0).to(device)
            )
            cat_tensor = torch.LongTensor([features["category_id"]]).to(device)
            txt_tensor = (
                torch.FloatTensor(features["text_embedding"]).unsqueeze(0).to(device)
            )

            with torch.no_grad():
                score = model(num_tensor, cat_tensor, txt_tensor).item()

            if score > threshold:
                price = float(snapshot_price)
                if price <= 0 or price >= 1:
                    continue

                bet_amount = capital * position_size
                shares = bet_amount / price
                # market.get("resolution") es siempre None en la API.
                # Inferir de outcomePrices: [1,0] = Yes, [0,1] = No.
                resolution = infer_resolution_from_market(market)
                payout = shares * (1.0 if resolution == "yes" else 0.0)
                pnl = payout - bet_amount
                capital += pnl

                trades.append({
                    "market_id": market_id,
                    "question": market.get("question", "")[:80],
                    "price_yes": price,
                    "snapshot_offset_days": snapshot_offset_days,
                    "score": score,
                    "resolution": resolution,
                    "bet_amount": bet_amount,
                    "pnl": pnl,
                    "capital_after": capital,
                })
        except Exception as e:
            logger.warning(
                "Backtest: market %s omitido: %s",
                market_id, e,
            )
            continue

    if not trades:
        logger.warning("Backtest: no se ejecutó ningún trade.")

    trades_df = pd.DataFrame(trades)
    return trades_df, capital
