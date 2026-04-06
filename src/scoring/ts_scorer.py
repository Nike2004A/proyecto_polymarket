"""Scoring para el modelo GRU puro de series de tiempo."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch

from ..config import load_config
from ..data.client import PolymarketDataClient
from ..data.preprocessing import infer_resolution_from_market
from ..features.ts_sequence import (
    build_live_price_sequence,
    build_market_price_sequence,
)
from ..model.ts_architecture import PriceSequenceGRU

logger = logging.getLogger(__name__)


def _load_cached_histories(raw_dir: str | Path | None) -> dict:
    if raw_dir is None:
        return {}
    path = Path(raw_dir) / "price_histories.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        loaded = json.load(f)
    return loaded if isinstance(loaded, dict) else {}


def _fetch_live_history(
    client: PolymarketDataClient,
    market: dict,
) -> list[dict] | None:
    token_ids = market.get("clobTokenIds", [])
    if not token_ids:
        return None

    try:
        return client.get_price_history(token_ids[0])
    except Exception as exc:
        logger.debug(
            "No se pudo obtener price_history live para market=%s: %s",
            market.get("id", "?"),
            exc,
        )
        return None


def _score_sequence(
    model: PriceSequenceGRU,
    sequence,
    sequence_length: int,
    device: str,
) -> float:
    seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
    len_tensor = torch.LongTensor([sequence_length]).to(device)
    with torch.no_grad():
        return float(torch.sigmoid(model(seq_tensor, len_tensor)).item())


def _resolve_market_price(market: dict, fallback_price: float) -> float:
    outcome_prices = market.get("outcomePrices", [])
    if isinstance(outcome_prices, str):
        try:
            outcome_prices = json.loads(outcome_prices)
        except (ValueError, TypeError):
            outcome_prices = []

    if isinstance(outcome_prices, list) and outcome_prices:
        return float(outcome_prices[0])
    return float(market.get("lastTradePrice") or fallback_price)


def score_active_markets_ts(
    model: PriceSequenceGRU,
    client: PolymarketDataClient,
    price_histories: dict | None = None,
    fetch_missing_history: bool = True,
    device: str | None = None,
    top_k: int = 20,
    max_markets: int = 1000,
    seq_len: int = 64,
    min_points: int = 5,
) -> pd.DataFrame:
    """Puntúa mercados activos usando solo su historia de precios."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)
    price_histories = price_histories or {}

    raw_markets = client.get_all_active_markets(max_markets=max_markets)
    parsed = [client.parse_market(market) for market in raw_markets]

    results = []
    skipped = {
        "insufficient_history": 0,
        "exceptions": 0,
    }
    fetched_histories = 0

    for market in parsed:
        market_id = str(market.get("id", ""))
        try:
            history = price_histories.get(market_id)
            if fetch_missing_history and not history:
                history = _fetch_live_history(client, market)
                if history:
                    fetched_histories += 1

            sequence, sequence_length, latest_price = build_live_price_sequence(
                history,
                seq_len=seq_len,
                min_points=min_points,
            )
            if sequence is None or sequence_length is None or latest_price is None:
                skipped["insufficient_history"] += 1
                continue

            score = _score_sequence(model, sequence, sequence_length, device=device)
            price_yes = _resolve_market_price(market, fallback_price=latest_price)

            results.append({
                "id": market_id,
                "question": market.get("question", ""),
                "price_yes": float(price_yes),
                "volume_24h": float(market.get("volume24hr", 0) or 0),
                "liquidity": float(market.get("liquidity", 0) or 0),
                "spread": float(market.get("spread", 0) or 0),
                "model_score": score,
                "score_minus_price": score - float(price_yes),
                "sequence_length": int(sequence_length),
                "slug": market.get("slug", ""),
                "end_date": market.get("endDate", ""),
            })
        except Exception as exc:
            skipped["exceptions"] += 1
            logger.debug("TS scorer omitió market=%s: %s", market_id, exc)

    if fetched_histories:
        logger.info("TS scorer completó %d histories live desde API.", fetched_histories)
    if any(skipped.values()):
        logger.warning("TS scorer omitió mercados: %s", skipped)

    df = pd.DataFrame(results)
    if df.empty:
        return df
    return df.sort_values("model_score", ascending=False).head(top_k).reset_index(drop=True)


def score_resolved_markets_ts(
    model: PriceSequenceGRU,
    historical_markets: list[dict],
    price_histories: dict,
    snapshot_offset_days: int = 7,
    seq_len: int = 64,
    min_points: int = 5,
    use_adaptive_cutoff: bool = True,
    device: str | None = None,
    max_markets: int | None = None,
) -> pd.DataFrame:
    """Scorea mercados resueltos usando secuencias truncadas al snapshot."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    source_markets = historical_markets[:max_markets] if max_markets else historical_markets
    rows = []
    skipped = {"no_sequence": 0, "ambiguous": 0, "exceptions": 0}

    for market in source_markets:
        market_id = str(market.get("id", ""))
        try:
            sequence, sequence_length, snapshot_price, _ = build_market_price_sequence(
                market,
                price_histories=price_histories,
                snapshot_offset_days=snapshot_offset_days,
                seq_len=seq_len,
                min_points=min_points,
                use_adaptive_cutoff=use_adaptive_cutoff,
            )
            if sequence is None or sequence_length is None or snapshot_price is None:
                skipped["no_sequence"] += 1
                continue

            resolution = infer_resolution_from_market(market)
            if resolution not in ("yes", "no"):
                skipped["ambiguous"] += 1
                continue

            score = _score_sequence(model, sequence, sequence_length, device=device)
            rows.append({
                "id": market_id,
                "question": market.get("question", ""),
                "price_yes": float(snapshot_price),
                "snapshot_price": float(snapshot_price),
                "volume_24h": float(market.get("volume24hr", 0) or 0),
                "liquidity": float(market.get("liquidity", 0) or 0),
                "spread": float(market.get("spread", 0) or 0),
                "model_score": score,
                "resolved": 1 if resolution == "yes" else 0,
                "sequence_length": int(sequence_length),
                "end_date": market.get("endDate", ""),
                "slug": market.get("slug", ""),
            })
        except Exception as exc:
            skipped["exceptions"] += 1
            logger.debug("TS resolved scorer omitió market=%s: %s", market_id, exc)

    if any(skipped.values()):
        logger.warning("TS resolved scorer omitió mercados: %s", skipped)
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Score active markets with PriceSequenceGRU")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model-path", default="data/models/ts_gru/best_ts_gru_model.pt")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--max-markets", type=int, default=None)
    parser.add_argument("--no-fetch-missing-history", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    ts_data_cfg = cfg.get("ts_data", {})
    scoring_cfg = cfg.get("scoring", {})
    ts_model_cfg = cfg.get("ts_model", {})

    raw_dir = args.raw_dir or data_cfg.get("raw_dir", "data/raw")
    top_k = args.top or scoring_cfg.get("top_k", 20)
    max_markets = args.max_markets or scoring_cfg.get("max_markets", 1000)

    model = PriceSequenceGRU(
        input_dim=3,
        hidden_dim=ts_model_cfg.get("hidden_dim", 64),
        num_layers=ts_model_cfg.get("num_layers", 1),
        dropout=ts_model_cfg.get("dropout", 0.2),
        task="classification",
    )
    model.load_state_dict(torch.load(args.model_path, map_location="cpu", weights_only=True))

    client = PolymarketDataClient()
    price_histories = _load_cached_histories(raw_dir)
    df = score_active_markets_ts(
        model,
        client,
        price_histories=price_histories,
        fetch_missing_history=not args.no_fetch_missing_history,
        top_k=top_k,
        max_markets=max_markets,
        seq_len=ts_data_cfg.get("seq_len", 64),
        min_points=ts_data_cfg.get("min_points", 5),
    )

    print(f"\n{'='*80}")
    print(f"TOP {len(df)} MERCADOS POR TS-GRU SCORE")
    print(f"{'='*80}")
    for i, (_, row) in enumerate(df.iterrows(), 1):
        print(
            f"\n{i}. {row['question'][:70]}"
            f"\n   Score: {row['model_score']:.3f} | "
            f"Precio: ${row['price_yes']:.2f} | "
            f"SeqLen: {row['sequence_length']} | "
            f"Vol24h: ${row['volume_24h']:,.0f}"
        )


if __name__ == "__main__":
    main()
