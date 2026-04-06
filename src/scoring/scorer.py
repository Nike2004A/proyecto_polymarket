"""Scoring de mercados activos con el modelo entrenado."""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch

from ..data.client import PolymarketDataClient
from ..features.pipeline import FeaturePipeline
from ..model.architecture import MarketValueNet

logger = logging.getLogger(__name__)


def _load_cached_market_context(raw_dir: str | Path | None) -> tuple[dict, dict]:
    """Carga histories/order books cacheados desde data/raw si existen."""
    if raw_dir is None:
        return {}, {}

    raw_path = Path(raw_dir)
    price_histories = {}
    order_books = {}

    ph_path = raw_path / "price_histories.json"
    if ph_path.exists():
        with open(ph_path, encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            price_histories = loaded

    ob_path = raw_path / "order_books.json"
    if ob_path.exists():
        with open(ob_path, encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            order_books = loaded

    return price_histories, order_books


def _fetch_live_market_context(
    client: PolymarketDataClient,
    market: dict,
    need_history: bool,
    need_order_book: bool,
) -> tuple[list[dict] | None, dict | None]:
    """Obtiene contexto faltante directamente de la API para un mercado activo."""
    token_ids = market.get("clobTokenIds", [])
    if not token_ids:
        return None, None

    token_id = token_ids[0]
    price_history = None
    order_book = None

    if need_history:
        try:
            price_history = client.get_price_history(token_id)
        except Exception as exc:
            logger.debug(
                "No se pudo obtener price_history para market=%s: %s",
                market.get("id", "?"),
                exc,
            )

    if need_order_book:
        try:
            order_book = client.get_order_book(token_id)
        except Exception as exc:
            logger.debug(
                "No se pudo obtener order_book para market=%s: %s",
                market.get("id", "?"),
                exc,
            )

    return price_history, order_book


def score_active_markets(
    model: MarketValueNet,
    client: PolymarketDataClient,
    feature_pipeline: FeaturePipeline,
    price_histories: dict | None = None,
    order_books: dict | None = None,
    fetch_missing_context: bool = True,
    device: str | None = None,
    top_k: int = 20,
    max_markets: int = 1000,
) -> pd.DataFrame:
    """
    Puntúa mercados activos y devuelve los top-K con mayor score.

    NOTA sobre score_minus_price:
    El campo score_minus_price es solo una brecha heurística entre score y
    precio actual. No es alpha esperado ni retorno calibrado.

    Args:
        model: Modelo entrenado.
        client: Cliente de Polymarket.
        feature_pipeline: Pipeline de features (ya fitted).
        price_histories: Historiales keyed por market_id para features TS.
        order_books: Order books keyed por market_id.
        fetch_missing_context: Si True, intenta completar contexto faltante
            desde la API para mercados que no existan en el cache local.
        device: Dispositivo de cómputo.
        top_k: Número de mercados a devolver.
        max_markets: Máximo de mercados a evaluar.

    Returns:
        DataFrame con los top-K mercados ordenados por score.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)
    price_histories = price_histories or {}
    order_books = order_books or {}

    # 1. Fetch active markets
    logger.info("Descargando mercados activos (hasta %d)...", max_markets)
    raw_markets = client.get_all_active_markets(max_markets=max_markets)
    parsed = [client.parse_market(m) for m in raw_markets]
    logger.info("  -> %d mercados obtenidos.", len(parsed))

    # 2. Score each market
    results = []
    skipped_count = 0
    skipped_reasons: dict[str, int] = {}
    fetched_context = {"history": 0, "order_book": 0}
    missing_context = {"history": 0, "order_book": 0}

    for market in parsed:
        market_id = market.get("id", "unknown")
        try:
            price_history = price_histories.get(str(market_id))
            order_book = order_books.get(str(market_id))

            has_history = isinstance(price_history, list) and len(price_history) > 0
            has_order_book = isinstance(order_book, dict) and bool(order_book)

            if fetch_missing_context and (not has_history or not has_order_book):
                live_history, live_order_book = _fetch_live_market_context(
                    client,
                    market,
                    need_history=not has_history,
                    need_order_book=not has_order_book,
                )
                if live_history:
                    price_history = live_history
                    has_history = True
                    fetched_context["history"] += 1
                if live_order_book:
                    order_book = live_order_book
                    has_order_book = True
                    fetched_context["order_book"] += 1

            if not has_history:
                missing_context["history"] += 1
            if not has_order_book:
                missing_context["order_book"] += 1

            features = feature_pipeline.transform_single(
                market,
                price_history=price_history,
                order_book=order_book,
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

            # Precio real del dict (features["numerical"][0] está escalado por el scaler)
            op = market.get("outcomePrices", [])
            if isinstance(op, str):
                try:
                    op = json.loads(op)
                except (ValueError, TypeError):
                    op = []
            price_yes = float(op[0]) if isinstance(op, list) and op else float(
                market.get("lastTradePrice") or 0.5
            )
            results.append({
                "id": market_id,
                "question": market.get("question", ""),
                "price_yes": float(price_yes),
                "volume_24h": float(market.get("volume24hr", 0) or 0),
                "liquidity": float(market.get("liquidity", 0) or 0),
                "spread": float(market.get("spread", 0) or 0),
                "model_score": score,
                # Nota: esto es score - price, NO un retorno esperado calibrado.
                # Útil solo como ranking relativo entre mercados.
                "score_minus_price": score - float(price_yes),
                "slug": market.get("slug", ""),
                "end_date": market.get("endDate", ""),
            })
        except Exception as e:
            skipped_count += 1
            reason = type(e).__name__
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            logger.debug("Market %s omitido: %s: %s", market_id, reason, e)

    if skipped_count > 0:
        logger.warning(
            "Se omitieron %d/%d mercados durante scoring. Razones: %s",
            skipped_count, len(parsed), skipped_reasons,
        )
    if any(fetched_context.values()):
        logger.info(
            "Contexto live completado para %d histories y %d order books.",
            fetched_context["history"],
            fetched_context["order_book"],
        )
    if any(missing_context.values()):
        logger.warning(
            "Mercados aún sin contexto completo: histories=%d, order_books=%d.",
            missing_context["history"],
            missing_context["order_book"],
        )

    df = pd.DataFrame(results)
    if df.empty:
        logger.warning("No se pudo puntuar ningún mercado.")
        return df

    df = df.sort_values("model_score", ascending=False)
    return df.head(top_k)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Score Active Markets")
    parser.add_argument("--model-path", default="data/models/best_market_model.pt")
    parser.add_argument("--pipeline-dir", default="data/processed/pipeline")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--max-markets", type=int, default=1000)
    parser.add_argument("--output", default=None)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument(
        "--no-fetch-missing-context",
        action="store_true",
        help="No completar histories/order books faltantes desde la API.",
    )
    parser.add_argument("--config", default=None, help="Ruta a config.yaml")
    args = parser.parse_args()

    # Cargar config si se proporciona
    if args.config:
        from ..config import load_config
        cfg = load_config(args.config)
        data_cfg = cfg.get("data", {})
        scoring_cfg = cfg.get("scoring", {})
        args.top = scoring_cfg.get("top_k", args.top)
        args.max_markets = scoring_cfg.get("max_markets", args.max_markets)
        args.raw_dir = data_cfg.get("raw_dir", args.raw_dir)

    # Cargar pipeline y modelo
    logger.info("Cargando pipeline y modelo...")
    pipeline = FeaturePipeline.load(args.pipeline_dir, use_dummy_text=False)
    model = MarketValueNet(
        num_numerical_features=pipeline.num_numerical_features,
        num_categories=pipeline.num_categories,
        text_embed_dim=pipeline.text_embed_dim,
    )
    model.load_state_dict(
        torch.load(args.model_path, map_location="cpu", weights_only=True)
    )

    client = PolymarketDataClient()
    price_histories, order_books = _load_cached_market_context(args.raw_dir)

    # Scoring
    df = score_active_markets(
        model,
        client,
        pipeline,
        price_histories=price_histories,
        order_books=order_books,
        fetch_missing_context=not args.no_fetch_missing_context,
        top_k=args.top,
        max_markets=args.max_markets,
    )

    # Output
    print(f"\n{'='*80}")
    print(f"TOP {args.top} MERCADOS POR MODEL SCORE")
    print(f"{'='*80}")
    print(
        "\nNOTA: model_score es un score de clasificación, NO una probabilidad\n"
        "calibrada. Usar como ranking relativo, no como retorno esperado.\n"
    )
    for i, (_, row) in enumerate(df.iterrows(), 1):
        print(
            f"\n{i}. {row['question'][:70]}"
            f"\n   Score: {row['model_score']:.3f} | "
            f"Precio: ${row['price_yes']:.2f} | "
            f"Score-Price: {row['score_minus_price']:.3f} | "
            f"Vol24h: ${row['volume_24h']:,.0f}"
        )

    if args.output:
        df.to_csv(args.output, index=False)
        logger.info("Resultados guardados en %s", args.output)


if __name__ == "__main__":
    main()
