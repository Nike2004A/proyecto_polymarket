"""Scoring de mercados activos con el modelo entrenado."""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

from ..data.client import PolymarketDataClient
from ..features.pipeline import FeaturePipeline
from ..model.architecture import MarketValueNet

logger = logging.getLogger(__name__)


def score_active_markets(
    model: MarketValueNet,
    client: PolymarketDataClient,
    feature_pipeline: FeaturePipeline,
    device: str | None = None,
    top_k: int = 20,
    max_markets: int = 1000,
) -> pd.DataFrame:
    """
    Puntúa mercados activos y devuelve los top-K con mayor score.

    NOTA sobre expected_alpha:
    El campo 'expected_alpha' es una heurística (score - price_yes).
    El model_score NO es una probabilidad calibrada de que el mercado
    resuelva "Yes". Es un score de clasificación que indica si el modelo
    cree que es buena compra dado el precio actual. Por tanto,
    expected_alpha NO debe interpretarse como retorno esperado real,
    sino como una señal relativa de ranking.

    Args:
        model: Modelo entrenado.
        client: Cliente de Polymarket.
        feature_pipeline: Pipeline de features (ya fitted).
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

    # 1. Fetch active markets
    logger.info("Descargando mercados activos (hasta %d)...", max_markets)
    raw_markets = client.get_all_active_markets(max_markets=max_markets)
    parsed = [client.parse_market(m) for m in raw_markets]
    logger.info("  -> %d mercados obtenidos.", len(parsed))

    # 2. Score each market
    results = []
    skipped_count = 0
    skipped_reasons: dict[str, int] = {}

    for market in parsed:
        market_id = market.get("id", "unknown")
        try:
            features = feature_pipeline.transform_single(market)
            num_tensor = (
                torch.FloatTensor(features["numerical"]).unsqueeze(0).to(device)
            )
            cat_tensor = torch.LongTensor([features["category_id"]]).to(device)
            txt_tensor = (
                torch.FloatTensor(features["text_embedding"]).unsqueeze(0).to(device)
            )

            with torch.no_grad():
                score = model(num_tensor, cat_tensor, txt_tensor).item()

            price_yes = features["numerical"][0]
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
    parser.add_argument("--config", default=None, help="Ruta a config.yaml")
    args = parser.parse_args()

    # Cargar config si se proporciona
    if args.config:
        from ..config import load_config
        cfg = load_config(args.config)
        scoring_cfg = cfg.get("scoring", {})
        args.top = scoring_cfg.get("top_k", args.top)
        args.max_markets = scoring_cfg.get("max_markets", args.max_markets)

    # Cargar pipeline y modelo
    logger.info("Cargando pipeline y modelo...")
    pipeline = FeaturePipeline.load(args.pipeline_dir, use_dummy_text=False)
    model = MarketValueNet(
        num_numerical_features=pipeline.num_numerical_features,
        num_categories=pipeline.num_categories,
        text_embed_dim=pipeline.text_embed_dim,
    )
    model.load_state_dict(torch.load(args.model_path, weights_only=True))

    client = PolymarketDataClient()

    # Scoring
    df = score_active_markets(
        model, client, pipeline, top_k=args.top, max_markets=args.max_markets
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
