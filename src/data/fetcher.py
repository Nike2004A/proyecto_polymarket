"""Orquestador de fetch masivo con cache local."""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from .client import PolymarketDataClient

logger = logging.getLogger(__name__)


class DataFetcher:
    """Descarga y cachea datos de Polymarket en disco."""

    def __init__(self, output_dir: str = "data/raw", client: PolymarketDataClient | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = client or PolymarketDataClient()

    def _save_json(self, data: list | dict, filename: str) -> Path:
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        return path

    def fetch_active_markets(self, max_markets: int = 2000) -> list[dict]:
        """Descarga mercados activos y guarda en disco."""
        print(f"Descargando hasta {max_markets} mercados activos...")
        markets = self.client.get_all_active_markets(max_markets=max_markets)
        parsed = [self.client.parse_market(m) for m in markets]
        self._save_json(parsed, "active_markets.json")
        print(f"  -> {len(parsed)} mercados activos guardados.")
        return parsed

    def fetch_resolved_markets(self, max_markets: int = 5000) -> list[dict]:
        """Descarga mercados resueltos (para training labels)."""
        print(f"Descargando hasta {max_markets} mercados resueltos...")
        markets = self.client.get_all_resolved_markets(max_markets=max_markets)
        parsed = [self.client.parse_market(m) for m in markets]
        self._save_json(parsed, "resolved_markets.json")
        print(f"  -> {len(parsed)} mercados resueltos guardados.")
        return parsed

    def fetch_tags(self) -> list[dict]:
        """Descarga etiquetas/categorías."""
        print("Descargando tags...")
        tags = self.client.get_tags()
        self._save_json(tags, "tags.json")
        print(f"  -> {len(tags)} tags guardados.")
        return tags

    def fetch_order_books(self, markets: list[dict], max_books: int = 500) -> list[dict]:
        """Descarga order books para mercados con clobTokenIds."""
        print(f"Descargando order books (hasta {max_books})...")
        books = []
        count = 0
        for market in tqdm(markets, desc="Order books"):
            token_ids = market.get("clobTokenIds", [])
            if not token_ids or count >= max_books:
                break
            for token_id in token_ids[:1]:  # Solo el primer token (Yes)
                try:
                    book = self.client.get_order_book(token_id)
                    book["market_id"] = market.get("id", "")
                    book["token_id"] = token_id
                    books.append(book)
                    count += 1
                except Exception as e:
                    logger.debug(
                        "Order book fallido para market=%s token=%s: %s",
                        market.get("id", "?"), token_id, e,
                    )
                    continue
        self._save_json(books, "order_books.json")
        print(f"  -> {len(books)} order books guardados.")
        return books

    def fetch_price_histories(
        self, markets: list[dict], max_histories: int = 500
    ) -> list[dict]:
        """Descarga historial de precios para mercados con clobTokenIds."""
        print(f"Descargando historiales de precios (hasta {max_histories})...")
        histories = []
        count = 0
        for market in tqdm(markets, desc="Price histories"):
            token_ids = market.get("clobTokenIds", [])
            if not token_ids or count >= max_histories:
                break
            token_id = token_ids[0]
            try:
                history = self.client.get_price_history(token_id)
                histories.append({
                    "market_id": market.get("id", ""),
                    "token_id": token_id,
                    "history": history,
                })
                count += 1
            except Exception as e:
                logger.debug(
                    "Price history fallido para market=%s token=%s: %s",
                    market.get("id", "?"), token_id, e,
                )
                continue
        self._save_json(histories, "price_histories.json")
        print(f"  -> {len(histories)} historiales guardados.")
        return histories

    def fetch_all(self, max_active: int = 2000, max_resolved: int = 5000) -> dict:
        """Descarga completa de todos los datos necesarios."""
        timestamp = datetime.now().isoformat()
        print(f"=== Fetch completo iniciado: {timestamp} ===\n")

        tags = self.fetch_tags()
        active = self.fetch_active_markets(max_markets=max_active)
        resolved = self.fetch_resolved_markets(max_markets=max_resolved)
        books = self.fetch_order_books(active, max_books=500)
        histories = self.fetch_price_histories(active, max_histories=500)

        metadata = {
            "timestamp": timestamp,
            "active_count": len(active),
            "resolved_count": len(resolved),
            "books_count": len(books),
            "histories_count": len(histories),
            "tags_count": len(tags),
        }
        self._save_json(metadata, "fetch_metadata.json")

        print(f"\n=== Fetch completo finalizado ===")
        print(f"  Mercados activos:   {len(active)}")
        print(f"  Mercados resueltos: {len(resolved)}")
        print(f"  Order books:        {len(books)}")
        print(f"  Price histories:    {len(histories)}")

        return metadata


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Polymarket Data Fetcher")
    parser.add_argument(
        "--mode",
        choices=["full", "active", "resolved", "tags"],
        default="full",
        help="Modo de descarga",
    )
    parser.add_argument("--output", default=None, help="Directorio de salida")
    parser.add_argument("--max-active", type=int, default=None)
    parser.add_argument("--max-resolved", type=int, default=None)
    parser.add_argument("--config", default="config/config.yaml", help="Ruta a config.yaml")
    args = parser.parse_args()

    # Cargar config como base, CLI args sobrescriben
    try:
        from ..config import load_config
        cfg = load_config(args.config)
        data_cfg = cfg.get("data", {})
    except (FileNotFoundError, ImportError):
        data_cfg = {}

    output_dir = args.output or data_cfg.get("raw_dir", "data/raw/")
    max_active = args.max_active or data_cfg.get("max_active_markets", 2000)
    max_resolved = args.max_resolved or data_cfg.get("max_resolved_markets", 5000)

    fetcher = DataFetcher(output_dir=output_dir)

    if args.mode == "full":
        fetcher.fetch_all(max_active=max_active, max_resolved=max_resolved)
    elif args.mode == "active":
        fetcher.fetch_active_markets(max_markets=max_active)
    elif args.mode == "resolved":
        fetcher.fetch_resolved_markets(max_markets=max_resolved)
    elif args.mode == "tags":
        fetcher.fetch_tags()


if __name__ == "__main__":
    main()
