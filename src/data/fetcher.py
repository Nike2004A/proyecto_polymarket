"""Orquestador de fetch masivo con cache local."""

import argparse
import json
import logging
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

    def _load_json(self, filename: str, default: list | dict | None = None):
        path = self.output_dir / filename
        if not path.exists():
            return default if default is not None else {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def fetch_active_markets(self, max_markets: int = 3000) -> list[dict]:
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

    def fetch_order_books(self, markets: list[dict], max_books: int = 1000) -> dict:
        """
        Descarga order books para mercados activos.

        Retorna y guarda un dict {market_id: order_book}.
        Solo mercados activos (CLOB activo), no resueltos.
        """
        print(f"Descargando order books (hasta {max_books})...")
        books: dict = self._load_json("order_books.json", default={})
        fetched = 0

        for market in tqdm(markets, desc="Order books"):
            if fetched >= max_books:
                break
            market_id = market.get("id", "")
            if not market_id or market_id in books:
                continue
            token_ids = market.get("clobTokenIds", [])
            if not token_ids:
                continue
            try:
                book = self.client.get_order_book(token_ids[0])
                books[market_id] = book
                fetched += 1
            except Exception as e:
                logger.debug("Order book fallido para market=%s: %s", market_id, e)

        self._save_json(books, "order_books.json")
        print(f"  -> {len(books)} order books en disco ({fetched} nuevos).")
        return books

    def fetch_price_histories(
        self,
        markets: list[dict],
        max_histories: int = 8000,
    ) -> dict:
        """
        Descarga historial de precios para todos los mercados proporcionados.

        - Acepta activos + resueltos combinados.
        - Modo incremental: salta market_ids que ya tienen datos en disco.
        - Retorna y guarda un dict {market_id: [{t, p}, ...]}.

        Este formato es el esperado por preprocessing.py y numerical.py.
        """
        print(f"Descargando price histories (hasta {max_histories} nuevos)...")

        # Carga datos existentes para modo incremental
        histories: dict = self._load_json("price_histories.json", default={})
        already_have = len(histories)

        # Deduplica mercados por id
        seen_ids: set[str] = set()
        unique_markets: list[dict] = []
        for m in markets:
            mid = m.get("id", "")
            if mid and mid not in seen_ids:
                seen_ids.add(mid)
                unique_markets.append(m)

        to_fetch = [m for m in unique_markets if m.get("id", "") not in histories]
        print(f"  {already_have} ya en disco, {len(to_fetch)} pendientes de descargar.")

        fetched = 0
        for market in tqdm(to_fetch, desc="Price histories"):
            if fetched >= max_histories:
                break
            market_id = market.get("id", "")
            token_ids = market.get("clobTokenIds", [])
            if not token_ids:
                histories[market_id] = []
                continue
            try:
                history = self.client.get_price_history(token_ids[0])
                histories[market_id] = history
                fetched += 1
                # Guardado incremental cada 200 mercados para no perder progreso
                if fetched % 200 == 0:
                    self._save_json(histories, "price_histories.json")
                    logger.info("Checkpoint: %d histories guardadas.", len(histories))
            except Exception as e:
                logger.debug("Price history fallido para market=%s: %s", market_id, e)
                histories[market_id] = []

        self._save_json(histories, "price_histories.json")
        print(f"  -> {len(histories)} historiales en disco ({fetched} nuevos).")
        return histories

    def fetch_all(
        self,
        max_active: int = 1000,
        max_resolved: int = 20000,
        max_books: int | None = None,
    ) -> dict:
        """
        Descarga completa: training viene exclusivamente de resueltos.

        Los resueltos tienen la serie temporal completa + outcome conocido (label).
        Cualquier punto de su historia simula un "mercado activo" con su precio
        en ese momento — no necesitamos activos para entrenar.

        Los activos se descargan en cantidad pequeña solo para el demo de scoring
        en vivo (predicción sobre mercados sin outcome aún).

        Los order books solo existen para activos (el CLOB se cierra al resolver);
        se descargan todos sin límite artificial.

        Args:
            max_active: Mercados activos para scoring en vivo (pequeño).
            max_resolved: Mercados resueltos para entrenamiento (grande).
            max_books: Límite de order books. None = todos los activos descargados.
        """
        timestamp = datetime.now().isoformat()
        print(f"=== Fetch completo iniciado: {timestamp} ===\n")

        tags = self.fetch_tags()
        active   = self.fetch_active_markets(max_markets=max_active)
        resolved = self.fetch_resolved_markets(max_markets=max_resolved)

        # Order books para todos los activos (sin límite artificial)
        books = self.fetch_order_books(active, max_books=max_books or len(active))

        # Price histories para TODOS (activos + resueltos) — sin límite separado
        all_markets = active + resolved
        max_histories = len(all_markets)  # intentar todos
        histories = self.fetch_price_histories(all_markets, max_histories=max_histories)

        # Calcular cobertura de histories sobre resueltos (datos de entrenamiento)
        resolved_ids = {m.get("id", "") for m in resolved}
        covered_resolved = sum(
            1 for mid in resolved_ids if histories.get(mid)
        )
        coverage_pct = 100.0 * covered_resolved / len(resolved_ids) if resolved_ids else 0.0

        metadata = {
            "timestamp": timestamp,
            "active_count": len(active),
            "resolved_count": len(resolved),
            "books_count": len(books),
            "histories_total": len(histories),
            "histories_with_data": sum(1 for v in histories.values() if v),
            "resolved_histories_coverage_pct": round(coverage_pct, 1),
            "tags_count": len(tags),
        }
        self._save_json(metadata, "fetch_metadata.json")

        print(f"\n=== Fetch completo finalizado ===")
        print(f"  Mercados activos:            {len(active)}")
        print(f"  Mercados resueltos:          {len(resolved)}")
        print(f"  Order books:                 {len(books)}")
        print(f"  Price histories (total):     {len(histories)}")
        print(f"  Cobertura resueltos:         {coverage_pct:.1f}%")

        return metadata


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Polymarket Data Fetcher")
    parser.add_argument(
        "--mode",
        choices=["full", "active", "resolved", "histories", "tags"],
        default="full",
        help="Modo de descarga",
    )
    parser.add_argument("--output", default=None, help="Directorio de salida")
    parser.add_argument("--max-active", type=int, default=None,
                        help="Límite de mercados activos (default: config o 3000)")
    parser.add_argument("--max-resolved", type=int, default=None,
                        help="Límite de mercados resueltos (default: config o 20000)")
    parser.add_argument("--config", default="config/config.yaml", help="Ruta a config.yaml")
    args = parser.parse_args()

    try:
        from ..config import load_config
        cfg = load_config(args.config)
        data_cfg = cfg.get("data", {})
    except (FileNotFoundError, ImportError):
        data_cfg = {}

    output_dir   = args.output       or data_cfg.get("raw_dir", "data/raw/")
    max_active   = args.max_active   or data_cfg.get("max_active_markets", 1000)
    max_resolved = args.max_resolved or data_cfg.get("max_resolved_markets", 20000)
    # max_books None = sin límite artificial, descarga todos los activos
    max_books_cfg = data_cfg.get("max_order_books")

    fetcher = DataFetcher(output_dir=output_dir)

    if args.mode == "full":
        fetcher.fetch_all(max_active=max_active, max_resolved=max_resolved, max_books=max_books_cfg)
    elif args.mode == "active":
        fetcher.fetch_active_markets(max_markets=max_active)
    elif args.mode == "resolved":
        fetcher.fetch_resolved_markets(max_markets=max_resolved)
    elif args.mode == "histories":
        # Reanudar/completar solo histories con markets ya en disco
        active   = fetcher._load_json("active_markets.json",   default=[])
        resolved = fetcher._load_json("resolved_markets.json", default=[])
        fetcher.fetch_price_histories(active + resolved, max_histories=len(active) + len(resolved))
    elif args.mode == "tags":
        fetcher.fetch_tags()


if __name__ == "__main__":
    main()
