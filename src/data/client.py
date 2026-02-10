"""Cliente para consumir datos de las APIs de Polymarket (Gamma + CLOB + Data)."""

import json
import time
from typing import Optional

import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
DATA_BASE = "https://data-api.polymarket.com"


class PolymarketDataClient:
    """Cliente unificado para las APIs de Polymarket."""

    def __init__(self, rate_limit_delay: float = 0.2):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self.rate_limit_delay = rate_limit_delay

    def _get(self, url: str, params: Optional[dict] = None) -> dict | list:
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        time.sleep(self.rate_limit_delay)
        return resp.json()

    # ── Gamma API (Metadata) ─────────────────────────────────────────

    def get_active_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        order: str = "volume24hr",
        ascending: bool = False,
    ) -> list[dict]:
        """Obtiene mercados activos ordenados por volumen."""
        params = {
            "active": "true",
            "closed": "false",
            "limit": limit,
            "offset": offset,
            "order": order,
            "ascending": str(ascending).lower(),
        }
        return self._get(f"{GAMMA_BASE}/markets", params)

    def get_all_active_markets(self, max_markets: int = 2000) -> list[dict]:
        """Paginación completa de mercados activos."""
        all_markets: list[dict] = []
        offset = 0
        limit = 100
        while offset < max_markets:
            batch = self.get_active_markets(limit=limit, offset=offset)
            if not batch:
                break
            all_markets.extend(batch)
            offset += limit
        return all_markets

    def get_resolved_markets(
        self, limit: int = 100, offset: int = 0
    ) -> list[dict]:
        """Obtiene mercados ya resueltos (para generar labels de entrenamiento)."""
        params = {
            "closed": "true",
            "limit": limit,
            "offset": offset,
            "order": "endDate",
            "ascending": "false",
        }
        return self._get(f"{GAMMA_BASE}/markets", params)

    def get_all_resolved_markets(self, max_markets: int = 5000) -> list[dict]:
        """Paginación completa de mercados resueltos."""
        all_markets: list[dict] = []
        offset = 0
        limit = 100
        while offset < max_markets:
            batch = self.get_resolved_markets(limit=limit, offset=offset)
            if not batch:
                break
            all_markets.extend(batch)
            offset += limit
        return all_markets

    def get_market(self, market_id: str) -> dict:
        """Detalle de un mercado por ID."""
        return self._get(f"{GAMMA_BASE}/markets/{market_id}")

    def get_events(
        self, active: bool = True, limit: int = 100
    ) -> list[dict]:
        """Obtiene eventos."""
        params = {"active": str(active).lower(), "limit": limit}
        return self._get(f"{GAMMA_BASE}/events", params)

    def get_event(self, event_id: str) -> dict:
        """Detalle de un evento."""
        return self._get(f"{GAMMA_BASE}/events/{event_id}")

    def get_tags(self) -> list[dict]:
        """Obtiene las categorías/etiquetas disponibles."""
        return self._get(f"{GAMMA_BASE}/tags")

    # ── CLOB API (Order Book) ────────────────────────────────────────

    def get_order_book(self, token_id: str) -> dict:
        """Obtiene el order book del CLOB para un token."""
        return self._get(f"{CLOB_BASE}/book", {"token_id": token_id})

    def get_clob_price(self, token_id: str) -> dict:
        """Precio actual de un token en el CLOB."""
        return self._get(f"{CLOB_BASE}/price", {"token_id": token_id})

    def get_clob_markets(self) -> list[dict]:
        """Mercados del CLOB con datos de trading."""
        return self._get(f"{CLOB_BASE}/markets")

    # ── Data API (Series Temporales) ─────────────────────────────────

    def get_price_history(
        self,
        token_id: str,
        interval: str = "1d",
        fidelity: int = 60,
    ) -> list[dict]:
        """Historial de precios por token."""
        params = {
            "market": token_id,
            "interval": interval,
            "fidelity": fidelity,
        }
        return self._get(f"{DATA_BASE}/prices-history", params)

    # ── Utilidades ───────────────────────────────────────────────────

    @staticmethod
    def parse_market(raw: dict) -> dict:
        """Parsea campos JSON-string del mercado."""
        parsed = raw.copy()
        for field in ["outcomes", "outcomePrices", "clobTokenIds"]:
            val = parsed.get(field)
            if isinstance(val, str):
                try:
                    parsed[field] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    parsed[field] = []
        return parsed
