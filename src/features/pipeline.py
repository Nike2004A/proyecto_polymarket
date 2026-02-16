"""Pipeline completo de feature engineering: raw market -> feature tensors."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .numerical import (
    extract_numerical_features,
    extract_numerical_features_batch,
    NUM_NUMERICAL_FEATURES,
    NUMERICAL_FEATURE_NAMES,
)
from .categorical import CategoryEncoder
from .text import TextEncoder, DummyTextEncoder

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Pipeline completo que transforma mercados crudos en tensores de features."""

    def __init__(
        self,
        text_encoder: TextEncoder | DummyTextEncoder | None = None,
        category_encoder: CategoryEncoder | None = None,
        scaler: StandardScaler | None = None,
        use_dummy_text: bool = False,
    ):
        if text_encoder is not None:
            self.text_encoder = text_encoder
        elif use_dummy_text:
            self.text_encoder = DummyTextEncoder()
        else:
            self.text_encoder = TextEncoder()

        self.category_encoder = category_encoder or CategoryEncoder()
        self.scaler = scaler or StandardScaler()
        self._fitted = False

    @property
    def num_categories(self) -> int:
        return self.category_encoder.num_categories

    @property
    def text_embed_dim(self) -> int:
        return self.text_encoder.embed_dim

    @property
    def num_numerical_features(self) -> int:
        return NUM_NUMERICAL_FEATURES

    def transform_single(
        self,
        market: dict,
        price_history: list[dict] | None = None,
        order_book: dict | None = None,
    ) -> dict:
        """
        Transforma un mercado individual en features.

        El mercado debe estar preprocesado (con preprocess_market_dict) para
        que las features temporales estén calculadas.

        Returns:
            Dict con keys: numerical (np.ndarray), category_id (int),
            text_embedding (np.ndarray).
        """
        from ..data.preprocessing import preprocess_market_dict

        # Asegurar que el mercado tiene features temporales y derivadas
        market = preprocess_market_dict(market)

        numerical = extract_numerical_features(market, price_history, order_book)

        if self._fitted:
            numerical = self.scaler.transform(numerical.reshape(1, -1)).squeeze()

        category_id = self.category_encoder.encode(market)
        question = market.get("question", "")
        text_embedding = self.text_encoder.encode(question)

        return {
            "numerical": numerical.astype(np.float32),
            "category_id": int(category_id),
            "text_embedding": text_embedding.astype(np.float32),
        }

    def fit_transform_batch(
        self,
        markets: list[dict],
        price_histories: dict | None = None,
        order_books: dict | None = None,
    ) -> dict:
        """
        Ajusta el scaler y transforma un batch de mercados.

        Args:
            markets: Lista de dicts de mercados.
            price_histories: Dict {market_id: [price_records]}. Si se
                proporcionan, se usan para calcular momentum y volatilidad.
            order_books: Dict {market_id: order_book}. Si se proporcionan,
                se usan para calcular bid_depth, ask_depth, book_imbalance.

        Returns:
            Dict con keys: numerical (N, 14), category_ids (N,),
            text_embeddings (N, 384).
        """
        from ..data.preprocessing import preprocess_market_dict

        # Preprocesar cada mercado para asegurar features temporales
        preprocessed = [preprocess_market_dict(m) for m in markets]

        # Numerical features
        markets_df = pd.DataFrame(preprocessed)
        numerical = extract_numerical_features_batch(
            markets_df, price_histories, order_books
        )

        # Log de features que quedaron en cero (posible dato faltante)
        zero_cols = (numerical == 0).all(axis=0)
        for i, is_zero in enumerate(zero_cols):
            if is_zero and NUMERICAL_FEATURE_NAMES[i] in (
                "price_momentum_7d", "price_volatility_7d",
                "bid_depth", "ask_depth", "book_imbalance",
            ):
                source = "price_histories" if "momentum" in NUMERICAL_FEATURE_NAMES[i] or "volatility" in NUMERICAL_FEATURE_NAMES[i] else "order_books"
                logger.warning(
                    "Feature '%s' es cero para todos los mercados. "
                    "Verifica que se está pasando '%s' al pipeline.",
                    NUMERICAL_FEATURE_NAMES[i], source,
                )

        # Fit scaler
        self.scaler.fit(numerical)
        numerical_scaled = self.scaler.transform(numerical).astype(np.float32)
        self._fitted = True

        # Category IDs
        category_ids = self.category_encoder.encode_batch(preprocessed)

        # Text embeddings
        text_embeddings = self.text_encoder.encode_markets(preprocessed)

        return {
            "numerical": numerical_scaled,
            "category_ids": category_ids,
            "text_embeddings": text_embeddings,
        }

    def transform_batch(
        self,
        markets: list[dict],
        price_histories: dict | None = None,
        order_books: dict | None = None,
    ) -> dict:
        """Transforma un batch sin reajustar el scaler."""
        from ..data.preprocessing import preprocess_market_dict

        preprocessed = [preprocess_market_dict(m) for m in markets]
        markets_df = pd.DataFrame(preprocessed)
        numerical = extract_numerical_features_batch(
            markets_df, price_histories, order_books
        )

        if self._fitted:
            numerical = self.scaler.transform(numerical).astype(np.float32)

        category_ids = self.category_encoder.encode_batch(preprocessed)
        text_embeddings = self.text_encoder.encode_markets(preprocessed)

        return {
            "numerical": numerical,
            "category_ids": category_ids,
            "text_embeddings": text_embeddings,
        }

    def save(self, directory: str) -> None:
        """Guarda el pipeline (scaler + category encoder) en disco."""
        import joblib

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path / "scaler.pkl")
        self.category_encoder.save(str(path / "category_encoder.json"))
        metadata = {
            "fitted": self._fitted,
            "num_numerical": self.num_numerical_features,
            "num_categories": self.num_categories,
            "text_embed_dim": self.text_embed_dim,
            "feature_names": NUMERICAL_FEATURE_NAMES,
        }
        with open(path / "pipeline_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, directory: str, use_dummy_text: bool = False) -> "FeaturePipeline":
        """Carga un pipeline previamente guardado."""
        import joblib

        path = Path(directory)
        scaler = joblib.load(path / "scaler.pkl")
        category_encoder = CategoryEncoder.load(str(path / "category_encoder.json"))

        pipeline = cls(
            category_encoder=category_encoder,
            scaler=scaler,
            use_dummy_text=use_dummy_text,
        )
        pipeline._fitted = True
        return pipeline


def _load_auxiliary_data(input_dir: Path) -> tuple[dict, dict]:
    """
    Carga price_histories y order_books desde disco y los indexa por market_id.

    Returns:
        (price_histories_by_id, order_books_by_id)
    """
    price_histories: dict = {}
    order_books: dict = {}

    ph_path = input_dir / "price_histories.json"
    if ph_path.exists():
        with open(ph_path) as f:
            raw_histories = json.load(f)
        for entry in raw_histories:
            mid = entry.get("market_id", "")
            if mid:
                price_histories[mid] = entry.get("history", [])
        logger.info("Cargados %d price histories.", len(price_histories))
    else:
        logger.warning(
            "No se encontró %s. Features de momentum/volatilidad serán cero.",
            ph_path,
        )

    ob_path = input_dir / "order_books.json"
    if ob_path.exists():
        with open(ob_path) as f:
            raw_books = json.load(f)
        for entry in raw_books:
            mid = entry.get("market_id", "")
            if mid:
                order_books[mid] = entry
        logger.info("Cargados %d order books.", len(order_books))
    else:
        logger.warning(
            "No se encontró %s. Features de order book serán cero.",
            ob_path,
        )

    return price_histories, order_books


def run_pipeline(input_dir: str = "data/raw", output_dir: str = "data/processed"):
    """Script para ejecutar el pipeline completo de features."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline")
    parser.add_argument("--input", default=input_dir)
    parser.add_argument("--output", default=output_dir)
    parser.add_argument("--dummy-text", action="store_true", help="Usar embeddings dummy")
    parser.add_argument(
        "--snapshot-offset", type=int, default=7,
        help="Días antes de resolución para tomar snapshot de precio (default: 7)",
    )
    parser.add_argument("--config", default=None, help="Ruta a config.yaml")
    args = parser.parse_args()

    # Cargar config si se proporciona
    if args.config:
        from ..config import load_config
        cfg = load_config(args.config)
        args.input = cfg.get("data", {}).get("raw_dir", args.input)
        args.output = cfg.get("data", {}).get("processed_dir", args.output)
        args.dummy_text = cfg.get("features", {}).get("use_dummy_text", args.dummy_text)

    from ..data.preprocessing import compute_label, get_snapshot_price

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Cargar mercados resueltos
    resolved_path = input_path / "resolved_markets.json"
    if not resolved_path.exists():
        logger.error("No se encontró %s", resolved_path)
        return

    with open(resolved_path) as f:
        resolved_markets = json.load(f)

    # Cargar datos auxiliares (price_histories, order_books)
    price_histories, order_books = _load_auxiliary_data(input_path)

    # Filtrar mercados con resolución válida usando snapshot prices
    valid_markets = []
    labels = []
    skipped = {"no_snapshot": 0, "ambiguous": 0, "no_prices": 0}

    for m in resolved_markets:
        snapshot_price = get_snapshot_price(
            m, price_histories, snapshot_offset_days=args.snapshot_offset
        )

        if snapshot_price is None:
            skipped["no_snapshot"] += 1
            continue

        label = compute_label(m, snapshot_price)
        if label == -1:
            skipped["ambiguous"] += 1
            continue

        # Inyectar el snapshot price como outcomePrices para que las features
        # numéricas usen el precio del snapshot, no el precio post-resolución
        m_copy = m.copy()
        m_copy["outcomePrices"] = [snapshot_price, 1.0 - snapshot_price]

        valid_markets.append(m_copy)
        labels.append(label)

    logger.info("Mercados válidos para entrenamiento: %d", len(valid_markets))
    logger.info("  Positivos (buy): %d", sum(labels))
    logger.info("  Negativos (no buy): %d", len(labels) - sum(labels))
    logger.info("  Descartados: %s", skipped)

    if not valid_markets:
        logger.error("No hay mercados válidos para entrenar. Verifica los datos.")
        return

    # Feature extraction (con datos auxiliares)
    pipeline = FeaturePipeline(use_dummy_text=args.dummy_text)
    features = pipeline.fit_transform_batch(
        valid_markets,
        price_histories=price_histories,
        order_books=order_books,
    )

    # Guardar timestamps de endDate para temporal split
    end_dates = []
    for m in valid_markets:
        ed = m.get("endDate", "")
        try:
            end_dates.append(pd.to_datetime(ed, utc=True).timestamp())
        except (ValueError, TypeError):
            end_dates.append(0.0)

    # Guardar
    np.save(output_path / "numerical_features.npy", features["numerical"])
    np.save(output_path / "category_ids.npy", features["category_ids"])
    np.save(output_path / "text_embeddings.npy", features["text_embeddings"])
    np.save(output_path / "labels.npy", np.array(labels, dtype=np.float32))
    np.save(output_path / "end_dates.npy", np.array(end_dates, dtype=np.float64))
    pipeline.save(str(output_path / "pipeline"))

    logger.info("Features guardadas en %s/", output_path)
    logger.info("  numerical_features: %s", features["numerical"].shape)
    logger.info("  category_ids: %s", features["category_ids"].shape)
    logger.info("  text_embeddings: %s", features["text_embeddings"].shape)
    logger.info("  labels: %d", len(labels))
    logger.info("  end_dates: %d (para temporal split)", len(end_dates))


if __name__ == "__main__":
    run_pipeline()
