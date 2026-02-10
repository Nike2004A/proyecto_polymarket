"""Pipeline completo de feature engineering: raw market -> feature tensors."""

import json
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

        Returns:
            Dict con keys: numerical (np.ndarray), category_id (int),
            text_embedding (np.ndarray).
        """
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

        Returns:
            Dict con keys: numerical (N, 14), category_ids (N,),
            text_embeddings (N, 384).
        """
        # Numerical features
        markets_df = pd.DataFrame(markets)
        numerical = extract_numerical_features_batch(
            markets_df, price_histories, order_books
        )

        # Fit scaler
        self.scaler.fit(numerical)
        numerical_scaled = self.scaler.transform(numerical).astype(np.float32)
        self._fitted = True

        # Category IDs
        category_ids = self.category_encoder.encode_batch(markets)

        # Text embeddings
        text_embeddings = self.text_encoder.encode_markets(markets)

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
        markets_df = pd.DataFrame(markets)
        numerical = extract_numerical_features_batch(
            markets_df, price_histories, order_books
        )

        if self._fitted:
            numerical = self.scaler.transform(numerical).astype(np.float32)

        category_ids = self.category_encoder.encode_batch(markets)
        text_embeddings = self.text_encoder.encode_markets(markets)

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


def run_pipeline(input_dir: str = "data/raw", output_dir: str = "data/processed"):
    """Script para ejecutar el pipeline completo de features."""
    import argparse

    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline")
    parser.add_argument("--input", default=input_dir)
    parser.add_argument("--output", default=output_dir)
    parser.add_argument("--dummy-text", action="store_true", help="Usar embeddings dummy")
    args = parser.parse_args()

    from ..data.preprocessing import preprocess_markets, compute_label

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Cargar mercados resueltos
    resolved_path = Path(args.input) / "resolved_markets.json"
    if not resolved_path.exists():
        print(f"Error: No se encontró {resolved_path}")
        return

    with open(resolved_path) as f:
        resolved_markets = json.load(f)

    # Filtrar mercados con resolución válida
    valid_markets = []
    labels = []
    for m in resolved_markets:
        outcome_prices = m.get("outcomePrices", [])
        if isinstance(outcome_prices, str):
            try:
                outcome_prices = json.loads(outcome_prices)
            except (json.JSONDecodeError, TypeError):
                continue

        if not outcome_prices:
            continue

        price_yes = float(outcome_prices[0]) if outcome_prices else 0.5
        label = compute_label(m, price_yes)
        if label >= 0:
            valid_markets.append(m)
            labels.append(label)

    print(f"Mercados válidos para entrenamiento: {len(valid_markets)}")
    print(f"  Positivos (buy): {sum(labels)}")
    print(f"  Negativos (no buy): {len(labels) - sum(labels)}")

    # Feature extraction
    pipeline = FeaturePipeline(use_dummy_text=args.dummy_text)
    features = pipeline.fit_transform_batch(valid_markets)

    # Guardar
    np.save(output_path / "numerical_features.npy", features["numerical"])
    np.save(output_path / "category_ids.npy", features["category_ids"])
    np.save(output_path / "text_embeddings.npy", features["text_embeddings"])
    np.save(output_path / "labels.npy", np.array(labels, dtype=np.float32))
    pipeline.save(str(output_path / "pipeline"))

    print(f"\nFeatures guardadas en {output_path}/")
    print(f"  numerical_features: {features['numerical'].shape}")
    print(f"  category_ids: {features['category_ids'].shape}")
    print(f"  text_embeddings: {features['text_embeddings'].shape}")
    print(f"  labels: {len(labels)}")


if __name__ == "__main__":
    run_pipeline()
