"""Embeddings de texto para preguntas de mercados usando Sentence Transformers."""

import numpy as np

TEXT_EMBED_DIM = 384  # Dimensión de all-MiniLM-L6-v2


class TextEncoder:
    """Genera embeddings semánticos de las preguntas de mercados."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy loading del modelo de sentence transformers."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            try:
                self._model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def embed_dim(self) -> int:
        return TEXT_EMBED_DIM

    def encode(self, text: str) -> np.ndarray:
        """Genera embedding para un texto."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Genera embeddings para un batch de textos."""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )
        return embeddings.astype(np.float32)

    def encode_markets(self, markets: list[dict]) -> np.ndarray:
        """Extrae y codifica las preguntas de una lista de mercados."""
        questions = [m.get("question", "") for m in markets]
        return self.encode_batch(questions)


class DummyTextEncoder:
    """Encoder de texto dummy que genera embeddings aleatorios (para testing)."""

    def __init__(self, embed_dim: int = TEXT_EMBED_DIM):
        self._embed_dim = embed_dim

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def encode(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        return rng.standard_normal(self._embed_dim).astype(np.float32)

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        return np.stack([self.encode(t) for t in texts])

    def encode_markets(self, markets: list[dict]) -> np.ndarray:
        questions = [m.get("question", "") for m in markets]
        return self.encode_batch(questions)
