import hashlib
import math
from functools import lru_cache
from typing import Protocol

from .config import Settings


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbeddingProvider:
    """Small deterministic CPU embedder used until a configured model is wired in."""

    def __init__(self, *, dimension: int = 32) -> None:
        self._dimension = dimension
        self.calls = 0
        self.embedded_text_count = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.embedded_text_count += len(texts)
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        values = [0.0] * self._dimension
        tokens = text.split() or [text]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for index, byte in enumerate(digest):
                values[index % self._dimension] += (byte / 255.0) - 0.5
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            return values
        return [value / norm for value in values]


class BgeEmbeddingProvider:
    """CPU BGE embedding provider loaded lazily on first indexing or query call."""

    def __init__(self, *, model_name: str, device: str, batch_size: int) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    @property
    def dimension(self) -> int:
        return int(self._get_model().get_sentence_embedding_dimension())

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._get_model().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [embedding.astype(float).tolist() for embedding in embeddings]

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model


@lru_cache
def _bge_provider(model_name: str, device: str, batch_size: int) -> BgeEmbeddingProvider:
    return BgeEmbeddingProvider(model_name=model_name, device=device, batch_size=batch_size)


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_backend == "hash":
        return HashEmbeddingProvider()
    if settings.embedding_backend == "bge":
        return _bge_provider(
            settings.embedding_model,
            settings.embedding_device,
            settings.embedding_batch_size,
        )
    raise ValueError(f"unsupported embedding backend: {settings.embedding_backend}")
