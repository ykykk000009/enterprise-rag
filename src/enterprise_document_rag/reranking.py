"""Local reranker providers used after hybrid candidate fusion."""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from .config import Settings


class Reranker(Protocol):
    def score(self, *, query: str, passages: list[str]) -> list[float]: ...


class BgeReranker:
    def __init__(self, *, model_name: str, device: str, batch_size: int) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def score(self, *, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self._get_model().predict(
            [[query, passage] for passage in passages],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model


@lru_cache
def _bge_reranker(model_name: str, device: str, batch_size: int) -> BgeReranker:
    return BgeReranker(model_name=model_name, device=device, batch_size=batch_size)


def build_reranker(settings: Settings) -> Reranker | None:
    if not settings.reranker_enabled:
        return None
    if settings.reranker_backend == "bge":
        return _bge_reranker(
            settings.reranker_model,
            settings.reranker_device,
            settings.reranker_batch_size,
        )
    raise ValueError(f"unsupported reranker backend: {settings.reranker_backend}")
