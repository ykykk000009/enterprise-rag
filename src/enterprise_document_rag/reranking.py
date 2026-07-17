"""Local reranker providers used after hybrid candidate fusion."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
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


class OnnxBgeReranker:
    """INT8 ONNX BGE reranker for the compact offline distribution."""

    def __init__(self, *, model_path: str, batch_size: int, max_length: int) -> None:
        self.model_path = Path(model_path)
        self.batch_size = batch_size
        self.max_length = max_length
        self._session = None
        self._tokenizer = None

    def score(self, *, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        session, tokenizer = self._get_runtime()
        input_names = {item.name for item in session.get_inputs()}
        scores: list[float] = []
        for offset in range(0, len(passages), self.batch_size):
            batch = passages[offset : offset + self.batch_size]
            encoded = tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            feed = {
                name: encoded[name]
                for name in ("input_ids", "attention_mask", "token_type_ids")
                if name in input_names and name in encoded
            }
            logits = session.run(None, feed)[0]
            scores.extend(float(value) for value in logits.reshape(-1))
        return scores

    def _get_runtime(self):
        if self._session is None or self._tokenizer is None:
            import onnxruntime
            from transformers import AutoTokenizer

            model_file = self.model_path / "model.int8.onnx"
            if not model_file.is_file():
                raise RuntimeError(f"INT8 reranker model does not exist: {model_file}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
            self._session = onnxruntime.InferenceSession(
                str(model_file),
                providers=["CPUExecutionProvider"],
            )
        return self._session, self._tokenizer


@lru_cache
def _bge_reranker(model_name: str, device: str, batch_size: int) -> BgeReranker:
    return BgeReranker(model_name=model_name, device=device, batch_size=batch_size)


@lru_cache
def _onnx_bge_reranker(
    model_path: str,
    batch_size: int,
    max_length: int,
) -> OnnxBgeReranker:
    return OnnxBgeReranker(
        model_path=model_path,
        batch_size=batch_size,
        max_length=max_length,
    )


def build_reranker(settings: Settings) -> Reranker | None:
    if not settings.reranker_enabled:
        return None
    if settings.reranker_backend == "bge":
        return _bge_reranker(
            settings.reranker_model,
            settings.reranker_device,
            settings.reranker_batch_size,
        )
    if settings.reranker_backend == "onnx":
        return _onnx_bge_reranker(
            settings.reranker_model,
            settings.reranker_batch_size,
            settings.reranker_max_length,
        )
    raise ValueError(f"unsupported reranker backend: {settings.reranker_backend}")
