"""Deterministic metrics used by the T09 evaluation scripts."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE)


def exact_match(answer: str, expected: str, aliases: list[str] | None = None) -> bool:
    normalized = normalize(answer)
    return normalized in {normalize(value) for value in [expected, *(aliases or [])]}


def token_f1(answer: str, expected: str) -> float:
    answer_tokens = list(normalize(answer))
    expected_tokens = list(normalize(expected))
    if not answer_tokens or not expected_tokens:
        return float(answer_tokens == expected_tokens)
    overlap = sum((Counter(answer_tokens) & Counter(expected_tokens)).values())
    precision = overlap / len(answer_tokens)
    recall = overlap / len(expected_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def numeric_match(answer: str, expected: str) -> dict[str, bool]:
    pattern = re.compile(r"(-?\d+(?:\.\d+)?)\s*([a-zA-Z%/]+)?")
    answer_match = pattern.search(unicodedata.normalize("NFKC", answer))
    expected_match = pattern.search(unicodedata.normalize("NFKC", expected))
    if answer_match is None or expected_match is None:
        return {"value_correct": False, "unit_correct": False, "both_correct": False}
    value_correct = float(answer_match.group(1)) == float(expected_match.group(1))
    unit_correct = (answer_match.group(2) or "").lower() == (expected_match.group(2) or "").lower()
    return {
        "value_correct": value_correct,
        "unit_correct": unit_correct,
        "both_correct": value_correct and unit_correct,
    }


def retrieval_metrics(relevance: list[list[bool]], k: int) -> dict[str, float]:
    hit_rates: list[float] = []
    recalls: list[float] = []
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for labels in relevance:
        top = labels[:k]
        relevant_total = sum(labels)
        hit_rates.append(float(any(top)))
        recalls.append(sum(top) / relevant_total if relevant_total else 0.0)
        precisions.append(sum(top) / k)
        first = next((index + 1 for index, value in enumerate(top) if value), 0)
        reciprocal_ranks.append(1 / first if first else 0.0)
        dcg = sum(value / _log2(index + 2) for index, value in enumerate(top))
        ideal = sum(1 / _log2(index + 2) for index in range(min(relevant_total, k)))
        ndcgs.append(dcg / ideal if ideal else 0.0)
    return {
        "hit_rate": _mean(hit_rates),
        "recall": _mean(recalls),
        "precision": _mean(precisions),
        "mrr": _mean(reciprocal_ranks),
        "ndcg": _mean(ndcgs),
    }


def source_matches(result: dict, expected_sources: list[dict]) -> bool:
    for source in expected_sources:
        if source.get("chunk_id") and source["chunk_id"] == result.get("chunk_id"):
            return True
        if source.get("document_id") and source["document_id"] == result.get("document_id"):
            return True
        expected_path = str(source.get("document_path", "")).replace("\\", "/")
        actual_path = str(result.get("document_path", "")).replace("\\", "/")
        if expected_path and actual_path.endswith(expected_path):
            return True
    return False


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _log2(value: int) -> float:
    from math import log2

    return log2(value)
