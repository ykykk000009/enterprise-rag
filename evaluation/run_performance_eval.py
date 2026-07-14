"""Measure retrieval-only latency against the running production API."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from adapters import retrieve


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 2)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 2)


def _latency_summary(values: list[float]) -> dict[str, float | str]:
    if not values:
        return {"status": "NOT_MEASURED"}
    return {
        "min": round(min(values), 2),
        "mean": round(statistics.mean(values), 2),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": round(max(values), 2),
    }


def run(
    *,
    base_url: str,
    knowledge_base_id: str,
    dataset: Path,
    output: Path,
    count: int,
) -> dict:
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line]
    cases = cases[:count]
    latencies: list[float] = []
    errors: list[dict[str, str]] = []
    if cases:
        try:
            retrieve(
                base_url=base_url,
                knowledge_base_id=knowledge_base_id,
                question=cases[0]["question"],
                top_k=5,
            )
        except Exception as exc:
            errors.append({"id": "warmup", "error": str(exc)})
    for case in cases:
        started = time.perf_counter()
        try:
            retrieve(
                base_url=base_url,
                knowledge_base_id=knowledge_base_id,
                question=case["question"],
                top_k=5,
            )
            latencies.append((time.perf_counter() - started) * 1000)
        except Exception as exc:
            errors.append({"id": case["id"], "error": str(exc)})
    official_count = sum(bool(case.get("validated")) for case in cases)
    result = {
        "status": "MEASURED" if official_count else "DIAGNOSTIC_UNVALIDATED",
        "case_count": len(cases),
        "official_case_count": official_count,
        "retrieval_only_ms": _latency_summary(latencies),
        "failure_rate": round(len(errors) / (len(cases) + bool(cases)), 6)
        if cases
        else "NOT_MEASURED",
        "errors": errors,
        "full_rag": {
            "status": "NOT_MEASURED",
            "reason": "Full RAG latency requires validated QA cases and a separate controlled run.",
        },
        "resource_usage": {
            "status": "NOT_MEASURED",
            "reason": (
                "The production Uvicorn process was not instrumented for process resource sampling."
            ),
        },
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge-base-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", default="evaluation/dataset.jsonl")
    parser.add_argument("--output", default="evaluation/results/performance_results.json")
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    result = run(
        base_url=args.base_url,
        knowledge_base_id=args.knowledge_base_id,
        dataset=Path(args.dataset),
        output=Path(args.output),
        count=args.count,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
