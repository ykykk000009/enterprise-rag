"""Provide a controlled one-hour stability runner without hidden background work."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from adapters import retrieve


def run(
    *,
    base_url: str,
    knowledge_base_id: str,
    dataset: Path,
    output: Path,
    duration_seconds: int,
    execute: bool,
) -> dict:
    if not execute:
        result = {
            "status": "NOT_MEASURED",
            "requested_duration_seconds": duration_seconds,
            "executed_duration_seconds": 0,
            "reason": (
                "Pass --execute to run the requested stability soak against the production API."
            ),
        }
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line]
    if not cases:
        raise ValueError("A non-empty dataset is required for a stability run.")
    started = time.monotonic()
    completed = 0
    failures = 0
    latencies: list[float] = []
    while time.monotonic() - started < duration_seconds:
        case = cases[completed % len(cases)]
        query_started = time.perf_counter()
        try:
            retrieve(
                base_url=base_url,
                knowledge_base_id=knowledge_base_id,
                question=case["question"],
                top_k=5,
            )
        except Exception:
            failures += 1
        else:
            latencies.append((time.perf_counter() - query_started) * 1000)
        completed += 1
    elapsed = round(time.monotonic() - started, 2)
    result = {
        "status": "DIAGNOSTIC_UNVALIDATED",
        "requested_duration_seconds": duration_seconds,
        "executed_duration_seconds": elapsed,
        "queries_completed": completed,
        "failure_rate": round(failures / completed, 6) if completed else "NOT_MEASURED",
        "max_latency_ms": round(max(latencies), 2) if latencies else "NOT_MEASURED",
        "note": (
            "Official T09 stability scoring still requires the validated dataset "
            "and resource monitoring."
        ),
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge-base-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", default="evaluation/dataset.jsonl")
    parser.add_argument("--output", default="evaluation/results/stability_results.json")
    parser.add_argument("--duration-seconds", type=int, default=3600)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = run(
        base_url=args.base_url,
        knowledge_base_id=args.knowledge_base_id,
        dataset=Path(args.dataset),
        output=Path(args.output),
        duration_seconds=args.duration_seconds,
        execute=args.execute,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
