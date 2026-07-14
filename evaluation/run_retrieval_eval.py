"""Run ranked retrieval evaluation through the production API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapters import retrieve
from metrics import retrieval_metrics, source_matches


def run(
    *, base_url: str, knowledge_base_id: str, dataset: Path, output: Path, include_unvalidated: bool
) -> dict:
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line]
    official_cases = [case for case in cases if case.get("validated")]
    evaluated_cases = cases if include_unvalidated else official_cases
    rows = []
    official_labels: list[list[bool]] = []
    diagnostic_labels: list[list[bool]] = []
    for case in evaluated_cases:
        try:
            results = retrieve(
                base_url=base_url,
                knowledge_base_id=knowledge_base_id,
                question=case["question"],
                top_k=10,
            )
            relevance = [source_matches(result, case["relevant_sources"]) for result in results]
            rows.append(
                {
                    "id": case["id"],
                    "validated": case["validated"],
                    "question": case["question"],
                    "results": results,
                    "relevance": relevance,
                    "error": None,
                }
            )
            if case["validated"]:
                official_labels.append(relevance)
            else:
                diagnostic_labels.append(relevance)
        except Exception as exc:
            rows.append(
                {
                    "id": case["id"],
                    "validated": case["validated"],
                    "question": case["question"],
                    "results": [],
                    "relevance": [],
                    "error": str(exc),
                }
            )
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if not official_cases:
        result = {
            "status": "NOT_MEASURED",
            "reason": "No manually validated T09 cases are available.",
            "official_case_count": 0,
            "diagnostic_case_count": len(rows),
        }
        if diagnostic_labels:
            result["diagnostic_source_match_metrics"] = {
                f"@{k}": retrieval_metrics(diagnostic_labels, k) for k in (1, 3, 5, 10)
            }
        return result
    return {
        "status": "MEASURED",
        "official_case_count": len(official_cases),
        "metrics": {f"@{k}": retrieval_metrics(official_labels, k) for k in (1, 3, 5, 10)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge-base-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", default="evaluation/dataset.jsonl")
    parser.add_argument("--output", default="evaluation/results/retrieval_results.jsonl")
    parser.add_argument("--summary-output")
    parser.add_argument("--include-unvalidated", action="store_true")
    args = parser.parse_args()
    result = run(
        base_url=args.base_url,
        knowledge_base_id=args.knowledge_base_id,
        dataset=Path(args.dataset),
        output=Path(args.output),
        include_unvalidated=args.include_unvalidated,
    )
    if args.summary_output:
        Path(args.summary_output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
