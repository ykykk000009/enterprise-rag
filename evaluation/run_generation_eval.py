"""Run official generation/citation evaluation only for validated cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapters import answer
from metrics import exact_match, token_f1


def run(*, base_url: str, knowledge_base_id: str, dataset: Path, output: Path) -> dict:
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line]
    cases = [case for case in cases if case.get("validated")]
    rows = []
    for case in cases:
        result = answer(
            base_url=base_url, knowledge_base_id=knowledge_base_id, question=case["question"]
        )
        rows.append(
            {
                "id": case["id"],
                "result": result,
                "exact_match": exact_match(
                    result["answer"], case["expected_answer"], case.get("answer_aliases")
                ),
                "token_f1": token_f1(result["answer"], case["expected_answer"]),
            }
        )
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if not cases:
        return {
            "status": "NOT_MEASURED",
            "reason": "No manually validated T09 cases are available.",
        }
    return {"status": "MEASURED", "case_count": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge-base-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", default="evaluation/dataset.jsonl")
    parser.add_argument("--output", default="evaluation/results/generation_results.jsonl")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                base_url=args.base_url,
                knowledge_base_id=args.knowledge_base_id,
                dataset=Path(args.dataset),
                output=Path(args.output),
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
