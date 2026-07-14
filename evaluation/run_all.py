"""Run the reproducible T09 evaluation suite and generate its report."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from run_generation_eval import run as run_generation
from run_incremental_eval import run as run_incremental
from run_ingestion_eval import run as run_ingestion
from run_performance_eval import run as run_performance
from run_retrieval_eval import run as run_retrieval
from run_stability_eval import run as run_stability

ROOT = Path(__file__).resolve().parents[1]


def _build_candidates(*, knowledge_base_id: str, database: str, dataset: Path) -> dict:
    command = [
        sys.executable,
        str(ROOT / "evaluation" / "build_dataset.py"),
        "--knowledge-base-id",
        knowledge_base_id,
        "--database",
        database,
        "--output",
        str(dataset),
        "--limit",
        "120",
    ]
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def _write_report(*, output: Path, summary: dict, retrieval_rows: Path) -> None:
    rows = [
        json.loads(line) for line in retrieval_rows.read_text(encoding="utf-8").splitlines() if line
    ]
    samples = rows[:10]
    duplicate_rate = summary["ingestion"].get("duplicate_chunk_rate", "NOT_MEASURED")
    too_long_rate = summary["ingestion"].get("too_long_chunk_rate", "NOT_MEASURED")
    failed_documents = summary["ingestion"].get("failed_documents", [])
    body = [
        "# T09 RAG MVP Evaluation Report",
        "",
        f"- Evaluation time (UTC): {summary['timestamp_utc']}",
        f"- Knowledge base: `{summary['knowledge_base_id']}`",
        f"- Overall result: **{summary['overall_result']}**",
        "",
        "## Official Result",
        "",
        "The corpus contains no manually validated T09 cases. "
        "Official retrieval, answer, citation, and robustness metrics are `NOT_MEASURED`; "
        "candidate data was not counted as ground truth.",
        "Under the T09 hard-fail criteria, this evaluation is a **FAIL**.",
        "",
        "## Measured Diagnostics",
        "",
        f"- Candidate source-grounded cases: {summary['dataset']['candidate_case_count']}",
        f"- Validated official cases: {summary['dataset']['validated_case_count']}",
        f"- Active chunks: {summary['ingestion'].get('chunk_count', 'NOT_MEASURED')}",
        f"- Empty chunk rate: {summary['ingestion'].get('empty_chunk_rate', 'NOT_MEASURED')}",
        f"- Duplicate chunk rate: {duplicate_rate}",
        f"- Chunks over 800 tokens: {too_long_rate}",
        f"- Retrieval diagnostic status: {summary['retrieval']['status']}",
        f"- Retrieval-only performance status: {summary['performance']['status']}",
        "",
        "## Required Remediation",
        "",
        "1. Manually validate at least 100 dataset cases against original documents, "
        "including paths, evidence spans, answers, and citation expectations.",
        "2. Re-run retrieval and generation evaluation with only `validated: true` cases.",
        "3. Run the isolated add/update/move/delete corpus evaluation "
        "and the one-hour stability soak.",
        "",
        "## Ingestion Failures",
        "",
    ]
    if failed_documents:
        for item in failed_documents:
            body.extend(
                [
                    f"- `{item['document_path']}`",
                    f"  - Error: `{item.get('error') or 'not reported'}`",
                ]
            )
    else:
        body.append("- No document is currently in the failed state.")
    body.extend(
        [
            "",
            "## Unvalidated Candidate Samples",
            "",
            "These are source-derived samples for review, not failed cases "
            "and not official metric evidence.",
            "",
        ]
    )
    for row in samples:
        first = row["results"][0] if row.get("results") else {}
        body.extend(
            [
                f"- `{row['id']}`: `{row['question'][:100]}`",
                f"  - First result: `{first.get('document_path', 'NO_RESULT')}`",
                f"  - Error: `{row.get('error') or 'none'}`",
            ]
        )
    output.write_text("\n".join(body) + "\n", encoding="utf-8")


def run(*, knowledge_base_id: str, database: str, base_url: str, results_dir: Path) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset = ROOT / "evaluation" / "dataset.jsonl"
    if not dataset.exists():
        _build_candidates(knowledge_base_id=knowledge_base_id, database=database, dataset=dataset)
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line]
    ingestion = run_ingestion(
        knowledge_base_id=knowledge_base_id,
        database=database,
        output=results_dir / "ingestion_results.json",
    )
    retrieval = run_retrieval(
        base_url=base_url,
        knowledge_base_id=knowledge_base_id,
        dataset=dataset,
        output=results_dir / "retrieval_results.jsonl",
        include_unvalidated=True,
    )
    generation = run_generation(
        base_url=base_url,
        knowledge_base_id=knowledge_base_id,
        dataset=dataset,
        output=results_dir / "generation_results.jsonl",
    )
    performance = run_performance(
        base_url=base_url,
        knowledge_base_id=knowledge_base_id,
        dataset=dataset,
        output=results_dir / "performance_results.json",
        count=100,
    )
    incremental = run_incremental(output=results_dir / "incremental_results.json")
    stability = run_stability(
        base_url=base_url,
        knowledge_base_id=knowledge_base_id,
        dataset=dataset,
        output=results_dir / "stability_results.json",
        duration_seconds=3600,
        execute=False,
    )
    validated = [case for case in cases if case.get("validated")]
    hard_fail_conditions = [
        "No manually validated dataset cases are available for official T09 scoring.",
        "Official retrieval, generation, citation, robustness, incremental, "
        "and stability gates are not measured.",
    ]
    if (
        ingestion["too_long_chunk_rate"] != "NOT_MEASURED"
        and ingestion["too_long_chunk_rate"] > 0.1
    ):
        hard_fail_conditions.append("More than 10% of chunks exceed the T09 maximum of 800 tokens.")
    if ingestion["failed_documents"]:
        hard_fail_conditions.append(
            "At least one visible document remains in the failed ingestion state."
        )
    summary = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "knowledge_base_id": knowledge_base_id,
        "overall_result": "FAIL",
        "hard_fail_conditions": hard_fail_conditions,
        "dataset": {
            "candidate_case_count": len(cases),
            "validated_case_count": len(validated),
            "candidate_status": "UNVALIDATED_NOT_COUNTED",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor() or "NOT_REPORTED",
            "git_commit": "NOT_AVAILABLE",
            "base_url": base_url,
            "database": database,
        },
        "ingestion": ingestion,
        "retrieval": retrieval,
        "generation": generation,
        "performance": performance,
        "incremental": incremental,
        "stability": stability,
    }
    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(
        output=ROOT / "evaluation" / "report.md",
        summary=summary,
        retrieval_rows=results_dir / "retrieval_results.jsonl",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge-base-id", required=True)
    parser.add_argument("--database", default="data/agent.db")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--results-dir", default="evaluation/results")
    args = parser.parse_args()
    summary = run(
        knowledge_base_id=args.knowledge_base_id,
        database=args.database,
        base_url=args.base_url,
        results_dir=Path(args.results_dir),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
