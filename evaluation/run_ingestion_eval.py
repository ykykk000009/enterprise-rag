"""Measure current corpus ingestion and chunk health from production metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enterprise_document_rag.config import Settings
from enterprise_document_rag.db import sqlite_connection


def run(*, knowledge_base_id: str, database: str, output: Path) -> dict:
    settings = Settings(database_url=database)
    with sqlite_connection(settings) as connection:
        statuses = connection.execute(
            """
            SELECT latest.state, COUNT(*) AS count
            FROM documents
            LEFT JOIN document_versions AS latest ON latest.id = (
                SELECT id FROM document_versions WHERE document_id = documents.id
                ORDER BY created_at DESC, id DESC LIMIT 1
            )
            WHERE documents.knowledge_base_id = ? AND documents.visibility_state = 'visible'
            GROUP BY latest.state
            """,
            (knowledge_base_id,),
        ).fetchall()
        chunk_rows = connection.execute(
            """
            SELECT chunks.document_version_id, chunks.token_count, chunks.text, chunks.text_hash,
                chunks.page_no, chunks.section_path
            FROM chunks
            JOIN document_versions ON document_versions.id = chunks.document_version_id
            JOIN documents ON documents.active_version_id = document_versions.id
            WHERE documents.knowledge_base_id = ? AND documents.visibility_state = 'visible'
            """,
            (knowledge_base_id,),
        ).fetchall()
        failed_rows = connection.execute(
            """
            SELECT documents.canonical_path, latest.error
            FROM documents
            JOIN document_versions AS latest ON latest.id = (
                SELECT id FROM document_versions WHERE document_id = documents.id
                ORDER BY created_at DESC, id DESC LIMIT 1
            )
            WHERE documents.knowledge_base_id = ?
                AND documents.visibility_state = 'visible'
                AND latest.state = 'failed'
            ORDER BY documents.canonical_path
            """,
            (knowledge_base_id,),
        ).fetchall()
    counts = {row["state"] or "no_version": row["count"] for row in statuses}
    tokens = sorted(row["token_count"] for row in chunk_rows)
    hashes = [row["text_hash"] for row in chunk_rows]
    document_hashes = [(row["document_version_id"], row["text_hash"]) for row in chunk_rows]
    too_short = sum(token_count < 300 for token_count in tokens)
    too_long = sum(token_count > 800 for token_count in tokens)
    metadata_complete = sum(
        bool(row["page_no"] is not None or row["section_path"]) for row in chunk_rows
    )
    result = {
        "status": "MEASURED",
        "document_status_counts": counts,
        "chunk_count": len(chunk_rows),
        "empty_chunk_rate": _rate(
            sum(not row["text"].strip() for row in chunk_rows), len(chunk_rows)
        ),
        "duplicate_chunk_rate": _rate(len(hashes) - len(set(hashes)), len(hashes)),
        "document_local_duplicate_chunk_rate": _rate(
            len(document_hashes) - len(set(document_hashes)),
            len(document_hashes),
        ),
        "cross_document_duplicate_chunk_rate": _rate(
            (len(hashes) - len(set(hashes))) - (len(document_hashes) - len(set(document_hashes))),
            len(hashes),
        ),
        "chunk_size_target_tokens": {"min": 300, "max": 800},
        "chunk_size_target_rate": _rate(len(tokens) - too_short - too_long, len(tokens)),
        "too_short_chunk_rate": _rate(too_short, len(tokens)),
        "too_long_chunk_rate": _rate(too_long, len(tokens)),
        "source_metadata_completeness": _rate(metadata_complete, len(chunk_rows)),
        "token_distribution": _distribution(tokens),
        "failed_documents": [
            {"document_path": row["canonical_path"], "error": row["error"]} for row in failed_rows
        ],
        "limitations": [
            (
                "Parser success by original file type requires a manually validated "
                "representative file sample."
            ),
            "Table semantic completeness requires human review.",
        ],
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _rate(numerator: int, denominator: int) -> float | str:
    return round(numerator / denominator, 6) if denominator else "NOT_MEASURED"


def _distribution(values: list[int]) -> dict[str, int | str]:
    if not values:
        return {"status": "NOT_MEASURED"}
    return {
        "min": values[0],
        "median": values[len(values) // 2],
        "p90": values[int((len(values) - 1) * 0.9)],
        "p95": values[int((len(values) - 1) * 0.95)],
        "max": values[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge-base-id", required=True)
    parser.add_argument("--database", default="data/agent.db")
    parser.add_argument("--output", default="evaluation/results/ingestion_results.json")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                knowledge_base_id=args.knowledge_base_id,
                database=args.database,
                output=Path(args.output),
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
