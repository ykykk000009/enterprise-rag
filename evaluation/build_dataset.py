"""Build unvalidated, source-grounded candidate cases for human review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enterprise_document_rag.config import Settings
from enterprise_document_rag.db import sqlite_connection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge-base-id", required=True)
    parser.add_argument("--database", default="data/agent.db")
    parser.add_argument("--output", default="evaluation/dataset.jsonl")
    parser.add_argument("--limit", type=int, default=120)
    args = parser.parse_args()
    settings = Settings(database_url=args.database)
    with sqlite_connection(settings) as connection:
        rows = connection.execute(
            """
            SELECT chunks.id AS chunk_id, chunks.text, chunks.page_no, chunks.section_path,
                documents.id AS document_id, documents.canonical_path
            FROM chunks
            JOIN document_versions ON document_versions.id = chunks.document_version_id
            JOIN documents ON documents.active_version_id = document_versions.id
            WHERE documents.knowledge_base_id = ? AND documents.visibility_state = 'visible'
            ORDER BY documents.canonical_path, chunks.chunk_index
            LIMIT ?
            """,
            (args.knowledge_base_id, args.limit),
        ).fetchall()
    output = Path(args.output)
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            evidence = " ".join(row["text"].split())[:180]
            case = {
                "id": f"candidate-{index:04d}",
                "question": f"Locate the source containing this excerpt: {evidence[:60]}",
                "category": "candidate_source_location",
                "expected_answer": evidence,
                "answer_aliases": [],
                "relevant_sources": [
                    {
                        "document_path": row["canonical_path"],
                        "document_id": row["document_id"],
                        "chunk_id": row["chunk_id"],
                        "page": row["page_no"],
                        "section": row["section_path"],
                        "evidence_text": evidence,
                    }
                ],
                "difficulty": "unknown",
                "requires_multiple_sources": False,
                "validated": False,
                "notes": (
                    "Auto-generated candidate. Human validation is required "
                    "before T09 official scoring."
                ),
            }
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(json.dumps({"candidate_cases": len(rows), "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
