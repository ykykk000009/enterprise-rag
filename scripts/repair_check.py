"""Report SQLite index inconsistencies without changing source documents."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def _count(connection: sqlite3.Connection, query: str) -> int:
    return int(connection.execute(query).fetchone()[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local RAG index consistency.")
    parser.add_argument("--database", default="data/agent.db")
    args = parser.parse_args()

    database = Path(args.database)
    with sqlite3.connect(database) as connection:
        report = {
            "active_chunks": _count(
                connection,
                """
                SELECT COUNT(*) FROM chunks
                JOIN document_versions ON document_versions.id = chunks.document_version_id
                JOIN documents ON documents.active_version_id = document_versions.id
                WHERE documents.visibility_state = 'visible'
                """,
            ),
            "missing_fts_records": _count(
                connection,
                """
                SELECT COUNT(*) FROM chunks
                JOIN document_versions ON document_versions.id = chunks.document_version_id
                JOIN documents ON documents.active_version_id = document_versions.id
                WHERE documents.visibility_state = 'visible'
                  AND NOT EXISTS (
                    SELECT 1 FROM chunks_fts WHERE chunks_fts.chunk_id = chunks.id
                  )
                """,
            ),
            "missing_index_records": _count(
                connection,
                """
                SELECT COUNT(*) FROM chunks
                JOIN document_versions ON document_versions.id = chunks.document_version_id
                JOIN documents ON documents.active_version_id = document_versions.id
                WHERE documents.visibility_state = 'visible'
                  AND NOT EXISTS (
                    SELECT 1 FROM index_records
                    WHERE index_records.chunk_id = chunks.id
                      AND index_records.index_kind = 'qdrant_local'
                  )
                """,
            ),
            "orphan_fts_records": _count(
                connection,
                """
                SELECT COUNT(*) FROM chunks_fts
                LEFT JOIN chunks ON chunks.id = chunks_fts.chunk_id
                WHERE chunks.id IS NULL
                """,
            ),
        }
    report["ok"] = not any(value for key, value in report.items() if key != "active_chunks")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
