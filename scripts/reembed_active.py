"""Populate a new vector collection from active SQLite chunks without changing FTS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from enterprise_document_rag.config import Settings
from enterprise_document_rag.db import sqlite_connection
from enterprise_document_rag.embeddings import build_embedding_provider
from enterprise_document_rag.vector_store import QdrantLocalVectorStore, VectorRecord


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-embed active chunks into a new collection.")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--database", default="data/agent.db")
    parser.add_argument("--qdrant", default="data/qdrant")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    settings = Settings(
        database_url=args.database,
        qdrant_path=Path(args.qdrant),
        embedding_backend="bge",
        vector_collection_name=args.collection,
        embedding_batch_size=args.batch_size,
    )
    provider = build_embedding_provider(settings)
    store = QdrantLocalVectorStore(path=str(settings.qdrant_path))
    with sqlite_connection(settings) as connection:
        rows = connection.execute(
            """
            SELECT chunks.id, chunks.text, chunks.chunk_index, document_versions.id AS version_id,
                documents.id AS document_id, documents.knowledge_base_id
            FROM chunks
            JOIN document_versions ON document_versions.id = chunks.document_version_id
            JOIN documents ON documents.active_version_id = document_versions.id
            WHERE documents.visibility_state = 'visible'
            ORDER BY documents.id, chunks.chunk_index
            """
        ).fetchall()
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            vectors = provider.embed_texts([row["text"] for row in batch])
            store.upsert(
                collection_name=args.collection,
                dimension=provider.dimension,
                records=[
                    VectorRecord(
                        id=row["id"],
                        vector=vector,
                        payload={
                            "chunk_id": row["id"],
                            "document_id": row["document_id"],
                            "knowledge_base_id": row["knowledge_base_id"],
                            "document_version_id": row["version_id"],
                            "chunk_index": row["chunk_index"],
                        },
                    )
                    for row, vector in zip(batch, vectors, strict=True)
                ],
            )
    print(json.dumps({"collection": args.collection, "vectors": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
