import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunk
from .embeddings import EmbeddingProvider
from .vector_store import VectorRecord, VectorStore


class IndexingError(RuntimeError):
    pass


@dataclass(frozen=True)
class IndexingResult:
    chunk_count: int
    fts_count: int
    vector_count: int


class SqliteFtsIndex:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def count_for_version(self, document_version_id: str) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) FROM chunks_fts WHERE document_version_id = ?",
            (document_version_id,),
        ).fetchone()
        return int(row[0])

    def search_active(self, *, knowledge_base_id: str, query: str, limit: int = 10) -> list[str]:
        rows = self.connection.execute(
            """
            SELECT chunks_fts.text
            FROM chunks_fts
            JOIN chunks ON chunks.id = chunks_fts.chunk_id
            JOIN document_versions ON document_versions.id = chunks.document_version_id
            JOIN documents ON documents.active_version_id = document_versions.id
            WHERE documents.knowledge_base_id = ?
                AND documents.visibility_state = 'visible'
                AND chunks_fts MATCH ?
            LIMIT ?
            """,
            (knowledge_base_id, query, limit),
        ).fetchall()
        return [row["text"] for row in rows]


class DocumentIndexer:
    def __init__(
        self,
        *,
        connection: sqlite3.Connection,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        collection_name: str = "document_chunks",
        embedding_batch_size: int = 8,
        index_version: str = "v1",
    ) -> None:
        self.connection = connection
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.embedding_batch_size = embedding_batch_size
        self.index_version_name = index_version

    def index_document_version(
        self,
        *,
        document_version_id: str,
        chunks: tuple[Chunk, ...],
    ) -> IndexingResult:
        version = self._get_version(document_version_id)
        document_id = version["document_id"]
        document = self._get_document(document_id)
        chunk_ids = [str(uuid.uuid4()) for _ in chunks]
        canonical_path = str(document["canonical_path"])
        file_name = Path(canonical_path).name
        indexed_texts = tuple(
            _indexed_chunk_text(canonical_path=canonical_path, text=chunk.text) for chunk in chunks
        )

        self._set_version_state(document_version_id, "indexing")
        try:
            self.connection.execute("BEGIN")
            self._delete_version_records(document_version_id)
            self._insert_chunks(document_version_id, chunk_ids, chunks, indexed_texts)
            self.connection.commit()

            vector_count = self._embed_and_store_vectors(
                knowledge_base_id=document["knowledge_base_id"],
                document_id=document_id,
                document_version_id=document_version_id,
                chunk_ids=chunk_ids,
                chunks=chunks,
                indexed_texts=indexed_texts,
                file_name=file_name,
                canonical_path=canonical_path,
            )

            self.connection.execute("BEGIN")
            self._insert_index_records(chunk_ids)
            chunk_count = self._count_chunks(document_version_id)
            fts_count = SqliteFtsIndex(self.connection).count_for_version(document_version_id)
            counts_match = (
                chunk_count == len(chunks)
                and fts_count == len(chunks)
                and vector_count == len(chunks)
            )
            if not counts_match:
                raise IndexingError("chunk, FTS and vector counts do not match")
            self.connection.execute(
                """
                UPDATE document_versions
                SET state = 'ready', updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (document_version_id,),
            )
            self.connection.execute(
                """
                UPDATE documents
                SET active_version_id = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (document_version_id, document_id),
            )
            self.connection.commit()
            return IndexingResult(
                chunk_count=chunk_count,
                fts_count=fts_count,
                vector_count=vector_count,
            )
        except Exception as exc:
            if self.connection.in_transaction:
                self.connection.rollback()
            self._mark_failed(document_version_id, str(exc))
            raise

    def _get_version(self, document_version_id: str) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM document_versions WHERE id = ?",
            (document_version_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"document version not found: {document_version_id}")
        return row

    def _get_document(self, document_id: str) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"document not found: {document_id}")
        return row

    def _set_version_state(self, document_version_id: str, state: str) -> None:
        self.connection.execute(
            """
            UPDATE document_versions
            SET state = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (state, document_version_id),
        )
        self.connection.commit()

    def _delete_version_records(self, document_version_id: str) -> None:
        chunk_rows = self.connection.execute(
            "SELECT id FROM chunks WHERE document_version_id = ?",
            (document_version_id,),
        ).fetchall()
        chunk_ids = [row["id"] for row in chunk_rows]
        if chunk_ids:
            self.vector_store.delete_by_filter(
                filter_conditions={"document_version_id": document_version_id}
            )
            self.connection.executemany(
                "DELETE FROM chunks_fts WHERE chunk_id = ?",
                [(chunk_id,) for chunk_id in chunk_ids],
            )
            self.connection.executemany(
                "DELETE FROM index_records WHERE chunk_id = ?",
                [(chunk_id,) for chunk_id in chunk_ids],
            )
        self.connection.execute(
            "DELETE FROM chunks WHERE document_version_id = ?",
            (document_version_id,),
        )

    def _insert_chunks(
        self,
        document_version_id: str,
        chunk_ids: list[str],
        chunks: tuple[Chunk, ...],
        indexed_texts: tuple[str, ...],
    ) -> None:
        for index, chunk in enumerate(chunks):
            previous_id = chunk_ids[index - 1] if index > 0 else None
            next_id = chunk_ids[index + 1] if index < len(chunks) - 1 else None
            self.connection.execute(
                """
                INSERT INTO chunks (
                    id, document_version_id, chunk_index, text, page_no, section_path,
                    bbox, token_count, text_hash, previous_chunk_id, next_chunk_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_ids[index],
                    document_version_id,
                    chunk.chunk_index,
                    indexed_texts[index],
                    chunk.page_no,
                    " / ".join(chunk.section_path),
                    json.dumps(chunk.bbox) if chunk.bbox is not None else None,
                    chunk.token_count,
                    chunk.text_hash,
                    previous_id,
                    next_id,
                ),
            )
            self.connection.execute(
                """
                INSERT INTO chunks_fts (chunk_id, document_version_id, text)
                VALUES (?, ?, ?)
                """,
                (chunk_ids[index], document_version_id, indexed_texts[index]),
            )

    def _embed_and_store_vectors(
        self,
        *,
        knowledge_base_id: str,
        document_id: str,
        document_version_id: str,
        chunk_ids: list[str],
        chunks: tuple[Chunk, ...],
        indexed_texts: tuple[str, ...],
        file_name: str,
        canonical_path: str,
    ) -> int:
        total = 0
        for start in range(0, len(chunks), self.embedding_batch_size):
            batch = chunks[start : start + self.embedding_batch_size]
            batch_ids = chunk_ids[start : start + self.embedding_batch_size]
            batch_texts = indexed_texts[start : start + self.embedding_batch_size]
            vectors = self.embedding_provider.embed_texts(list(batch_texts))
            records = [
                VectorRecord(
                    id=batch_ids[index],
                    vector=vector,
                    payload={
                        "chunk_id": batch_ids[index],
                        "document_id": document_id,
                        "knowledge_base_id": knowledge_base_id,
                        "document_version_id": document_version_id,
                        "chunk_index": batch[index].chunk_index,
                        "file_name": file_name,
                        "canonical_path": canonical_path,
                    },
                )
                for index, vector in enumerate(vectors)
            ]
            self.vector_store.upsert(
                collection_name=self.collection_name,
                records=records,
                dimension=self.embedding_provider.dimension,
            )
            total += len(records)
        return total

    def _insert_index_records(self, chunk_ids: list[str]) -> None:
        for chunk_id in chunk_ids:
            self.connection.execute(
                """
                INSERT INTO index_records (id, chunk_id, index_kind, external_id, index_version)
                VALUES (?, ?, ?, ?, ?)
                """,
                (str(uuid.uuid4()), chunk_id, "fts5", chunk_id, self.index_version_name),
            )
            self.connection.execute(
                """
                INSERT INTO index_records (id, chunk_id, index_kind, external_id, index_version)
                VALUES (?, ?, ?, ?, ?)
                """,
                (str(uuid.uuid4()), chunk_id, "qdrant_local", chunk_id, self.index_version_name),
            )

    def _count_chunks(self, document_version_id: str) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) FROM chunks WHERE document_version_id = ?",
            (document_version_id,),
        ).fetchone()
        return int(row[0])

    def _mark_failed(self, document_version_id: str, error: str) -> None:
        self.connection.execute(
            """
            UPDATE document_versions
            SET state = 'failed', error = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (error, document_version_id),
        )
        self.connection.commit()


def _indexed_chunk_text(*, canonical_path: str, text: str) -> str:
    return f"文件名：{Path(canonical_path).name}\n文件路径：{canonical_path}\n{text}"
