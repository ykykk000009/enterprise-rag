import json
import sqlite3
import uuid
from collections.abc import Sequence
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


@dataclass(frozen=True)
class BatchIndexDocument:
    document_version_id: str
    chunks: tuple[Chunk, ...]


@dataclass(frozen=True)
class _PreparedIndexDocument:
    item: BatchIndexDocument
    document: sqlite3.Row
    chunk_ids: list[str]
    indexed_texts: tuple[str, ...]


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
        return self.index_document_versions(
            [BatchIndexDocument(document_version_id=document_version_id, chunks=chunks)]
        )[0]

    def index_document_versions(
        self,
        documents: Sequence[BatchIndexDocument],
    ) -> tuple[IndexingResult, ...]:
        """Index several prepared documents while embedding their chunks globally.

        All calls to SQLite and the vector store stay on this writer thread. The
        parser can prepare documents concurrently, then this method amortizes
        embedding overhead across document boundaries.
        """
        if not documents:
            return ()
        prepared: list[_PreparedIndexDocument] = []
        for item in documents:
            if not item.chunks:
                raise ValueError("no indexable text was extracted from the document")
            version = self._get_version(item.document_version_id)
            document_id = version["document_id"]
            document = self._get_document(document_id)
            canonical_path = str(document["canonical_path"])
            chunk_ids = [str(uuid.uuid4()) for _ in item.chunks]
            indexed_texts = tuple(
                _indexed_chunk_text(
                    canonical_path=canonical_path,
                    section_path=chunk.section_path,
                    text=chunk.text,
                )
                for chunk in item.chunks
            )
            prepared.append(
                _PreparedIndexDocument(
                    item=item,
                    document=document,
                    chunk_ids=chunk_ids,
                    indexed_texts=indexed_texts,
                )
            )

        version_ids = [item.item.document_version_id for item in prepared]
        for version_id in version_ids:
            self._set_version_state(version_id, "indexing")
        try:
            for item in prepared:
                self.connection.execute("BEGIN")
                self._delete_version_records(item.item.document_version_id)
                self._insert_chunks(
                    item.item.document_version_id,
                    item.chunk_ids,
                    item.item.chunks,
                    item.indexed_texts,
                )
                self.connection.commit()

            vector_counts = self._embed_and_store_vectors_batch(prepared)
            results: list[IndexingResult] = []
            self.connection.execute("BEGIN")
            for item in prepared:
                version_id = item.item.document_version_id
                expected = len(item.item.chunks)
                self._insert_index_records(item.chunk_ids)
                chunk_count = self._count_chunks(version_id)
                fts_count = SqliteFtsIndex(self.connection).count_for_version(version_id)
                vector_count = vector_counts.get(version_id, 0)
                if (chunk_count, fts_count, vector_count) != (expected, expected, expected):
                    raise IndexingError("chunk, FTS and vector counts do not match")
                self.connection.execute(
                    """
                    UPDATE document_versions
                    SET state = 'ready', updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (version_id,),
                )
                self.connection.execute(
                    """
                    UPDATE documents
                    SET active_version_id = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (version_id, item.document["id"]),
                )
                results.append(
                    IndexingResult(
                        chunk_count=chunk_count,
                        fts_count=fts_count,
                        vector_count=vector_count,
                    )
                )
            self.connection.commit()
            return tuple(results)
        except Exception as exc:
            if self.connection.in_transaction:
                self.connection.rollback()
            for version_id in version_ids:
                self._mark_failed(version_id, str(exc))
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
        chunk_rows = []
        fts_rows = []
        for index, chunk in enumerate(chunks):
            previous_id = chunk_ids[index - 1] if index > 0 else None
            next_id = chunk_ids[index + 1] if index < len(chunks) - 1 else None
            chunk_rows.append(
                (
                    chunk_ids[index],
                    document_version_id,
                    chunk.chunk_index,
                    chunk.text,
                    chunk.page_no,
                    json.dumps(chunk.page_range) if chunk.page_range is not None else None,
                    " / ".join(chunk.section_path),
                    json.dumps(chunk.bbox) if chunk.bbox is not None else None,
                    json.dumps(chunk.bbox_list),
                    chunk.content_type,
                    chunk.source_type,
                    chunk.ocr_confidence,
                    json.dumps(chunk.block_types),
                    chunk.table_markdown,
                    chunk.image_path,
                    chunk.caption,
                    json.dumps(chunk.image_metadata) if chunk.image_metadata is not None else None,
                    chunk.token_count,
                    chunk.text_hash,
                    previous_id,
                    next_id,
                )
            )
            fts_rows.append((chunk_ids[index], document_version_id, indexed_texts[index]))
        self.connection.executemany(
            """
            INSERT INTO chunks (
                id, document_version_id, chunk_index, text, page_no, page_range,
                section_path, bbox, bbox_list, content_type, source_type,
                ocr_confidence, block_types, table_markdown, image_path, caption, image_metadata,
                token_count, text_hash, previous_chunk_id, next_chunk_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            chunk_rows,
        )
        self.connection.executemany(
            """
            INSERT INTO chunks_fts (chunk_id, document_version_id, text)
            VALUES (?, ?, ?)
            """,
            fts_rows,
        )

    def _embed_and_store_vectors_batch(
        self, documents: Sequence["_PreparedIndexDocument"]
    ) -> dict[str, int]:
        rows = [
            (document, index, chunk, document.indexed_texts[index], document.chunk_ids[index])
            for document in documents
            for index, chunk in enumerate(document.item.chunks)
        ]
        counts = {document.item.document_version_id: 0 for document in documents}
        for start in range(0, len(rows), self.embedding_batch_size):
            batch_rows = rows[start : start + self.embedding_batch_size]
            vectors = self.embedding_provider.embed_texts(
                [row[3] for row in batch_rows]
            )
            if len(vectors) != len(batch_rows):
                raise IndexingError("embedding provider returned an unexpected vector count")
            records = []
            for row, vector in zip(batch_rows, vectors, strict=True):
                document, _, chunk, _, chunk_id = row
                records.append(
                    VectorRecord(
                        id=chunk_id,
                        vector=vector,
                        payload={
                            "chunk_id": chunk_id,
                            "document_id": document.document["id"],
                            "knowledge_base_id": document.document["knowledge_base_id"],
                            "document_version_id": document.item.document_version_id,
                            "chunk_index": chunk.chunk_index,
                            "page_range": list(chunk.page_range)
                            if chunk.page_range is not None
                            else None,
                            "section_path": list(chunk.section_path),
                            "bbox_list": [list(item) for item in chunk.bbox_list],
                            "content_type": chunk.content_type,
                            "source_type": chunk.source_type,
                            "ocr_confidence": chunk.ocr_confidence,
                            "table_markdown": chunk.table_markdown,
                            "image_path": chunk.image_path,
                            "caption": chunk.caption,
                            "image_metadata": chunk.image_metadata,
                            "file_name": Path(document.document["canonical_path"]).name,
                            "canonical_path": str(document.document["canonical_path"]),
                        },
                    )
                )
                counts[document.item.document_version_id] += 1
            self.vector_store.upsert(
                collection_name=self.collection_name,
                records=records,
                dimension=self.embedding_provider.dimension,
            )
        return counts

    def _insert_index_records(self, chunk_ids: list[str]) -> None:
        records = [
            (str(uuid.uuid4()), chunk_id, kind, chunk_id, self.index_version_name)
            for chunk_id in chunk_ids
            for kind in ("fts5", "qdrant_local")
        ]
        self.connection.executemany(
            """
            INSERT INTO index_records (id, chunk_id, index_kind, external_id, index_version)
            VALUES (?, ?, ?, ?, ?)
            """,
            records,
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


def _indexed_chunk_text(
    *, canonical_path: str, section_path: tuple[str, ...], text: str
) -> str:
    metadata = [
        f"文件名：{Path(canonical_path).name}",
        f"文件路径：{canonical_path}",
    ]
    if section_path:
        metadata.append(f"位置：{' / '.join(section_path)}")
    return "\n".join([*metadata, text])
