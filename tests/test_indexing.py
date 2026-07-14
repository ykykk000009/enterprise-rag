import sqlite3
from pathlib import Path

import pytest

from enterprise_document_rag.chunking import StructureAwareChunker
from enterprise_document_rag.config import Settings
from enterprise_document_rag.db import initialize_sqlite, sqlite_connection
from enterprise_document_rag.embeddings import HashEmbeddingProvider
from enterprise_document_rag.indexing import DocumentIndexer, SqliteFtsIndex
from enterprise_document_rag.parsing import parse_document
from enterprise_document_rag.repositories import (
    DocumentRepository,
    JobRepository,
    KnowledgeBaseRepository,
)
from enterprise_document_rag.scanner import SourceScanner
from enterprise_document_rag.vector_store import QdrantLocalVectorStore, VectorRecord


class FailingVectorStore:
    def upsert(self, *, collection_name: str, records: list[VectorRecord], dimension: int) -> None:
        raise RuntimeError("vector backend unavailable")

    def count(self, *, collection_name: str) -> int:
        return 0


@pytest.fixture()
def initialized_connection(tmp_path: Path) -> sqlite3.Connection:
    settings = Settings(database_url=str(tmp_path / "agent.db"))
    initialize_sqlite(settings)
    with sqlite_connection(settings) as connection:
        yield connection


@pytest.fixture()
def knowledge_base_id(initialized_connection: sqlite3.Connection) -> str:
    return (
        KnowledgeBaseRepository(initialized_connection)
        .create(
            name="index-kb",
            embedding_model="hash-cpu",
            vector_backend="qdrant_local",
        )
        .id
    )


def _chunks_for_text_file(path: Path):
    parsed = parse_document(path)
    return StructureAwareChunker(target_tokens=20, overlap_tokens=2, min_tokens=1).chunk(parsed)


def test_vector_and_fts_counts_match_chunks(
    tmp_path: Path,
    initialized_connection: sqlite3.Connection,
    knowledge_base_id: str,
) -> None:
    file_path = tmp_path / "source.txt"
    file_path.write_text("alpha beta gamma\n\nindexed contract clause", encoding="utf-8")
    documents = DocumentRepository(initialized_connection)
    document = documents.create(knowledge_base_id=knowledge_base_id, canonical_path=file_path)
    version = documents.create_version(
        document_id=document.id,
        sha256="1" * 64,
        size_bytes=file_path.stat().st_size,
        mtime_ns=file_path.stat().st_mtime_ns,
        parser_version="parser-v1",
        state="parsed",
    )
    vector_store = QdrantLocalVectorStore(path=":memory:")
    indexer = DocumentIndexer(
        connection=initialized_connection,
        embedding_provider=HashEmbeddingProvider(dimension=16),
        vector_store=vector_store,
    )

    chunks = _chunks_for_text_file(file_path)
    result = indexer.index_document_version(document_version_id=version.id, chunks=chunks)

    assert result.chunk_count == len(chunks)
    assert result.fts_count == len(chunks)
    assert result.vector_count == len(chunks)
    assert vector_store.count(collection_name="document_chunks") == len(chunks)
    stored_text = initialized_connection.execute(
        "SELECT text FROM chunks WHERE document_version_id = ?",
        (version.id,),
    ).fetchone()["text"]
    assert stored_text.startswith(f"文件名：source.txt\n文件路径：{file_path}\n")
    assert SqliteFtsIndex(initialized_connection).search_active(
        knowledge_base_id=knowledge_base_id,
        query="contract",
    )
    points, _ = vector_store.client.scroll(
        collection_name="document_chunks", limit=1, with_payload=True, with_vectors=False
    )
    assert points[0].payload["file_name"] == "source.txt"
    assert points[0].payload["canonical_path"] == str(file_path)


def test_reindex_replaces_existing_vectors_for_a_document_version(
    tmp_path: Path,
    initialized_connection: sqlite3.Connection,
    knowledge_base_id: str,
) -> None:
    file_path = tmp_path / "source.txt"
    file_path.write_text("old indexed content", encoding="utf-8")
    documents = DocumentRepository(initialized_connection)
    document = documents.create(knowledge_base_id=knowledge_base_id, canonical_path=file_path)
    version = documents.create_version(
        document_id=document.id,
        sha256="9" * 64,
        size_bytes=file_path.stat().st_size,
        mtime_ns=file_path.stat().st_mtime_ns,
        parser_version="parser-v1",
        state="parsed",
    )
    vector_store = QdrantLocalVectorStore(path=":memory:")
    indexer = DocumentIndexer(
        connection=initialized_connection,
        embedding_provider=HashEmbeddingProvider(dimension=16),
        vector_store=vector_store,
    )
    indexer.index_document_version(
        document_version_id=version.id,
        chunks=_chunks_for_text_file(file_path),
    )

    file_path.write_text("new indexed content", encoding="utf-8")
    replacement_chunks = _chunks_for_text_file(file_path)
    result = indexer.index_document_version(
        document_version_id=version.id,
        chunks=replacement_chunks,
    )

    stored_text = initialized_connection.execute(
        "SELECT text FROM chunks WHERE document_version_id = ?",
        (version.id,),
    ).fetchone()["text"]
    assert stored_text == f"文件名：source.txt\n文件路径：{file_path}\nnew indexed content"
    assert result.chunk_count == len(replacement_chunks)
    assert vector_store.count(collection_name="document_chunks") == len(replacement_chunks)


def test_failed_update_leaves_old_version_queryable(
    tmp_path: Path,
    initialized_connection: sqlite3.Connection,
    knowledge_base_id: str,
) -> None:
    file_path = tmp_path / "policy.txt"
    file_path.write_text("old searchable policy", encoding="utf-8")
    documents = DocumentRepository(initialized_connection)
    document = documents.create(knowledge_base_id=knowledge_base_id, canonical_path=file_path)
    old_version = documents.create_version(
        document_id=document.id,
        sha256="2" * 64,
        size_bytes=file_path.stat().st_size,
        mtime_ns=file_path.stat().st_mtime_ns,
        parser_version="parser-v1",
        state="parsed",
    )
    good_indexer = DocumentIndexer(
        connection=initialized_connection,
        embedding_provider=HashEmbeddingProvider(dimension=16),
        vector_store=QdrantLocalVectorStore(path=":memory:"),
    )
    good_indexer.index_document_version(
        document_version_id=old_version.id,
        chunks=_chunks_for_text_file(file_path),
    )

    file_path.write_text("new uncommitted policy", encoding="utf-8")
    new_version = documents.create_version(
        document_id=document.id,
        sha256="3" * 64,
        size_bytes=file_path.stat().st_size,
        mtime_ns=file_path.stat().st_mtime_ns,
        parser_version="parser-v1",
        state="parsed",
    )
    failing_indexer = DocumentIndexer(
        connection=initialized_connection,
        embedding_provider=HashEmbeddingProvider(dimension=16),
        vector_store=FailingVectorStore(),
    )

    with pytest.raises(RuntimeError):
        failing_indexer.index_document_version(
            document_version_id=new_version.id,
            chunks=_chunks_for_text_file(file_path),
        )

    active = documents.get_active_version(document.id)
    assert active is not None
    assert active.id == old_version.id
    assert SqliteFtsIndex(initialized_connection).search_active(
        knowledge_base_id=knowledge_base_id,
        query="old",
    )
    assert not SqliteFtsIndex(initialized_connection).search_active(
        knowledge_base_id=knowledge_base_id,
        query="uncommitted",
    )


def test_rename_with_same_hash_avoids_reembedding(
    tmp_path: Path,
    initialized_connection: sqlite3.Connection,
    knowledge_base_id: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    file_path = source / "machine.md"
    file_path.write_text("# Machine\nsame hash document", encoding="utf-8")
    scanner = SourceScanner(
        documents=DocumentRepository(initialized_connection),
        jobs=JobRepository(initialized_connection),
    )
    scanner.reconcile(knowledge_base_id=knowledge_base_id, root_path=source)

    documents = DocumentRepository(initialized_connection)
    document = documents.get_by_path(
        knowledge_base_id=knowledge_base_id,
        canonical_path=file_path.resolve(),
    )
    assert document is not None
    version = documents.get_latest_version(document.id)
    assert version is not None
    embedding_provider = HashEmbeddingProvider(dimension=16)
    DocumentIndexer(
        connection=initialized_connection,
        embedding_provider=embedding_provider,
        vector_store=QdrantLocalVectorStore(path=":memory:"),
    ).index_document_version(
        document_version_id=version.id,
        chunks=_chunks_for_text_file(file_path),
    )
    assert embedding_provider.embedded_text_count > 0
    embedded_count = embedding_provider.embedded_text_count

    renamed = source / "machine-renamed.md"
    file_path.rename(renamed)
    result = scanner.reconcile(knowledge_base_id=knowledge_base_id, root_path=source)

    assert result.counts["move"] == 1
    assert embedding_provider.embedded_text_count == embedded_count
