import sqlite3
from pathlib import Path

import pytest

from enterprise_document_rag.chunking import StructureAwareChunker
from enterprise_document_rag.config import Settings
from enterprise_document_rag.db import initialize_sqlite, sqlite_connection
from enterprise_document_rag.embeddings import HashEmbeddingProvider
from enterprise_document_rag.indexing import DocumentIndexer
from enterprise_document_rag.parsing import parse_document
from enterprise_document_rag.qa import RAGAnswerer, _answer_from_table_total, _short_quote
from enterprise_document_rag.repositories import DocumentRepository, KnowledgeBaseRepository
from enterprise_document_rag.retrieval import HybridRetriever, SearchResult
from enterprise_document_rag.vector_store import QdrantLocalVectorStore


class StubReranker:
    def score(self, *, query: str, passages: list[str]) -> list[float]:
        del query
        return [10.0 if "preferred passage" in passage else 0.0 for passage in passages]


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
            name="qa-kb",
            embedding_model="hash-cpu",
            vector_backend="qdrant_local",
        )
        .id
    )


@pytest.fixture()
def vector_store() -> QdrantLocalVectorStore:
    return QdrantLocalVectorStore(path=":memory:")


@pytest.fixture()
def embedding_provider() -> HashEmbeddingProvider:
    return HashEmbeddingProvider(dimension=16)


def _index_text_document(
    *,
    tmp_path: Path,
    connection: sqlite3.Connection,
    knowledge_base_id: str,
    embedding_provider: HashEmbeddingProvider,
    vector_store: QdrantLocalVectorStore,
    file_name: str,
    text: str,
):
    file_path = tmp_path / file_name
    file_path.write_text(text, encoding="utf-8")
    documents = DocumentRepository(connection)
    document = documents.create(knowledge_base_id=knowledge_base_id, canonical_path=file_path)
    version = documents.create_version(
        document_id=document.id,
        sha256=file_name.ljust(64, "0")[:64],
        size_bytes=file_path.stat().st_size,
        mtime_ns=file_path.stat().st_mtime_ns,
        parser_version="parser-v1",
        state="parsed",
    )
    chunks = StructureAwareChunker(target_tokens=24, overlap_tokens=2, min_tokens=1).chunk(
        parse_document(file_path)
    )
    DocumentIndexer(
        connection=connection,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    ).index_document_version(document_version_id=version.id, chunks=chunks)
    return document


def test_hybrid_search_returns_valid_authorized_chunks(
    tmp_path: Path,
    initialized_connection: sqlite3.Connection,
    knowledge_base_id: str,
    embedding_provider: HashEmbeddingProvider,
    vector_store: QdrantLocalVectorStore,
) -> None:
    document = _index_text_document(
        tmp_path=tmp_path,
        connection=initialized_connection,
        knowledge_base_id=knowledge_base_id,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        file_name="manual.md",
        text="# Washer\nCleaning pressure is 0.25 MPa.",
    )
    retriever = HybridRetriever(
        connection=initialized_connection,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    results = retriever.search(
        knowledge_base_id=knowledge_base_id,
        query="pressure",
        allowed_document_ids={document.id},
    )

    assert results
    assert results[0].document_id == document.id
    assert "Cleaning pressure" in results[0].quote
    assert "fts" in results[0].sources


def test_reranker_reorders_fused_candidates(
    tmp_path: Path,
    initialized_connection: sqlite3.Connection,
    knowledge_base_id: str,
    embedding_provider: HashEmbeddingProvider,
    vector_store: QdrantLocalVectorStore,
) -> None:
    _index_text_document(
        tmp_path=tmp_path,
        connection=initialized_connection,
        knowledge_base_id=knowledge_base_id,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        file_name="first.md",
        text="# First\nShared field appears in a normal passage.",
    )
    preferred = _index_text_document(
        tmp_path=tmp_path,
        connection=initialized_connection,
        knowledge_base_id=knowledge_base_id,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        file_name="preferred.md",
        text="# Preferred\nShared field appears in the preferred passage.",
    )
    retriever = HybridRetriever(
        connection=initialized_connection,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        vector_top_k=10,
        fts_top_k=10,
        candidate_top_k=10,
        final_top_k=2,
        reranker=StubReranker(),
    )

    results = retriever.search(knowledge_base_id=knowledge_base_id, query="Shared field")

    assert results[0].document_id == preferred.id


def test_quoted_field_is_prioritized_by_lexical_recall(
    tmp_path: Path,
    initialized_connection: sqlite3.Connection,
    knowledge_base_id: str,
    embedding_provider: HashEmbeddingProvider,
    vector_store: QdrantLocalVectorStore,
) -> None:
    target = _index_text_document(
        tmp_path=tmp_path,
        connection=initialized_connection,
        knowledge_base_id=knowledge_base_id,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        file_name="target.md",
        text="# Target\n设备唯一字段ZX-2026-TEST用于定位。",
    )
    _index_text_document(
        tmp_path=tmp_path,
        connection=initialized_connection,
        knowledge_base_id=knowledge_base_id,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        file_name="other.md",
        text="# Other\n设备字段用于一般定位。",
    )
    retriever = HybridRetriever(
        connection=initialized_connection,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    results = retriever.search(
        knowledge_base_id=knowledge_base_id,
        query="请定位包含字段“ZX-2026-TEST”的文件",
    )

    assert results[0].document_id == target.id


def test_exact_quoted_field_overrides_a_generic_reranker_preference(
    tmp_path: Path,
    initialized_connection: sqlite3.Connection,
    knowledge_base_id: str,
    embedding_provider: HashEmbeddingProvider,
    vector_store: QdrantLocalVectorStore,
) -> None:
    target = _index_text_document(
        tmp_path=tmp_path,
        connection=initialized_connection,
        knowledge_base_id=knowledge_base_id,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        file_name="exact.md",
        text="# Exact\n唯一字段ZX-2026-TEST在此文件。",
    )
    _index_text_document(
        tmp_path=tmp_path,
        connection=initialized_connection,
        knowledge_base_id=knowledge_base_id,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        file_name="generic.md",
        text="# Generic\npreferred passage contains a general device description。",
    )
    retriever = HybridRetriever(
        connection=initialized_connection,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        reranker=StubReranker(),
    )

    results = retriever.search(
        knowledge_base_id=knowledge_base_id,
        query="请定位包含字段“ZX-2026-TEST”的文件",
    )

    assert results[0].document_id == target.id


def test_unknown_question_refuses_without_model_knowledge(
    tmp_path: Path,
    initialized_connection: sqlite3.Connection,
    knowledge_base_id: str,
    embedding_provider: HashEmbeddingProvider,
    vector_store: QdrantLocalVectorStore,
) -> None:
    _index_text_document(
        tmp_path=tmp_path,
        connection=initialized_connection,
        knowledge_base_id=knowledge_base_id,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        file_name="scope.md",
        text="# Scope\nThis document only mentions washer installation.",
    )
    retriever = HybridRetriever(
        connection=initialized_connection,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )
    answerer = RAGAnswerer(connection=initialized_connection, retriever=retriever)

    answer = answerer.answer(knowledge_base_id=knowledge_base_id, question="banana warranty")

    assert answer.insufficient_evidence is True
    assert answer.citations == ()
    assert "未找到足够的相关内容" in answer.answer


def test_chinese_question_uses_text_contains_recall(
    tmp_path: Path,
    initialized_connection: sqlite3.Connection,
    knowledge_base_id: str,
    embedding_provider: HashEmbeddingProvider,
    vector_store: QdrantLocalVectorStore,
) -> None:
    _index_text_document(
        tmp_path=tmp_path,
        connection=initialized_connection,
        knowledge_base_id=knowledge_base_id,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        file_name="endoscope.md",
        text="# 规范\n软式内镜清洗消毒前应完成泄漏测试和预处理。",
    )
    retriever = HybridRetriever(
        connection=initialized_connection,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )
    results = retriever.search(
        knowledge_base_id=knowledge_base_id,
        query="软式内镜清洗消毒的关键要求是什么",
    )
    answer = RAGAnswerer(connection=initialized_connection, retriever=retriever).answer(
        knowledge_base_id=knowledge_base_id,
        question="软式内镜清洗消毒的关键要求是什么",
    )

    assert answer.insufficient_evidence is False
    assert answer.citations
    assert "contains" in results[0].sources


def test_acl_filtering_occurs_before_retrieval(
    tmp_path: Path,
    initialized_connection: sqlite3.Connection,
    knowledge_base_id: str,
    embedding_provider: HashEmbeddingProvider,
    vector_store: QdrantLocalVectorStore,
) -> None:
    public_document = _index_text_document(
        tmp_path=tmp_path,
        connection=initialized_connection,
        knowledge_base_id=knowledge_base_id,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        file_name="public.md",
        text="# Public\nPublic washer color is blue.",
    )
    restricted_document = _index_text_document(
        tmp_path=tmp_path,
        connection=initialized_connection,
        knowledge_base_id=knowledge_base_id,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        file_name="restricted.md",
        text="# Restricted\nSecret washer code is ZX-900.",
    )
    retriever = HybridRetriever(
        connection=initialized_connection,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    results = retriever.search(
        knowledge_base_id=knowledge_base_id,
        query="secret",
        allowed_document_ids={public_document.id},
    )
    unrestricted = retriever.search(
        knowledge_base_id=knowledge_base_id,
        query="secret",
        allowed_document_ids={public_document.id, restricted_document.id},
    )

    assert all(result.document_id == public_document.id for result in results)
    assert not any("Secret washer code" in result.quote for result in results)
    assert any(result.document_id == restricted_document.id for result in unrestricted)


def test_answer_contains_valid_source_citation(
    tmp_path: Path,
    initialized_connection: sqlite3.Connection,
    knowledge_base_id: str,
    embedding_provider: HashEmbeddingProvider,
    vector_store: QdrantLocalVectorStore,
) -> None:
    document = _index_text_document(
        tmp_path=tmp_path,
        connection=initialized_connection,
        knowledge_base_id=knowledge_base_id,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        file_name="params.md",
        text="# Parameters\nFlow rate is 20 L/min.",
    )
    retriever = HybridRetriever(
        connection=initialized_connection,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )
    answerer = RAGAnswerer(connection=initialized_connection, retriever=retriever)

    answer = answerer.answer(
        knowledge_base_id=knowledge_base_id,
        question="flow rate",
        allowed_document_ids={document.id},
    )

    assert answer.insufficient_evidence is False
    assert answer.answer.endswith("[1]")
    assert answer.citations
    assert answer.citations[0].file_name == "params.md"
    assert answer.citations[0].document_id == document.id
    assert "Flow rate is 20 L/min" in answer.citations[0].quote


def test_table_total_answer_uses_cached_excel_values() -> None:
    result = SearchResult(
        chunk_id="chunk-total",
        document_id="document-total",
        file_name="quote.xlsx",
        canonical_path="E:/work/quote.xlsx",
        page_no=None,
        section_path="工作表：报价",
        quote=(
            "序号：合计 | 总价：=SUM(I7:I31)（计算值：1410413.95） | "
            "医院总价：=SUM(K7:K31)（计算值：347.16）"
        ),
        bbox=None,
        score=1.0,
    )

    answer = _answer_from_table_total(question="设备总和报价是多少", evidence=[result])

    assert answer == "根据《quote.xlsx》的合计行，总价为 1410413.95，医院总价为 347.16。[1]"


def test_table_total_citation_keeps_the_total_row() -> None:
    total_row = "序号：合计 | 医院总价：=SUM(K7:K31)（计算值：347.16）"
    quote = f"文件名：quote.xlsx\n工作表：报价\n{total_row}"

    assert _short_quote(quote) == total_row
