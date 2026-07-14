import sqlite3
from pathlib import Path

import pytest

from enterprise_document_rag.config import Settings
from enterprise_document_rag.db import initialize_sqlite, sqlite_connection
from enterprise_document_rag.repositories import (
    DocumentRepository,
    InvalidStateTransitionError,
    JobRepository,
    KnowledgeBaseRepository,
)


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(database_url=str(tmp_path / "agent.db"))


@pytest.fixture()
def initialized_connection(settings: Settings) -> sqlite3.Connection:
    initialize_sqlite(settings)
    with sqlite_connection(settings) as connection:
        yield connection


def test_unique_document_path_constraint_is_enforced(
    initialized_connection: sqlite3.Connection,
) -> None:
    kb = KnowledgeBaseRepository(initialized_connection).create(
        name="contracts",
        embedding_model="BAAI/bge-small-zh-v1.5",
        vector_backend="qdrant_local",
    )
    documents = DocumentRepository(initialized_connection)
    documents.create(knowledge_base_id=kb.id, canonical_path="E:/work2607/a.pdf")

    with pytest.raises(sqlite3.IntegrityError):
        documents.create(knowledge_base_id=kb.id, canonical_path="E:/work2607/a.pdf")


def test_illegal_document_version_transition_is_rejected(
    initialized_connection: sqlite3.Connection,
) -> None:
    kb = KnowledgeBaseRepository(initialized_connection).create(
        name="policies",
        embedding_model="BAAI/bge-small-zh-v1.5",
        vector_backend="qdrant_local",
    )
    documents = DocumentRepository(initialized_connection)
    document = documents.create(knowledge_base_id=kb.id, canonical_path="E:/work2607/policy.pdf")
    version = documents.create_version(
        document_id=document.id,
        sha256="a" * 64,
        size_bytes=12,
        mtime_ns=100,
        parser_version="parser-v1",
    )

    with pytest.raises(InvalidStateTransitionError):
        documents.transition_version(version_id=version.id, new_state="ready")


def test_duplicate_job_event_returns_existing_job_without_duplicate_task(
    initialized_connection: sqlite3.Connection,
) -> None:
    kb = KnowledgeBaseRepository(initialized_connection).create(
        name="engineering",
        embedding_model="BAAI/bge-small-zh-v1.5",
        vector_backend="qdrant_local",
    )
    jobs = JobRepository(initialized_connection)

    first = jobs.enqueue(
        knowledge_base_id=kb.id,
        operation="add",
        path="E:/work2607/spec.pdf",
        expected_sha256="b" * 64,
    )
    second = jobs.enqueue(
        knowledge_base_id=kb.id,
        operation="add",
        path="E:/work2607/spec.pdf",
        expected_sha256="b" * 64,
    )

    assert second.id == first.id
    assert second.job_key == first.job_key
    assert jobs.count() == 1


def test_force_new_job_event_creates_a_new_task(
    initialized_connection: sqlite3.Connection,
) -> None:
    kb = KnowledgeBaseRepository(initialized_connection).create(
        name="reindex-jobs",
        embedding_model="BAAI/bge-small-zh-v1.5",
        vector_backend="qdrant_local",
    )
    jobs = JobRepository(initialized_connection)
    first = jobs.enqueue(
        knowledge_base_id=kb.id,
        operation="reindex",
        path="E:/work2607/manual.docx",
        expected_sha256="a" * 64,
        force_new=True,
    )
    second = jobs.enqueue(
        knowledge_base_id=kb.id,
        operation="reindex",
        path="E:/work2607/manual.docx",
        expected_sha256="a" * 64,
        force_new=True,
    )

    assert first.id != second.id
    assert jobs.count() == 2
