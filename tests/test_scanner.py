import sqlite3
from pathlib import Path

import pytest

from enterprise_document_rag.config import Settings
from enterprise_document_rag.db import initialize_sqlite, sqlite_connection
from enterprise_document_rag.repositories import (
    DocumentRepository,
    JobRepository,
    KnowledgeBaseRepository,
)
from enterprise_document_rag.scanner import FileEventDebouncer, SourceScanner
from enterprise_document_rag.security import PathAuthorizationError, authorize_path


@pytest.fixture()
def initialized_connection(tmp_path: Path) -> sqlite3.Connection:
    settings = Settings(database_url=str(tmp_path / "agent.db"))
    initialize_sqlite(settings)
    with sqlite_connection(settings) as connection:
        yield connection


@pytest.fixture()
def knowledge_base_id(initialized_connection: sqlite3.Connection) -> str:
    return KnowledgeBaseRepository(initialized_connection).create(
        name="scan-kb",
        embedding_model="BAAI/bge-small-zh-v1.5",
        vector_backend="qdrant_local",
    ).id


@pytest.fixture()
def scanner(initialized_connection: sqlite3.Connection) -> SourceScanner:
    return SourceScanner(
        documents=DocumentRepository(initialized_connection),
        jobs=JobRepository(initialized_connection),
    )


def test_path_escape_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    escaped = outside / "secret.pdf"
    escaped.write_text("secret", encoding="utf-8")

    with pytest.raises(PathAuthorizationError):
        authorize_path(escaped, root=root)


def test_new_update_move_delete_are_detected(
    tmp_path: Path,
    initialized_connection: sqlite3.Connection,
    knowledge_base_id: str,
    scanner: SourceScanner,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    first_file = source / "a.txt"
    first_file.write_text("alpha", encoding="utf-8")

    add_result = scanner.reconcile(knowledge_base_id=knowledge_base_id, root_path=source)
    assert add_result.counts["add"] == 1

    docs = DocumentRepository(initialized_connection)
    document = docs.get_by_path(
        knowledge_base_id=knowledge_base_id,
        canonical_path=first_file.resolve(),
    )
    assert document is not None
    version = docs.get_latest_version(document.id)
    assert version is not None
    docs.transition_version(version_id=version.id, new_state="parsed")
    docs.transition_version(version_id=version.id, new_state="indexing")
    docs.transition_version(version_id=version.id, new_state="ready")

    first_file.write_text("alpha updated", encoding="utf-8")
    update_result = scanner.reconcile(knowledge_base_id=knowledge_base_id, root_path=source)
    assert update_result.counts["update"] == 1

    new_version = docs.get_active_version(document.id)
    assert new_version is not None
    pending = initialized_connection.execute(
        """
        SELECT id FROM document_versions
        WHERE document_id = ? AND state = 'pending'
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (document.id,),
    ).fetchone()
    assert pending is not None
    docs.transition_version(version_id=pending["id"], new_state="parsed")
    docs.transition_version(version_id=pending["id"], new_state="indexing")
    docs.transition_version(version_id=pending["id"], new_state="ready")

    moved_file = source / "renamed.txt"
    first_file.rename(moved_file)
    move_result = scanner.reconcile(knowledge_base_id=knowledge_base_id, root_path=source)
    assert move_result.counts["move"] == 1

    moved_file.unlink()
    delete_result = scanner.reconcile(knowledge_base_id=knowledge_base_id, root_path=source)
    assert delete_result.counts["delete"] == 1

    jobs = JobRepository(initialized_connection)
    assert jobs.count() == 4


def test_unchanged_file_is_skipped(
    tmp_path: Path,
    initialized_connection: sqlite3.Connection,
    knowledge_base_id: str,
    scanner: SourceScanner,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    file_path = source / "stable.md"
    file_path.write_text("# Stable\nSame content", encoding="utf-8")

    scanner.reconcile(knowledge_base_id=knowledge_base_id, root_path=source)
    docs = DocumentRepository(initialized_connection)
    document = docs.get_by_path(
        knowledge_base_id=knowledge_base_id,
        canonical_path=file_path.resolve(),
    )
    assert document is not None
    version = docs.get_latest_version(document.id)
    assert version is not None
    docs.transition_version(version_id=version.id, new_state="parsed")
    docs.transition_version(version_id=version.id, new_state="indexing")
    docs.transition_version(version_id=version.id, new_state="ready")

    result = scanner.reconcile(knowledge_base_id=knowledge_base_id, root_path=source)

    assert result.counts["unchanged"] == 1
    assert result.events == ()
    assert JobRepository(initialized_connection).count() == 1


def test_file_event_debouncer_requires_two_identical_snapshots(tmp_path: Path) -> None:
    file_path = tmp_path / "event.txt"
    file_path.write_text("loading", encoding="utf-8")
    debouncer = FileEventDebouncer()

    assert debouncer.is_stable(file_path) is False
    assert debouncer.is_stable(file_path) is True


def test_scanner_skips_office_temporary_files(
    initialized_connection: sqlite3.Connection,
    knowledge_base_id: str,
    scanner: SourceScanner,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "~$draft.docx").write_text("not a DOCX package", encoding="utf-8")
    (source / "normal.md").write_text("# Normal\ncontent", encoding="utf-8")

    result = scanner.reconcile(knowledge_base_id=knowledge_base_id, root_path=source)

    assert result.counts["add"] == 1
    assert result.counts["skipped_unsupported"] == 1
