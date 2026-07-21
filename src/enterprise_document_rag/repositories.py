import hashlib
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DOCUMENT_VERSION_TRANSITIONS = {
    "pending": {"parsed", "indexing", "failed"},
    "parsed": {"indexing", "failed"},
    "indexing": {"ready", "failed"},
    "ready": {"deleted"},
    "failed": {"pending", "deleted"},
    "deleted": set(),
}

JOB_TRANSITIONS = {
    "queued": {"leased", "succeeded", "failed"},
    "leased": {"queued", "succeeded", "failed"},
    "succeeded": set(),
    "failed": {"queued"},
}


class InvalidStateTransitionError(ValueError):
    pass


@dataclass(frozen=True)
class KnowledgeBase:
    id: str
    name: str
    embedding_model: str
    vector_backend: str


@dataclass(frozen=True)
class Document:
    id: str
    knowledge_base_id: str
    canonical_path: str
    active_version_id: str | None
    visibility_state: str


@dataclass(frozen=True)
class DocumentVersion:
    id: str
    document_id: str
    sha256: str
    size_bytes: int
    mtime_ns: int
    parser_version: str
    layout_version: str | None
    document_type: str | None
    state: str


@dataclass(frozen=True)
class Source:
    id: str
    knowledge_base_id: str
    root_path: str
    include_patterns: tuple[str, ...]
    exclude_patterns: tuple[str, ...]
    scan_state: str


@dataclass(frozen=True)
class Job:
    id: str
    job_key: str
    knowledge_base_id: str
    operation: str
    path: str
    expected_sha256: str | None
    state: str
    attempts: int
    error: str | None


def _new_id() -> str:
    return str(uuid.uuid4())


def _row_to_knowledge_base(row: sqlite3.Row) -> KnowledgeBase:
    return KnowledgeBase(
        id=row["id"],
        name=row["name"],
        embedding_model=row["embedding_model"],
        vector_backend=row["vector_backend"],
    )


def _row_to_document(row: sqlite3.Row) -> Document:
    return Document(
        id=row["id"],
        knowledge_base_id=row["knowledge_base_id"],
        canonical_path=row["canonical_path"],
        active_version_id=row["active_version_id"],
        visibility_state=row["visibility_state"],
    )


def _row_to_document_version(row: sqlite3.Row) -> DocumentVersion:
    return DocumentVersion(
        id=row["id"],
        document_id=row["document_id"],
        sha256=row["sha256"],
        size_bytes=row["size_bytes"],
        mtime_ns=row["mtime_ns"],
        parser_version=row["parser_version"],
        layout_version=row["layout_version"],
        document_type=row["document_type"],
        state=row["state"],
    )


def _row_to_source(row: sqlite3.Row) -> Source:
    return Source(
        id=row["id"],
        knowledge_base_id=row["knowledge_base_id"],
        root_path=row["root_path"],
        include_patterns=tuple(json.loads(row["include_patterns"])),
        exclude_patterns=tuple(json.loads(row["exclude_patterns"])),
        scan_state=row["scan_state"],
    )


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        job_key=row["job_key"],
        knowledge_base_id=row["knowledge_base_id"],
        operation=row["operation"],
        path=row["path"],
        expected_sha256=row["expected_sha256"],
        state=row["state"],
        attempts=row["attempts"],
        error=row["error"],
    )


class KnowledgeBaseRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def create(
        self,
        *,
        name: str,
        embedding_model: str,
        vector_backend: str,
    ) -> KnowledgeBase:
        knowledge_base_id = _new_id()
        self.connection.execute(
            """
            INSERT INTO knowledge_bases (id, name, embedding_model, vector_backend)
            VALUES (?, ?, ?, ?)
            """,
            (knowledge_base_id, name, embedding_model, vector_backend),
        )
        self.connection.commit()
        return self.get(knowledge_base_id)

    def get(self, knowledge_base_id: str) -> KnowledgeBase:
        row = self.connection.execute(
            "SELECT * FROM knowledge_bases WHERE id = ?",
            (knowledge_base_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"knowledge base not found: {knowledge_base_id}")
        return _row_to_knowledge_base(row)

    def list(self) -> list[KnowledgeBase]:
        rows = self.connection.execute(
            "SELECT * FROM knowledge_bases ORDER BY created_at, name"
        ).fetchall()
        return [_row_to_knowledge_base(row) for row in rows]

    def delete(self, knowledge_base_id: str) -> KnowledgeBase:
        """Remove one knowledge base and its relational records.

        SQLite's FTS virtual table does not participate in foreign-key cascades, so
        its rows are removed explicitly before the knowledge-base cascade.
        """
        knowledge_base = self.get(knowledge_base_id)
        self.connection.execute(
            """
            DELETE FROM chunks_fts
            WHERE document_version_id IN (
                SELECT document_versions.id
                FROM document_versions
                JOIN documents ON documents.id = document_versions.document_id
                WHERE documents.knowledge_base_id = ?
            )
            """,
            (knowledge_base_id,),
        )
        self.connection.execute("DELETE FROM knowledge_bases WHERE id = ?", (knowledge_base_id,))
        self.connection.commit()
        return knowledge_base


class SourceRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def create(
        self,
        *,
        knowledge_base_id: str,
        root_path: str | Path,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> Source:
        source_id = _new_id()
        normalized_root = str(root_path)
        self.connection.execute(
            """
            INSERT INTO sources (
                id, knowledge_base_id, root_path, include_patterns, exclude_patterns
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                source_id,
                knowledge_base_id,
                normalized_root,
                json.dumps(include_patterns or [], sort_keys=True),
                json.dumps(exclude_patterns or [], sort_keys=True),
            ),
        )
        self.connection.commit()
        return self.get(source_id)

    def get(self, source_id: str) -> Source:
        row = self.connection.execute(
            "SELECT * FROM sources WHERE id = ?",
            (source_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"source not found: {source_id}")
        return _row_to_source(row)

    def list(self, *, knowledge_base_id: str | None = None) -> list[Source]:
        if knowledge_base_id is None:
            rows = self.connection.execute("SELECT * FROM sources ORDER BY created_at").fetchall()
        else:
            rows = self.connection.execute(
                "SELECT * FROM sources WHERE knowledge_base_id = ? ORDER BY created_at",
                (knowledge_base_id,),
            ).fetchall()
        return [_row_to_source(row) for row in rows]

    def set_scan_state(self, *, source_id: str, scan_state: str) -> Source:
        self.connection.execute(
            "UPDATE sources SET scan_state = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (scan_state, source_id),
        )
        self.connection.commit()
        return self.get(source_id)


class DocumentRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def create(self, *, knowledge_base_id: str, canonical_path: str | Path) -> Document:
        document_id = _new_id()
        self.connection.execute(
            """
            INSERT INTO documents (id, knowledge_base_id, canonical_path)
            VALUES (?, ?, ?)
            """,
            (document_id, knowledge_base_id, str(canonical_path)),
        )
        self.connection.commit()
        return self.get(document_id)

    def get(self, document_id: str) -> Document:
        row = self.connection.execute(
            "SELECT * FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"document not found: {document_id}")
        return _row_to_document(row)

    def get_by_path(self, *, knowledge_base_id: str, canonical_path: str | Path) -> Document | None:
        row = self.connection.execute(
            """
            SELECT * FROM documents
            WHERE knowledge_base_id = ? AND canonical_path = ?
            """,
            (knowledge_base_id, str(canonical_path)),
        ).fetchone()
        if row is None:
            return None
        return _row_to_document(row)

    def get_active_version(self, document_id: str) -> DocumentVersion | None:
        row = self.connection.execute(
            """
            SELECT document_versions.*
            FROM documents
            JOIN document_versions ON document_versions.id = documents.active_version_id
            WHERE documents.id = ?
            """,
            (document_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_document_version(row)

    def find_by_active_sha256(
        self,
        *,
        knowledge_base_id: str,
        sha256: str,
    ) -> Document | None:
        row = self.connection.execute(
            """
            SELECT documents.*
            FROM documents
            JOIN document_versions ON document_versions.id = documents.active_version_id
            WHERE documents.knowledge_base_id = ?
                AND documents.visibility_state = 'visible'
                AND document_versions.sha256 = ?
            ORDER BY documents.updated_at DESC
            LIMIT 1
            """,
            (knowledge_base_id, sha256),
        ).fetchone()
        if row is None:
            return None
        return _row_to_document(row)

    def list_visible_for_root(
        self,
        *,
        knowledge_base_id: str,
        root_path: str | Path,
    ) -> list[Document]:
        root = str(root_path)
        rows = self.connection.execute(
            """
            SELECT *
            FROM documents
            WHERE knowledge_base_id = ?
                AND visibility_state = 'visible'
                AND (canonical_path = ? OR canonical_path LIKE ?)
            ORDER BY canonical_path
            """,
            (knowledge_base_id, root, f"{root}{os.sep}%"),
        ).fetchall()
        return [_row_to_document(row) for row in rows]

    def create_version(
        self,
        *,
        document_id: str,
        sha256: str,
        size_bytes: int,
        mtime_ns: int,
        parser_version: str,
        state: str = "pending",
    ) -> DocumentVersion:
        if state not in DOCUMENT_VERSION_TRANSITIONS:
            raise ValueError(f"unknown document version state: {state}")
        version_id = _new_id()
        self.connection.execute(
            """
            INSERT INTO document_versions (
                id, document_id, size_bytes, mtime_ns, sha256, parser_version, state
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (version_id, document_id, size_bytes, mtime_ns, sha256, parser_version, state),
        )
        self.connection.commit()
        return self.get_version(version_id)

    def update_path(self, *, document_id: str, canonical_path: str | Path) -> Document:
        self.connection.execute(
            """
            UPDATE documents
            SET canonical_path = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (str(canonical_path), document_id),
        )
        self.connection.commit()
        return self.get(document_id)

    def mark_deleted(self, *, document_id: str) -> Document:
        self.connection.execute(
            """
            UPDATE documents
            SET visibility_state = 'deleted', updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (document_id,),
        )
        self.connection.commit()
        return self.get(document_id)

    def list_statuses(self, *, knowledge_base_id: str) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT
                documents.id,
                documents.canonical_path,
                documents.visibility_state,
                documents.updated_at,
                latest_versions.id AS version_id,
                latest_versions.state AS version_state,
                latest_versions.error AS version_error,
                COUNT(chunks.id) AS chunk_count
            FROM documents
            LEFT JOIN document_versions AS active_versions
                ON active_versions.id = documents.active_version_id
            LEFT JOIN document_versions AS latest_versions
                ON latest_versions.id = (
                    SELECT id FROM document_versions
                    WHERE document_id = documents.id
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                )
            LEFT JOIN chunks ON chunks.document_version_id = active_versions.id
            WHERE documents.knowledge_base_id = ?
            GROUP BY documents.id, active_versions.id, latest_versions.id
            ORDER BY documents.updated_at DESC, documents.canonical_path
            """,
            (knowledge_base_id,),
        ).fetchall()

    def status_summary(self, *, knowledge_base_id: str) -> sqlite3.Row:
        return self.connection.execute(
            """
            SELECT
                COUNT(*) AS total_files,
                SUM(CASE WHEN latest_versions.state = 'ready' THEN 1 ELSE 0 END) AS completed_files,
                SUM(CASE WHEN latest_versions.state = 'failed' THEN 1 ELSE 0 END) AS failed_files,
                SUM(
                    CASE WHEN latest_versions.state IN ('pending', 'parsed', 'indexing')
                    THEN 1 ELSE 0 END
                ) AS processing_files
            FROM documents
            LEFT JOIN document_versions AS latest_versions
                ON latest_versions.id = (
                    SELECT id FROM document_versions
                    WHERE document_id = documents.id
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                )
            WHERE documents.knowledge_base_id = ?
                AND documents.visibility_state = 'visible'
            """,
            (knowledge_base_id,),
        ).fetchone()

    def list_failed(self, *, knowledge_base_id: str) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT
                documents.id AS document_id,
                documents.canonical_path,
                latest_versions.error AS error,
                latest_versions.updated_at AS updated_at
            FROM documents
            JOIN document_versions AS latest_versions
                ON latest_versions.id = (
                    SELECT id FROM document_versions
                    WHERE document_id = documents.id
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                )
            WHERE documents.knowledge_base_id = ?
                AND documents.visibility_state = 'visible'
                AND latest_versions.state = 'failed'
            ORDER BY latest_versions.updated_at DESC, documents.canonical_path
            """,
            (knowledge_base_id,),
        ).fetchall()

    def update_version_fingerprint(
        self,
        *,
        version_id: str,
        size_bytes: int,
        mtime_ns: int,
    ) -> DocumentVersion:
        self.connection.execute(
            """
            UPDATE document_versions
            SET size_bytes = ?, mtime_ns = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (size_bytes, mtime_ns, version_id),
        )
        self.connection.commit()
        return self.get_version(version_id)

    def get_version(self, version_id: str) -> DocumentVersion:
        row = self.connection.execute(
            "SELECT * FROM document_versions WHERE id = ?",
            (version_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"document version not found: {version_id}")
        return _row_to_document_version(row)

    def get_latest_version(self, document_id: str) -> DocumentVersion | None:
        row = self.connection.execute(
            """
            SELECT *
            FROM document_versions
            WHERE document_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (document_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_document_version(row)

    def transition_version(self, *, version_id: str, new_state: str) -> DocumentVersion:
        current = self.get_version(version_id)
        allowed = DOCUMENT_VERSION_TRANSITIONS[current.state]
        if new_state not in allowed:
            raise InvalidStateTransitionError(
                f"cannot transition document version from {current.state} to {new_state}"
            )
        self.connection.execute(
            """
            UPDATE document_versions
            SET state = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (new_state, version_id),
        )
        if new_state == "ready":
            self.connection.execute(
                """
                UPDATE documents
                SET active_version_id = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (version_id, current.document_id),
            )
        self.connection.commit()
        return self.get_version(version_id)


class JobRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def enqueue(
        self,
        *,
        knowledge_base_id: str,
        operation: str,
        path: str | Path,
        expected_sha256: str | None = None,
        payload: dict[str, Any] | None = None,
        force_new: bool = False,
    ) -> Job:
        normalized_path = str(path)
        job_key = self.build_job_key(
            knowledge_base_id=knowledge_base_id,
            operation=operation,
            path=normalized_path,
            expected_sha256=expected_sha256,
        )
        if force_new:
            job_key = hashlib.sha256(f"{job_key}\x1f{uuid.uuid4()}".encode()).hexdigest()
        self.connection.execute(
            """
            INSERT INTO jobs (
                id, job_key, knowledge_base_id, operation, path, expected_sha256, payload
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_key) DO UPDATE SET
                state = CASE WHEN jobs.state = 'failed' THEN 'queued' ELSE jobs.state END,
                error = CASE WHEN jobs.state = 'failed' THEN NULL ELSE jobs.error END,
                updated_at = CASE
                    WHEN jobs.state = 'failed' THEN CURRENT_TIMESTAMP ELSE jobs.updated_at
                END
            """,
            (
                _new_id(),
                job_key,
                knowledge_base_id,
                operation,
                normalized_path,
                expected_sha256,
                json.dumps(payload or {}, sort_keys=True),
            ),
        )
        self.connection.commit()
        return self.get_by_key(job_key)

    def get_by_key(self, job_key: str) -> Job:
        row = self.connection.execute(
            "SELECT * FROM jobs WHERE job_key = ?",
            (job_key,),
        ).fetchone()
        if row is None:
            raise KeyError(f"job not found: {job_key}")
        return _row_to_job(row)

    def get(self, job_id: str) -> Job:
        row = self.connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"job not found: {job_id}")
        return _row_to_job(row)

    def list_runnable(self, *, knowledge_base_id: str | None = None) -> list[Job]:
        sql = "SELECT * FROM jobs WHERE state = 'queued'"
        parameters: tuple[str, ...] = ()
        if knowledge_base_id is not None:
            sql += " AND knowledge_base_id = ?"
            parameters = (knowledge_base_id,)
        sql += " ORDER BY created_at, id"
        rows = self.connection.execute(sql, parameters).fetchall()
        return [_row_to_job(row) for row in rows]

    def release_leases(self) -> int:
        cursor = self.connection.execute(
            """
            UPDATE jobs
            SET state = 'queued', lease_owner = NULL, lease_expires_at = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE state = 'leased'
            """
        )
        self.connection.commit()
        return int(cursor.rowcount)

    def lease(self, *, job_id: str, lease_owner: str) -> Job:
        current = self.get(job_id)
        if current.state != "queued":
            raise InvalidStateTransitionError(f"cannot lease job in state {current.state}")
        self.connection.execute(
            """
            UPDATE jobs
            SET state = 'leased', attempts = attempts + 1, lease_owner = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (lease_owner, job_id),
        )
        self.connection.commit()
        return self.get(job_id)

    def succeed(self, *, job_id: str) -> Job:
        current = self.get(job_id)
        if current.state not in {"queued", "leased"}:
            raise InvalidStateTransitionError(f"cannot complete job in state {current.state}")
        self.connection.execute(
            """
            UPDATE jobs
            SET state = 'succeeded', error = NULL, lease_owner = NULL,
                lease_expires_at = NULL, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (job_id,),
        )
        self.connection.commit()
        return self.get(job_id)

    def fail(self, *, job_id: str, error: str) -> Job:
        current = self.get(job_id)
        if current.state not in {"queued", "leased"}:
            raise InvalidStateTransitionError(f"cannot fail job in state {current.state}")
        self.connection.execute(
            """
            UPDATE jobs
            SET state = 'failed', error = ?, lease_owner = NULL,
                lease_expires_at = NULL, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (error, job_id),
        )
        self.connection.commit()
        return self.get(job_id)

    def count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM jobs").fetchone()
        return int(row[0])

    def transition(self, *, job_id: str, new_state: str) -> Job:
        row = self.connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"job not found: {job_id}")
        current = _row_to_job(row)
        allowed = JOB_TRANSITIONS[current.state]
        if new_state not in allowed:
            raise InvalidStateTransitionError(
                f"cannot transition job from {current.state} to {new_state}"
            )
        self.connection.execute(
            "UPDATE jobs SET state = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (new_state, job_id),
        )
        self.connection.commit()
        return self.get_by_key(current.job_key)

    @staticmethod
    def build_job_key(
        *,
        knowledge_base_id: str,
        operation: str,
        path: str,
        expected_sha256: str | None,
    ) -> str:
        material = "\x1f".join([knowledge_base_id, operation.upper(), path, expected_sha256 or ""])
        return hashlib.sha256(material.encode("utf-8")).hexdigest()
