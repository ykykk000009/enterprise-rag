"""Single-process orchestration for source scans and persistent ingestion jobs."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .chunking import StructureAwareChunker
from .config import Settings
from .embeddings import build_embedding_provider
from .indexing import DocumentIndexer
from .parsing import parse_document
from .repositories import DocumentRepository, Job, JobRepository, SourceRepository
from .scanner import ScanResult, SourceScanner
from .vector_store import QdrantLocalVectorStore, get_local_vector_store


@dataclass(frozen=True)
class ScanRun:
    result: ScanResult
    jobs: tuple[Job, ...]


class IngestionService:
    """Runs one persisted ingestion job at a time in the FastAPI process."""

    def __init__(self, *, connection: sqlite3.Connection, settings: Settings) -> None:
        self.connection = connection
        self.settings = settings
        self.documents = DocumentRepository(connection)
        self.jobs = JobRepository(connection)
        self.sources = SourceRepository(connection)
        self.scanner = SourceScanner(documents=self.documents, jobs=self.jobs)
        self.embedding_provider = build_embedding_provider(settings)
        self._vector_store: QdrantLocalVectorStore | None = None

    def scan_source(self, *, source_id: str) -> ScanRun:
        source = self.sources.get(source_id)
        self.sources.set_scan_state(source_id=source_id, scan_state="scanning")
        try:
            result = self.scanner.reconcile(
                knowledge_base_id=source.knowledge_base_id,
                root_path=source.root_path,
                include_patterns=list(source.include_patterns),
                exclude_patterns=list(source.exclude_patterns),
            )
            jobs = tuple(
                self.jobs.get_by_key(
                    self.jobs.build_job_key(
                        knowledge_base_id=source.knowledge_base_id,
                        operation=event.operation,
                        path=event.path,
                        expected_sha256=event.sha256,
                    )
                )
                for event in result.events
            )
        except Exception:
            self.sources.set_scan_state(source_id=source_id, scan_state="failed")
            raise
        self.sources.set_scan_state(source_id=source_id, scan_state="idle")
        return ScanRun(result=result, jobs=jobs)

    def resume_pending(self) -> list[Job]:
        self.jobs.release_leases()
        return self.process_pending()

    def process_pending(self, *, knowledge_base_id: str | None = None) -> list[Job]:
        completed: list[Job] = []
        for job in self.jobs.list_runnable(knowledge_base_id=knowledge_base_id):
            completed.append(self.process_job(job_id=job.id))
        return completed

    def process_job(self, *, job_id: str) -> Job:
        job = self.jobs.lease(job_id=job_id, lease_owner="fastapi")
        try:
            if job.operation.lower() in {"add", "update"}:
                self._index_job(job)
            elif job.operation.lower() == "reindex":
                self._reindex_job(job)
            elif job.operation.lower() not in {"move", "delete"}:
                raise ValueError(f"unsupported ingestion operation: {job.operation}")
        except Exception as exc:
            self._mark_pending_version_failed(job=job, error=_safe_error(exc))
            return self.jobs.fail(job_id=job.id, error=_safe_error(exc))
        return self.jobs.succeed(job_id=job.id)

    def enqueue_reindex(self, *, document_id: str) -> Job:
        document = self.documents.get(document_id)
        if document.visibility_state != "visible":
            raise ValueError("deleted documents cannot be reindexed")
        version = self.documents.get_active_version(document_id) or (
            self.documents.get_latest_version(document_id)
        )
        if version is None:
            raise ValueError("document has no indexed or pending version to reindex")
        if version.state == "failed":
            version = self.documents.transition_version(version_id=version.id, new_state="pending")
        return self.jobs.enqueue(
            knowledge_base_id=document.knowledge_base_id,
            operation="reindex" if document.active_version_id else "add",
            path=document.canonical_path,
            expected_sha256=version.sha256,
            force_new=bool(document.active_version_id),
        )

    def retry_failed_documents(self, *, knowledge_base_id: str) -> list[Job]:
        rows = self.connection.execute(
            """
            SELECT documents.id
            FROM documents
            JOIN document_versions ON document_versions.id = (
                SELECT id FROM document_versions
                WHERE document_id = documents.id
                ORDER BY created_at DESC, id DESC
                LIMIT 1
            )
            WHERE documents.knowledge_base_id = ?
                AND documents.visibility_state = 'visible'
                AND document_versions.state = 'failed'
            """,
            (knowledge_base_id,),
        ).fetchall()
        return [self.enqueue_reindex(document_id=row["id"]) for row in rows]

    def _index_job(self, job: Job) -> None:
        path = Path(job.path)
        if not path.is_file():
            raise FileNotFoundError("source file is no longer available")
        document = self.documents.get_by_path(
            knowledge_base_id=job.knowledge_base_id,
            canonical_path=path.resolve(),
        )
        if document is None:
            raise ValueError("document metadata is missing")
        version = self.documents.get_latest_version(document.id)
        if version is None:
            raise ValueError("document version is missing")
        if job.expected_sha256 is not None and version.sha256 != job.expected_sha256:
            raise ValueError("stale job fingerprint does not match the latest document version")

        self._index_version(path=path, version_id=version.id)

    def _index_version(self, *, path: Path, version_id: str) -> None:

        parsed = parse_document(
            path,
            ocr_enabled=self.settings.ocr_enabled,
            ocr_min_text_chars_per_page=self.settings.ocr_min_text_chars_per_page,
            ocr_render_dpi=self.settings.ocr_render_dpi,
            archive_max_members=self.settings.archive_max_members,
            archive_max_member_bytes=self.settings.archive_max_member_bytes,
            archive_max_uncompressed_bytes=self.settings.archive_max_uncompressed_bytes,
            archive_max_compression_ratio=self.settings.archive_max_compression_ratio,
        )
        chunks = StructureAwareChunker(
            target_tokens=self.settings.chunk_size_tokens,
            overlap_tokens=self.settings.chunk_overlap_tokens,
            min_tokens=self.settings.chunk_min_tokens,
            max_tokens=self.settings.chunk_max_tokens,
        ).chunk(parsed)
        if not chunks:
            raise ValueError("no indexable text was extracted from the document")
        DocumentIndexer(
            connection=self.connection,
            embedding_provider=self.embedding_provider,
            vector_store=self._get_vector_store(),
            collection_name=self.settings.vector_collection_name,
            embedding_batch_size=self.settings.embedding_batch_size,
        ).index_document_version(document_version_id=version_id, chunks=chunks)

    def close(self) -> None:
        self._vector_store = None

    def _get_vector_store(self) -> QdrantLocalVectorStore:
        if self._vector_store is None:
            self._vector_store = get_local_vector_store(path=str(self.settings.qdrant_path))
        return self._vector_store

    def _reindex_job(self, job: Job) -> None:
        document = self.documents.get_by_path(
            knowledge_base_id=job.knowledge_base_id,
            canonical_path=Path(job.path).resolve(),
        )
        if document is None:
            raise ValueError("document metadata is missing")
        version = self.documents.get_active_version(document.id)
        if version is None:
            self._index_job(job)
            return
        if job.expected_sha256 is not None and version.sha256 != job.expected_sha256:
            raise ValueError("stale job fingerprint does not match the active version")
        self._index_version(
            path=Path(job.path),
            version_id=version.id,
        )

    def _mark_pending_version_failed(self, *, job: Job, error: str) -> None:
        if job.operation.lower() not in {"add", "update"}:
            return
        row = self.connection.execute(
            """
            SELECT document_versions.id
            FROM documents
            JOIN document_versions ON document_versions.document_id = documents.id
            WHERE documents.knowledge_base_id = ?
                AND documents.canonical_path = ?
                AND document_versions.sha256 = ?
                AND document_versions.state IN ('pending', 'parsed', 'indexing')
            ORDER BY document_versions.created_at DESC
            LIMIT 1
            """,
            (job.knowledge_base_id, job.path, job.expected_sha256),
        ).fetchone()
        if row is None:
            return
        self.connection.execute(
            """
            UPDATE document_versions
            SET state = 'failed', error = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (error, row["id"]),
        )
        self.connection.commit()


def _safe_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return message[:500]
