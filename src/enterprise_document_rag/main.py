from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .config import Settings, configure_huggingface_cache, get_settings
from .db import initialize_sqlite, sqlite_connection, sqlite_health
from .embeddings import build_embedding_provider
from .operations import IngestionService
from .preview import render_document_preview
from .qa import RAGAnswerer, build_llm_provider
from .repositories import (
    DocumentRepository,
    Job,
    JobRepository,
    KnowledgeBase,
    KnowledgeBaseRepository,
    Source,
    SourceRepository,
)
from .reranking import build_reranker
from .retrieval import HybridRetriever
from .security import PathAuthorizationError, resolve_authorized_root
from .text_utils import clean_display_text
from .vector_store import (
    close_local_vector_store,
    get_local_vector_store,
)
from .worker import IngestionWorker

STATIC_DIR = Path(__file__).with_name("static")


class KnowledgeBaseRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class SourceRequest(BaseModel):
    knowledge_base_id: str
    root_path: str = Field(min_length=1)
    include_patterns: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    knowledge_base_id: str
    query: str = Field(min_length=1)
    allowed_document_ids: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)


class QueryRequest(BaseModel):
    knowledge_base_id: str
    question: str = Field(min_length=1)
    allowed_document_ids: list[str] | None = None


class FieldSearchRequest(BaseModel):
    knowledge_base_id: str
    fields: list[str] = Field(min_length=1, max_length=50)
    mode: Literal["exact", "hybrid"] = "exact"


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    configure_huggingface_cache(app_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        initialize_sqlite(app_settings)
        with sqlite_connection(app_settings) as connection:
            JobRepository(connection).release_leases()
        worker = IngestionWorker(settings=app_settings)
        app.state.ingestion_worker = worker
        worker.start()
        try:
            yield
        finally:
            worker.stop()
            close_local_vector_store(path=str(app_settings.qdrant_path))

    app = FastAPI(
        title="Enterprise Document Local RAG Agent",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = app_settings

    @app.get("/")
    def home() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready() -> dict[str, bool | str]:
        initialize_sqlite(app_settings)
        return sqlite_health(app_settings)

    @app.post("/api/v1/knowledge-bases")
    def create_knowledge_base(request: KnowledgeBaseRequest) -> dict[str, str]:
        with sqlite_connection(app_settings) as connection:
            try:
                knowledge_base = KnowledgeBaseRepository(connection).create(
                    name=request.name.strip(),
                    embedding_model=app_settings.embedding_model,
                    vector_backend=app_settings.vector_backend,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=409,
                    detail="knowledge base name already exists",
                ) from exc
        return _knowledge_base_payload(knowledge_base)

    @app.get("/api/v1/knowledge-bases")
    def list_knowledge_bases() -> list[dict[str, str]]:
        with sqlite_connection(app_settings) as connection:
            return [
                _knowledge_base_payload(item) for item in KnowledgeBaseRepository(connection).list()
            ]

    @app.delete("/api/v1/knowledge-bases/{knowledge_base_id}")
    def delete_knowledge_base(knowledge_base_id: str) -> dict[str, object]:
        vector_store = get_local_vector_store(path=str(app_settings.qdrant_path))
        with vector_store.exclusive():
            with sqlite_connection(app_settings) as connection:
                knowledge_bases = KnowledgeBaseRepository(connection)
                try:
                    knowledge_base = knowledge_bases.get(knowledge_base_id)
                except KeyError as exc:
                    raise HTTPException(status_code=404, detail="knowledge base not found") from exc
                knowledge_bases.delete(knowledge_base_id)
            deleted_vectors = vector_store.delete_by_filter(
                filter_conditions={"knowledge_base_id": knowledge_base_id}
            )
        return {
            "id": knowledge_base.id,
            "name": knowledge_base.name,
            "deleted_vectors": deleted_vectors,
        }

    @app.post("/api/v1/sources")
    def create_source(request: SourceRequest) -> dict[str, object]:
        try:
            root = resolve_authorized_root(request.root_path)
        except PathAuthorizationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        with sqlite_connection(app_settings) as connection:
            knowledge_bases = KnowledgeBaseRepository(connection)
            try:
                knowledge_bases.get(request.knowledge_base_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="knowledge base not found") from exc
            try:
                source = SourceRepository(connection).create(
                    knowledge_base_id=request.knowledge_base_id,
                    root_path=root,
                    include_patterns=request.include_patterns,
                    exclude_patterns=request.exclude_patterns,
                )
            except Exception as exc:
                raise HTTPException(status_code=409, detail="source already exists") from exc
        return _source_payload(source)

    @app.get("/api/v1/sources")
    def list_sources(knowledge_base_id: str | None = None) -> list[dict[str, object]]:
        with sqlite_connection(app_settings) as connection:
            return [
                _source_payload(item)
                for item in SourceRepository(connection).list(knowledge_base_id=knowledge_base_id)
            ]

    @app.post("/api/v1/sources/{source_id}/scan")
    def scan_source(source_id: str) -> dict[str, object]:
        with sqlite_connection(app_settings) as connection:
            try:
                run = IngestionService(connection=connection, settings=app_settings).scan_source(
                    source_id=source_id
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="source not found") from exc
            except PathAuthorizationError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        app.state.ingestion_worker.wake()
        return {
            "counts": run.result.counts,
            "jobs": [_job_payload(job) for job in run.jobs],
        }

    @app.get("/api/v1/documents")
    def list_documents(knowledge_base_id: str = Query(min_length=1)) -> list[dict[str, object]]:
        with sqlite_connection(app_settings) as connection:
            return [
                {
                    "document_id": row["id"],
                    "canonical_path": row["canonical_path"],
                    "visibility_state": row["visibility_state"],
                    "version_id": row["version_id"],
                    "version_state": row["version_state"],
                    "version_error": row["version_error"],
                    "chunk_count": row["chunk_count"],
                    "updated_at": row["updated_at"],
                }
                for row in DocumentRepository(connection).list_statuses(
                    knowledge_base_id=knowledge_base_id
                )
            ]

    @app.get("/api/v1/documents/summary")
    def document_summary(knowledge_base_id: str = Query(min_length=1)) -> dict[str, int]:
        with sqlite_connection(app_settings) as connection:
            row = DocumentRepository(connection).status_summary(knowledge_base_id=knowledge_base_id)
        return {key: int(value or 0) for key, value in dict(row).items()}

    @app.get("/api/v1/documents/failed")
    def failed_documents(
        knowledge_base_id: str = Query(min_length=1),
    ) -> list[dict[str, str | None]]:
        with sqlite_connection(app_settings) as connection:
            rows = DocumentRepository(connection).list_failed(knowledge_base_id=knowledge_base_id)
        return [
            {
                "document_id": row["document_id"],
                "canonical_path": row["canonical_path"],
                "error": row["error"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    @app.post("/api/v1/documents/retry-failed")
    def retry_failed_documents(knowledge_base_id: str = Query(min_length=1)) -> dict[str, int]:
        with sqlite_connection(app_settings) as connection:
            service = IngestionService(connection=connection, settings=app_settings)
            jobs = service.retry_failed_documents(knowledge_base_id=knowledge_base_id)
        app.state.ingestion_worker.wake()
        return {"queued_jobs": len(jobs)}

    @app.post("/api/v1/documents/{document_id}/reindex")
    def reindex_document(document_id: str) -> dict[str, object]:
        with sqlite_connection(app_settings) as connection:
            service = IngestionService(connection=connection, settings=app_settings)
            try:
                job = service.enqueue_reindex(document_id=document_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="document not found") from exc
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        app.state.ingestion_worker.wake()
        return _job_payload(job)

    @app.delete("/api/v1/documents/{document_id}")
    def delete_document(document_id: str) -> dict[str, object]:
        with sqlite_connection(app_settings) as connection:
            documents = DocumentRepository(connection)
            try:
                document = documents.mark_deleted(document_id=document_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="document not found") from exc
            active_version = documents.get_active_version(document_id)
            job = JobRepository(connection).enqueue(
                knowledge_base_id=document.knowledge_base_id,
                operation="delete",
                path=document.canonical_path,
                expected_sha256=active_version.sha256 if active_version else None,
            )
            job = IngestionService(connection=connection, settings=app_settings).process_job(
                job_id=job.id
            )
        return _job_payload(job)

    @app.get("/api/v1/documents/{document_id}/source-file")
    def open_source_file(document_id: str) -> FileResponse:
        with sqlite_connection(app_settings) as connection:
            try:
                document = DocumentRepository(connection).get(document_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="document not found") from exc
        if document.visibility_state != "visible":
            raise HTTPException(status_code=404, detail="document is not visible")
        source_path = Path(document.canonical_path)
        if not source_path.is_file():
            raise HTTPException(status_code=404, detail="source file is no longer available")
        return FileResponse(source_path)

    @app.get("/api/v1/documents/{document_id}/preview")
    def preview_document(document_id: str, chunk_id: str | None = None) -> HTMLResponse:
        with sqlite_connection(app_settings) as connection:
            try:
                document = DocumentRepository(connection).get(document_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="document not found") from exc
            if document.visibility_state != "visible":
                raise HTTPException(status_code=404, detail="document is not visible")
            rows = connection.execute(
                """
                SELECT chunks.id, chunks.text, chunks.page_no, chunks.section_path
                FROM chunks
                JOIN document_versions ON document_versions.id = chunks.document_version_id
                JOIN documents ON documents.active_version_id = document_versions.id
                WHERE documents.id = ?
                ORDER BY chunks.chunk_index
                """,
                (document_id,),
            ).fetchall()
        if chunk_id is not None and not any(row["id"] == chunk_id for row in rows):
            raise HTTPException(status_code=404, detail="chunk not found in document")
        source_path = Path(document.canonical_path)
        if not source_path.is_file():
            raise HTTPException(status_code=404, detail="source file is no longer available")
        return HTMLResponse(
            render_document_preview(
                source_path=source_path,
                indexed_chunks=[dict(row) for row in rows],
                focus_chunk_id=chunk_id,
            )
        )

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, object]:
        with sqlite_connection(app_settings) as connection:
            try:
                job = JobRepository(connection).get(job_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="job not found") from exc
        return _job_payload(job)

    @app.post("/api/v1/search")
    def search(request: SearchRequest) -> list[dict[str, object]]:
        with sqlite_connection(app_settings) as connection:
            retriever = _retriever(connection=connection, settings=app_settings)
            if request.top_k is not None:
                retriever.final_top_k = request.top_k
                retriever.candidate_top_k = max(retriever.candidate_top_k, request.top_k)
            results = retriever.search(
                knowledge_base_id=request.knowledge_base_id,
                query=request.query,
                allowed_document_ids=_allowed_document_ids(request.allowed_document_ids),
            )
        return [_search_payload(result) for result in results]

    @app.post("/api/v1/field-search")
    def field_search(request: FieldSearchRequest) -> list[dict[str, object]]:
        fields = [value.strip() for value in request.fields if value.strip()]
        if not fields:
            raise HTTPException(status_code=422, detail="at least one non-empty field is required")
        with sqlite_connection(app_settings) as connection:
            retriever = (
                _retriever(connection=connection, settings=app_settings)
                if request.mode == "hybrid"
                else None
            )
            return [
                {
                    "field": field,
                    "mode": request.mode,
                    "files": (
                        _hybrid_field_matches(
                            retriever=retriever,
                            knowledge_base_id=request.knowledge_base_id,
                            field=field,
                        )
                        if retriever is not None
                        else _field_matches(
                            connection=connection,
                            knowledge_base_id=request.knowledge_base_id,
                            field=field,
                        )
                    ),
                }
                for field in fields
            ]

    @app.post("/api/v1/query")
    def query(request: QueryRequest) -> dict[str, object]:
        with sqlite_connection(app_settings) as connection:
            retriever = _retriever(connection=connection, settings=app_settings)
            answer = RAGAnswerer(
                connection=connection,
                retriever=retriever,
                llm_provider=build_llm_provider(app_settings),
            ).answer(
                knowledge_base_id=request.knowledge_base_id,
                question=request.question,
                allowed_document_ids=_allowed_document_ids(request.allowed_document_ids),
            )
        return {
            "answer": answer.answer,
            "confidence": answer.confidence,
            "insufficient_evidence": answer.insufficient_evidence,
            "citations": [
                {
                    "citation_id": citation.citation_id,
                    "document_id": citation.document_id,
                    "file_name": citation.file_name,
                    "canonical_path": citation.canonical_path,
                    "page_no": citation.page_no,
                    "section_path": citation.section_path,
                    "quote": citation.quote,
                    "chunk_id": citation.chunk_id,
                    "bbox": citation.bbox,
                }
                for citation in answer.citations
            ],
        }

    return app


def _retriever(*, connection, settings: Settings) -> HybridRetriever:
    return HybridRetriever(
        connection=connection,
        embedding_provider=build_embedding_provider(settings),
        vector_store=get_local_vector_store(path=str(settings.qdrant_path)),
        collection_name=settings.vector_collection_name,
        vector_top_k=settings.vector_top_k,
        fts_top_k=settings.fts_top_k,
        candidate_top_k=settings.retrieval_candidate_top_k,
        max_chunks_per_document=settings.max_chunks_per_document,
        final_top_k=settings.final_top_k,
        reranker=build_reranker(settings),
    )


def _allowed_document_ids(values: list[str] | None) -> set[str] | None:
    return set(values) if values is not None else None


def _knowledge_base_payload(item: KnowledgeBase) -> dict[str, str]:
    return {
        "id": item.id,
        "name": item.name,
        "embedding_model": item.embedding_model,
        "vector_backend": item.vector_backend,
    }


def _source_payload(item: Source) -> dict[str, object]:
    return {
        "id": item.id,
        "knowledge_base_id": item.knowledge_base_id,
        "root_path": item.root_path,
        "include_patterns": item.include_patterns,
        "exclude_patterns": item.exclude_patterns,
        "scan_state": item.scan_state,
    }


def _job_payload(job: Job) -> dict[str, object]:
    return {
        "id": job.id,
        "knowledge_base_id": job.knowledge_base_id,
        "operation": job.operation,
        "path": job.path,
        "state": job.state,
        "attempts": job.attempts,
        "error": job.error,
    }


def _search_payload(result) -> dict[str, object]:
    return {
        "chunk_id": result.chunk_id,
        "document_id": result.document_id,
        "file_name": result.file_name,
        "canonical_path": result.canonical_path,
        "page_no": result.page_no,
        "section_path": result.section_path,
        "quote": result.quote,
        "bbox": result.bbox,
        "score": result.score,
        "sources": sorted(result.sources),
    }


def _field_matches(*, connection, knowledge_base_id: str, field: str) -> list[dict[str, object]]:
    escaped = field.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    rows = connection.execute(
        """
        SELECT documents.id AS document_id, documents.canonical_path, chunks.page_no,
            chunks.section_path, chunks.text
        FROM chunks
        JOIN document_versions ON document_versions.id = chunks.document_version_id
        JOIN documents ON documents.active_version_id = document_versions.id
        WHERE documents.knowledge_base_id = ?
            AND documents.visibility_state = 'visible'
            AND chunks.text LIKE ? ESCAPE '\\'
        ORDER BY documents.canonical_path, chunks.chunk_index
        LIMIT 300
        """,
        (knowledge_base_id, f"%{escaped}%"),
    ).fetchall()
    matches: dict[str, dict[str, object]] = {}
    for row in rows:
        document_id = row["document_id"]
        item = matches.setdefault(
            document_id,
            {
                "document_id": document_id,
                "file_name": Path(row["canonical_path"]).name,
                "canonical_path": row["canonical_path"],
                "matched_chunks": 0,
                "occurrences": 0,
                "page_no": row["page_no"],
                "section_path": row["section_path"],
                "quote": _field_quote(row["text"], field),
                "match_mode": "exact",
            },
        )
        item["matched_chunks"] += 1
        item["occurrences"] += row["text"].count(field)
    return list(matches.values())


def _hybrid_field_matches(
    *, retriever: HybridRetriever, knowledge_base_id: str, field: str
) -> list[dict[str, object]]:
    return [
        {
            "document_id": result.document_id,
            "file_name": result.file_name,
            "canonical_path": result.canonical_path,
            "matched_chunks": 1,
            "occurrences": result.quote.count(field),
            "page_no": result.page_no,
            "section_path": result.section_path,
            "quote": result.quote,
            "match_mode": "hybrid",
            "score": result.score,
            "sources": sorted(result.sources),
        }
        for result in retriever.search(knowledge_base_id=knowledge_base_id, query=field)
    ]


def _field_quote(text: str, field: str, *, radius: int = 100) -> str:
    text = clean_display_text(text)
    index = text.find(field)
    if index < 0:
        return text[: radius * 2]
    start = max(0, index - radius)
    end = min(len(text), index + len(field) + radius)
    prefix = "..." if start else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"


app = create_app()
