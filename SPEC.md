# Lightweight Local Enterprise Document RAG Agent

## 1. Product scope

The application solves local PDF/DOCX retrieval and evidence-grounded question answering. It supports directory authorization, incremental indexing, hybrid retrieval, local or configurable generation, and original-source navigation.

MVP includes PDF, DOCX, TXT and Markdown; file-level incremental indexing; SQLite FTS5 + vector retrieval; RRF fusion; optional reranking; cited answers; job status; and a minimal web UI/API.

MVP excludes whole-disk scanning, old `.doc`, full Excel/PPT semantics, mandatory OCR, mandatory graph construction, multi-user SSO, microservices, and distributed deployment.

## 2. Lightweight architecture

```text
Browser / API client
        |
   FastAPI app
    /       \
Indexer     Query pipeline
  |          |-- SQLite FTS5
Parser       |-- Qdrant Local vectors
  |          |-- RRF fusion
Chunker      |-- optional reranker
  |          |-- context builder
CPU embedder |-- small local LLM or configured API

SQLite: metadata, jobs, ACL, audit, optional graph tables
```

Default implementations:

- Python 3.11, FastAPI, Pydantic Settings.
- SQLite WAL + FTS5.
- Qdrant Local; vector backend behind a protocol.
- PyMuPDF for PDF and python-docx for DOCX.
- CPU-friendly multilingual/Chinese embedding model configured through `EmbeddingProvider`; default `BAAI/bge-small-zh-v1.5`.
- Reranker disabled by default. Enable a small CrossEncoder only when evaluation proves value.
- Generation through `LLMProvider`; default small GGUF Qwen model via llama.cpp, but allow a local OpenAI-compatible endpoint.
- `watchdog` events plus periodic reconciliation scan.

## 3. Resource behavior

- Idle CPU should remain near zero except scheduled reconciliation.
- Embedding batch size defaults to 8 and is configurable.
- Index one file at a time by default.
- Limit parsing/OCR concurrency to one.
- OCR is optional and page-routed only when extracted text density is below a threshold.
- A 10 GB source directory means the system can discover and incrementally process it; it does not mean all source text or vectors remain in RAM.
- Provide include/exclude patterns and an optional maximum initial indexing budget so users can index high-value subdirectories first.

## 4. Core data model

SQLite tables:

- `knowledge_bases`: isolation boundary and model/index profile.
- `sources`: authorized directory, include/exclude rules, scan state.
- `documents`: stable document identity and active version.
- `document_versions`: size, `mtime_ns`, SHA-256, parser version, state.
- `chunks`: text, page, section path, bbox, token count, text hash.
- `index_records`: vector/FTS identifiers and index version.
- `jobs`: persistent idempotent ingestion tasks, lease, attempts and error.
- `users`, `roles`, `permissions`: minimal ACL abstraction.
- `query_audits`: query, filters, cited chunks and latency.
- optional `entities`, `relations`, `relation_evidence` for Phase 2.

Required uniqueness:

- `(knowledge_base_id, canonical_path)` on documents.
- `(document_id, sha256)` on versions.
- `(document_version_id, chunk_index)` on chunks.
- unique `job_key = SHA256(kb_id + operation + path + expected_sha256)`.

## 5. Incremental indexing

Use both file events and scheduled reconciliation. File events are debounced and accepted only after size and `mtime_ns` remain stable across two checks. Scheduled scanning repairs missed events.

Fingerprint strategy:

1. Compare canonical path, size and `mtime_ns`.
2. Compute streaming SHA-256 only for new or changed candidates.
3. If content hash is unchanged after rename/move, update metadata without re-embedding.

Operations:

- ADD: parse, chunk, embed, write FTS/vector records, validate counts, then activate.
- UPDATE: build a new version without deleting the old one; atomically switch only after all base indexes succeed.
- MOVE/RENAME: update path when SHA-256 matches.
- DELETE: make the document immediately invisible, then remove derived records in the background.

Jobs must be idempotent and restartable. A failed document must not block other documents.

## 6. Parsing and chunking

Normalized parser output contains pages and ordered blocks. Every block may carry type, text, page, section path, bbox and confidence.

Chunk by heading and paragraph boundaries, target 500 tokens with 60-token overlap, and permit 300–800 tokens. Preserve tables when practical. Store page, section path, bbox and adjacent chunk IDs. Normalize whitespace only; do not alter numbers, units, identifiers or case.

## 7. Retrieval and generation

1. Authenticate and apply `knowledge_base_id` and ACL filters.
2. Run vector Top-15 and FTS5/BM25 Top-15.
3. Fuse with reciprocal rank fusion, `k=60`.
4. Optionally rerank and retain Top-6 when the feature flag is enabled.
5. Build context under a token budget with stable citation IDs.
6. Generate only from evidence.
7. Validate that every returned citation exists and remains authorized.

Answer schema:

```json
{
  "answer": "... [1]",
  "confidence": "high|medium|low",
  "insufficient_evidence": false,
  "citations": [{
    "citation_id": 1,
    "document_id": "uuid",
    "file_name": "example.pdf",
    "page_no": 12,
    "section_path": "3.2 Technical Parameters",
    "quote": "source excerpt",
    "chunk_id": "uuid",
    "bbox": [0, 0, 0, 0]
  }]
}
```

When retrieval scores or evidence coverage are below configured thresholds, set `insufficient_evidence=true` and do not fill gaps with model knowledge.

## 8. Security

- Resolve and normalize paths, then prove they remain inside an authorized root.
- Reject symlink escape, unsupported MIME types, oversized files, encrypted files and parser timeouts with explicit error states.
- Treat document text as untrusted data; instructions inside documents cannot change system policy or tool access.
- Do not log full document text, model secrets or sensitive absolute paths.
- Permission revocation must hide data before asynchronous index cleanup.

## 9. Minimum API

- `POST /api/v1/knowledge-bases`
- `POST /api/v1/sources`
- `POST /api/v1/sources/{id}/scan`
- `GET /api/v1/documents`
- `POST /api/v1/documents/{id}/reindex`
- `DELETE /api/v1/documents/{id}`
- `POST /api/v1/search`
- `POST /api/v1/query`
- `GET /api/v1/jobs/{id}`
- `GET /health/live`
- `GET /health/ready`

## 10. Phase 2 knowledge graph

Do not implement until the base hybrid RAG passes evaluation. Define a bounded domain schema, store entities/relations/evidence in SQLite, and use recursive CTE or bounded application traversal. Every relation must cite source chunks. Route only entity-relation and multi-hop questions to graph retrieval. Upgrade to Neo4j only after measured graph size/query complexity exceeds SQLite capability.

