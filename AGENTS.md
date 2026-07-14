# Enterprise Document Agent — Codex Instructions

## Goal

Build a lightweight, local-first MVP for enterprise PDF/DOCX retrieval and cited question answering.

## Non-negotiable constraints

1. Run on a normal office computer: target 4–8 CPU cores and 8–16 GB RAM.
2. GPU is optional. The default path must run on CPU.
3. Use a single Python application process plus at most one controlled background worker thread.
4. Use SQLite in WAL mode for metadata, jobs, ACL, and FTS5 keyword search.
5. Use Qdrant Local by default for vectors. Keep a `VectorStore` interface so FAISS or Qdrant Server can be substituted later.
6. Do not add PostgreSQL, OpenSearch, Elasticsearch, Neo4j, Redis, Celery, Kafka, Kubernetes, microservices, or multi-agent frameworks in the MVP.
7. Do not scan the whole computer. Read only administrator-authorized directories.
8. Treat source directories as read-only. Never modify, move, or delete source documents.
9. Never load the complete corpus into memory. Process one file at a time and embeddings in small batches.
10. Every answer must contain valid source citations. If evidence is insufficient, refuse to infer from model knowledge.
11. Apply knowledge-base and ACL filters before candidate retrieval, not after generation.
12. Knowledge graph support is Phase 2 and must not block the MVP. If implemented, use SQLite entity/relation/evidence tables first.

## Required workflow

1. Inspect existing files and preserve user changes.
2. Read `SPEC.md`, `spec.json`, `tasks.json`, and `acceptance.json` before coding.
3. Implement tasks in their declared order. Do not begin a later phase until the current phase acceptance checks pass.
4. Keep model, path, threshold, and backend choices configurable; do not hard-code them.
5. After every task, run its tests and update status only when evidence is available.
6. Prefer the smallest dependency that satisfies the requirement.

## Definition of done

The MVP is complete only when a user can authorize a folder, index supported files incrementally, ask a natural-language question, receive a grounded answer with file/page/section/quote citations, update or delete a source file without rebuilding the full knowledge base, and restart the application without duplicate chunks or lost job state.

