"""Measure local RAG correctness, latency, and process memory for T08."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import psutil

from enterprise_document_rag.config import Settings
from enterprise_document_rag.db import sqlite_connection
from enterprise_document_rag.embeddings import build_embedding_provider
from enterprise_document_rag.qa import RAGAnswerer
from enterprise_document_rag.retrieval import HybridRetriever
from enterprise_document_rag.vector_store import QdrantLocalVectorStore


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run T08 local RAG evaluation.")
    parser.add_argument("--knowledge-base-id", required=True)
    parser.add_argument("--evaluation-set", default="evaluation/rag_eval_set.json")
    parser.add_argument("--output", default="reports/t08-baseline.json")
    parser.add_argument("--database", default="data/agent.db")
    parser.add_argument("--qdrant", default="data/qdrant")
    parser.add_argument("--embedding-backend", choices=["hash", "bge"], default="hash")
    parser.add_argument("--collection", default="document_chunks")
    args = parser.parse_args()

    cases = json.loads(Path(args.evaluation_set).read_text(encoding="utf-8"))
    settings = Settings(
        database_url=args.database,
        qdrant_path=Path(args.qdrant),
        embedding_backend=args.embedding_backend,
        vector_collection_name=args.collection,
    )
    provider = build_embedding_provider(settings)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    rows: list[dict[str, object]] = []
    latencies_ms: list[float] = []

    with sqlite_connection(settings) as connection:
        retriever = HybridRetriever(
            connection=connection,
            embedding_provider=provider,
            vector_store=QdrantLocalVectorStore(path=str(settings.qdrant_path)),
            collection_name=settings.vector_collection_name,
            vector_top_k=settings.vector_top_k,
            fts_top_k=settings.fts_top_k,
            final_top_k=settings.final_top_k,
        )
        answerer = RAGAnswerer(connection=connection, retriever=retriever)
        for case in cases:
            started = time.perf_counter()
            answer = answerer.answer(
                knowledge_base_id=args.knowledge_base_id,
                question=case["question"],
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            latencies_ms.append(elapsed_ms)
            peak_rss = max(peak_rss, process.memory_info().rss)
            file_names = [citation.file_name for citation in answer.citations]
            expected = case["answerable"]
            if expected:
                expected_names = case.get("expected_file_contains", [])
                passed = not answer.insufficient_evidence and any(
                    marker in name for marker in expected_names for name in file_names
                )
            else:
                passed = answer.insufficient_evidence and not answer.citations
            rows.append(
                {
                    "id": case["id"],
                    "passed": passed,
                    "latency_ms": round(elapsed_ms, 2),
                    "insufficient_evidence": answer.insufficient_evidence,
                    "citations": file_names,
                }
            )

    report = {
        "provider": f"{args.embedding_backend}-cpu",
        "case_count": len(rows),
        "passed_count": sum(bool(row["passed"]) for row in rows),
        "p50_latency_ms": round(percentile(latencies_ms, 0.5), 2),
        "p95_latency_ms": round(percentile(latencies_ms, 0.95), 2),
        "peak_rss_mb": round(peak_rss / 1024 / 1024, 2),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed_count"] == report["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
