"""Thin T09 adapters over the running production API."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.request import Request, urlopen


def retrieve(
    *, base_url: str, knowledge_base_id: str, question: str, top_k: int
) -> list[dict[str, Any]]:
    payload = _request(
        base_url=base_url,
        path="/api/v1/search",
        body={"knowledge_base_id": knowledge_base_id, "query": question},
    )
    return [
        {
            "rank": rank,
            "document_path": item.get("canonical_path"),
            "document_id": item.get("document_id"),
            "chunk_id": item.get("chunk_id"),
            "page": item.get("page_no"),
            "section": item.get("section_path"),
            "text": item.get("quote"),
            "score": item.get("score"),
            "retrieval_method": item.get("sources", []),
        }
        for rank, item in enumerate(payload[:top_k], start=1)
    ]


def answer(*, base_url: str, knowledge_base_id: str, question: str) -> dict[str, Any]:
    started = time.perf_counter()
    payload = _request(
        base_url=base_url,
        path="/api/v1/query",
        body={"knowledge_base_id": knowledge_base_id, "question": question},
    )
    return {
        "answer": payload.get("answer", ""),
        "citations": payload.get("citations", []),
        "insufficient_evidence": bool(payload.get("insufficient_evidence")),
        "total_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def ingest(*, base_url: str, source_id: str) -> dict[str, Any]:
    return _request(base_url=base_url, path=f"/api/v1/sources/{source_id}/scan", body={})


def _request(*, base_url: str, path: str, body: dict[str, Any]) -> Any:
    request = Request(
        url=base_url.rstrip("/") + path,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=180) as response:
        return json.load(response)
