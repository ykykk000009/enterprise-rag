import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .embeddings import EmbeddingProvider
from .reranking import Reranker
from .vector_store import VectorStore


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    document_id: str
    file_name: str
    canonical_path: str
    page_no: int | None
    section_path: str | None
    quote: str
    bbox: tuple[float, float, float, float] | None
    score: float
    sources: frozenset[str] = field(default_factory=frozenset)


class HybridRetriever:
    def __init__(
        self,
        *,
        connection: sqlite3.Connection,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        collection_name: str = "document_chunks",
        vector_top_k: int = 15,
        fts_top_k: int = 15,
        rrf_k: int = 60,
        candidate_top_k: int = 60,
        max_chunks_per_document: int = 1,
        final_top_k: int = 6,
        reranker: Reranker | None = None,
    ) -> None:
        self.connection = connection
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.vector_top_k = vector_top_k
        self.fts_top_k = fts_top_k
        self.rrf_k = rrf_k
        self.candidate_top_k = candidate_top_k
        self.max_chunks_per_document = max_chunks_per_document
        self.final_top_k = final_top_k
        self.reranker = reranker

    def search(
        self,
        *,
        knowledge_base_id: str,
        query: str,
        allowed_document_ids: set[str] | None = None,
    ) -> list[SearchResult]:
        fts_ranked = self._fts_search(
            knowledge_base_id=knowledge_base_id,
            query=query,
            allowed_document_ids=allowed_document_ids,
        )
        contains_ranked = self._contains_search(
            knowledge_base_id=knowledge_base_id,
            query=query,
            allowed_document_ids=allowed_document_ids,
        )
        vector_ranked = self._vector_search(
            knowledge_base_id=knowledge_base_id,
            query=query,
            allowed_document_ids=allowed_document_ids,
        )
        fused_scores: dict[str, float] = {}
        sources: dict[str, set[str]] = {}
        for source_name, ranked_ids in [
            ("fts", fts_ranked),
            ("contains", contains_ranked),
            ("vector", vector_ranked),
        ]:
            for rank, chunk_id in enumerate(ranked_ids, start=1):
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
                sources.setdefault(chunk_id, set()).add(source_name)

        ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        candidate_ids = [chunk_id for chunk_id, _ in ranked[: self.candidate_top_k]]
        rows = self._load_chunk_rows(
            knowledge_base_id=knowledge_base_id,
            chunk_ids=candidate_ids,
            allowed_document_ids=allowed_document_ids,
        )
        by_id = {row["chunk_id"]: row for row in rows}
        rank_scores = self._rank_candidates(
            query=query,
            candidate_ids=candidate_ids,
            by_id=by_id,
            fused_scores=fused_scores,
        )
        chunk_ids = self._diversify_documents(
            candidate_ids=candidate_ids,
            by_id=by_id,
            rank_scores=rank_scores,
        )
        results: list[SearchResult] = []
        for chunk_id in chunk_ids:
            row = by_id.get(chunk_id)
            if row is None:
                continue
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=row["document_id"],
                    file_name=Path(row["canonical_path"]).name,
                    canonical_path=row["canonical_path"],
                    page_no=row["page_no"],
                    section_path=row["section_path"],
                    quote=row["text"],
                    bbox=_parse_bbox(row["bbox"]),
                    score=rank_scores[chunk_id],
                    sources=frozenset(sources.get(chunk_id, set())),
                )
            )
        return results

    def _rank_candidates(
        self,
        *,
        query: str,
        candidate_ids: list[str],
        by_id: dict[str, sqlite3.Row],
        fused_scores: dict[str, float],
    ) -> dict[str, float]:
        available_ids = [chunk_id for chunk_id in candidate_ids if chunk_id in by_id]
        if self.reranker is None:
            return {
                chunk_id: fused_scores[chunk_id]
                + _lexical_overlap_score(query, by_id[chunk_id]["text"])
                for chunk_id in available_ids
            }
        rerank_scores = self.reranker.score(
            query=query,
            passages=[by_id[chunk_id]["text"] for chunk_id in available_ids],
        )
        if len(rerank_scores) != len(available_ids):
            raise ValueError("reranker score count does not match candidate count")
        return {
            chunk_id: (
                rerank_score
                + fused_scores[chunk_id] * 0.01
                + _exact_field_bonus(query, by_id[chunk_id]["text"])
                + _table_total_bonus(query, by_id[chunk_id]["text"])
            )
            for chunk_id, rerank_score in zip(available_ids, rerank_scores, strict=True)
        }

    def _diversify_documents(
        self,
        *,
        candidate_ids: list[str],
        by_id: dict[str, sqlite3.Row],
        rank_scores: dict[str, float],
    ) -> list[str]:
        document_counts: dict[str, int] = {}
        selected: list[str] = []
        for chunk_id in sorted(
            candidate_ids, key=lambda item: rank_scores.get(item, float("-inf")), reverse=True
        ):
            row = by_id.get(chunk_id)
            if row is None:
                continue
            document_id = str(row["document_id"])
            if document_counts.get(document_id, 0) >= self.max_chunks_per_document:
                continue
            selected.append(chunk_id)
            document_counts[document_id] = document_counts.get(document_id, 0) + 1
            if len(selected) >= self.final_top_k:
                break
        return selected

    def _fts_search(
        self,
        *,
        knowledge_base_id: str,
        query: str,
        allowed_document_ids: set[str] | None,
    ) -> list[str]:
        params: list[object] = [knowledge_base_id, _fts_query(query)]
        acl_clause = ""
        if allowed_document_ids is not None:
            if not allowed_document_ids:
                return []
            placeholders = ",".join("?" for _ in allowed_document_ids)
            acl_clause = f" AND documents.id IN ({placeholders})"
            params.extend(sorted(allowed_document_ids))
        params.append(self.fts_top_k)
        rows = self.connection.execute(
            f"""
            SELECT chunks.id AS chunk_id
            FROM chunks_fts
            JOIN chunks ON chunks.id = chunks_fts.chunk_id
            JOIN document_versions ON document_versions.id = chunks.document_version_id
            JOIN documents ON documents.active_version_id = document_versions.id
            WHERE documents.knowledge_base_id = ?
                AND documents.visibility_state = 'visible'
                AND chunks_fts MATCH ?
                {acl_clause}
            ORDER BY bm25(chunks_fts)
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [row["chunk_id"] for row in rows]

    def _vector_search(
        self,
        *,
        knowledge_base_id: str,
        query: str,
        allowed_document_ids: set[str] | None,
    ) -> list[str]:
        if allowed_document_ids is not None and not allowed_document_ids:
            return []
        query_vector = self.embedding_provider.embed_texts([query])[0]
        filter_conditions: dict[str, str | list[str]] = {"knowledge_base_id": knowledge_base_id}
        if allowed_document_ids is not None:
            filter_conditions["document_id"] = sorted(allowed_document_ids)
        results = self.vector_store.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=self.vector_top_k,
            filter_conditions=filter_conditions,
        )
        return [
            str(result.payload["chunk_id"]) for result in results if "chunk_id" in result.payload
        ]

    def _contains_search(
        self,
        *,
        knowledge_base_id: str,
        query: str,
        allowed_document_ids: set[str] | None,
    ) -> list[str]:
        terms = _contains_terms(query)
        if not terms or (allowed_document_ids is not None and not allowed_document_ids):
            return []
        clauses = " OR ".join("chunks.text LIKE ? ESCAPE '\\'" for term, _ in terms)
        score_clauses = " + ".join(
            "CASE WHEN chunks.text LIKE ? ESCAPE '\\' THEN ? ELSE 0 END" for _ in terms
        )
        parameters: list[object] = []
        for term, weight in terms:
            parameters.extend([_like_pattern(term), weight])
        parameters.append(knowledge_base_id)
        parameters.extend(_like_pattern(term) for term, _ in terms)
        acl_clause = ""
        if allowed_document_ids is not None:
            placeholders = ",".join("?" for _ in allowed_document_ids)
            acl_clause = f" AND documents.id IN ({placeholders})"
            parameters.extend(sorted(allowed_document_ids))
        parameters.append(self.fts_top_k)
        rows = self.connection.execute(
            f"""
            SELECT chunks.id AS chunk_id, ({score_clauses}) AS lexical_score
            FROM chunks
            JOIN document_versions ON document_versions.id = chunks.document_version_id
            JOIN documents ON documents.active_version_id = document_versions.id
            WHERE documents.knowledge_base_id = ?
                AND documents.visibility_state = 'visible'
                AND ({clauses})
                {acl_clause}
            ORDER BY lexical_score DESC, chunks.token_count ASC, chunks.chunk_index
            LIMIT ?
            """,
            parameters,
        ).fetchall()
        return [row["chunk_id"] for row in rows]

    def _load_chunk_rows(
        self,
        *,
        knowledge_base_id: str,
        chunk_ids: list[str],
        allowed_document_ids: set[str] | None,
    ) -> list[sqlite3.Row]:
        if not chunk_ids:
            return []
        params: list[object] = [knowledge_base_id]
        chunk_placeholders = ",".join("?" for _ in chunk_ids)
        params.extend(chunk_ids)
        acl_clause = ""
        if allowed_document_ids is not None:
            if not allowed_document_ids:
                return []
            placeholders = ",".join("?" for _ in allowed_document_ids)
            acl_clause = f" AND documents.id IN ({placeholders})"
            params.extend(sorted(allowed_document_ids))
        return self.connection.execute(
            f"""
            SELECT
                chunks.id AS chunk_id,
                chunks.text,
                chunks.page_no,
                chunks.section_path,
                chunks.bbox,
                documents.id AS document_id,
                documents.canonical_path
            FROM chunks
            JOIN document_versions ON document_versions.id = chunks.document_version_id
            JOIN documents ON documents.active_version_id = document_versions.id
            WHERE documents.knowledge_base_id = ?
                AND documents.visibility_state = 'visible'
                AND chunks.id IN ({chunk_placeholders})
                {acl_clause}
            """,
            params,
        ).fetchall()


def _parse_bbox(value: str | None) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    loaded = json.loads(value)
    if loaded is None:
        return None
    return tuple(float(item) for item in loaded)


def _fts_query(query: str) -> str:
    terms = _fts_terms(query)
    if not terms:
        return '""'
    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)


def _fts_terms(query: str) -> list[str]:
    quoted = _quoted_phrases(query)
    if quoted:
        return quoted
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z0-9_]+", query):
        if len(token) >= 2:
            terms.append(token)
    for sequence in re.findall(r"[\u4e00-\u9fff]+", query):
        if len(sequence) <= 8:
            terms.append(sequence)
    return list(dict.fromkeys(terms))[:16]


def _contains_terms(query: str) -> list[tuple[str, float]]:
    terms: list[tuple[str, float]] = []

    def add(value: str, weight: float) -> None:
        if value and value not in {item[0] for item in terms}:
            terms.append((value, weight))

    for phrase in _quoted_phrases(query):
        add(phrase, 100.0)

    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]*", query):
        if len(token) >= 2:
            add(token, 20.0)
    for sequence in re.findall(r"[\u4e00-\u9fff]+", query):
        if len(sequence) <= 8:
            add(sequence, 30.0)
        elif sequence not in _quoted_phrases(query):
            add(sequence, 40.0)
        for index in range(len(sequence) - 1):
            add(sequence[index : index + 2], 2.0)
    return terms[:24]


def _quoted_phrases(query: str) -> list[str]:
    phrases = re.findall(r'[“"]([^”"]{2,120})[”"]', query)
    return list(dict.fromkeys(" ".join(phrase.split()) for phrase in phrases if phrase.strip()))


def _lexical_overlap_score(query: str, text: str) -> float:
    normalized = " ".join(text.split())
    return sum(weight for term, weight in _contains_terms(query) if term in normalized)


def _exact_field_bonus(query: str, text: str) -> float:
    normalized = " ".join(text.split())
    return 100.0 * sum(phrase in normalized for phrase in _quoted_phrases(query))


def _table_total_bonus(query: str, text: str) -> float:
    if not any(term in query for term in ("总价", "总和", "合计", "总额", "总报价")):
        return 0.0
    normalized = " ".join(text.split())
    if "序号：合计" not in normalized:
        return 0.0
    score = 20.0
    if "计算值：" in normalized:
        score += 10.0
    if "医院总价：" in normalized:
        score += 5.0
    return score


def _like_pattern(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return "%" + escaped + "%"
