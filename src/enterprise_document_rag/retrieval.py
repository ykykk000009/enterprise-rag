import json
import re
import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from .embeddings import EmbeddingProvider
from .reranking import Reranker
from .text_utils import clean_display_text, is_near_duplicate, normalized_text_key
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
    page_range: tuple[int, int] | None = None
    bbox_list: tuple[tuple[float, float, float, float], ...] = ()
    content_type: str = "text"
    source_type: str = "native_text"
    ocr_confidence: float | None = None
    block_types: tuple[str, ...] = ()
    table_markdown: str | None = None
    image_path: str | None = None
    caption: str | None = None
    image_metadata: dict[str, object] | None = None


class _QueryVectorCache:
    def __init__(self, *, max_size: int = 256) -> None:
        self.max_size = max_size
        self._values: OrderedDict[tuple[int, str], tuple[float, ...]] = OrderedDict()
        self._lock = threading.Lock()

    def get_or_create(self, *, provider: EmbeddingProvider, query: str) -> list[float]:
        key = (id(provider), " ".join(query.split()))
        with self._lock:
            cached = self._values.get(key)
            if cached is not None:
                self._values.move_to_end(key)
                return list(cached)
        vector = tuple(float(value) for value in provider.embed_texts([query])[0])
        with self._lock:
            self._values[key] = vector
            self._values.move_to_end(key)
            while len(self._values) > self.max_size:
                self._values.popitem(last=False)
        return list(vector)


_QUERY_VECTOR_CACHE = _QueryVectorCache()


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
        simple_candidate_top_k: int = 6,
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
        self.simple_candidate_top_k = simple_candidate_top_k
        self.final_top_k = final_top_k
        self.reranker = reranker

    def search(
        self,
        *,
        knowledge_base_id: str,
        query: str,
        allowed_document_ids: set[str] | None = None,
        max_chunks_per_document: int | None = None,
        candidate_top_k: int | None = None,
        final_top_k: int | None = None,
    ) -> list[SearchResult]:
        exact_ranked = self._exact_search(
            knowledge_base_id=knowledge_base_id,
            query=query,
            allowed_document_ids=allowed_document_ids,
        )
        if exact_ranked:
            fts_ranked = exact_ranked
            contains_ranked = exact_ranked
            vector_ranked = []
        else:
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
        effective_candidate_top_k = candidate_top_k or (
            self.candidate_top_k
            if is_complex_query(query)
            else min(self.candidate_top_k, self.simple_candidate_top_k)
        )
        candidate_ids = [chunk_id for chunk_id, _ in ranked[:effective_candidate_top_k]]
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
            max_chunks_per_document=max_chunks_per_document,
            final_top_k=final_top_k,
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
                    quote=_focused_quote(text=str(row["text"]), query=query),
                    bbox=_parse_bbox(row["bbox"]),
                    score=rank_scores[chunk_id],
                    sources=frozenset(sources.get(chunk_id, set())),
                    page_range=_parse_page_range(row["page_range"]),
                    bbox_list=_parse_bbox_list(row["bbox_list"]),
                    content_type=str(row["content_type"]),
                    source_type=str(row["source_type"]),
                    ocr_confidence=(
                        float(row["ocr_confidence"])
                        if row["ocr_confidence"] is not None
                        else None
                    ),
                    block_types=tuple(json.loads(row["block_types"] or "[]")),
                    table_markdown=row["table_markdown"],
                    image_path=row["image_path"],
                    caption=row["caption"],
                    image_metadata=json.loads(row["image_metadata"])
                    if row["image_metadata"]
                    else None,
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
                + _structure_relevance_bonus(query, by_id[chunk_id])
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
                + _structure_relevance_bonus(query, by_id[chunk_id])
            )
            for chunk_id, rerank_score in zip(available_ids, rerank_scores, strict=True)
        }

    def _diversify_documents(
        self,
        *,
        candidate_ids: list[str],
        by_id: dict[str, sqlite3.Row],
        rank_scores: dict[str, float],
        max_chunks_per_document: int | None,
        final_top_k: int | None,
    ) -> list[str]:
        max_per_document = max_chunks_per_document or self.max_chunks_per_document
        result_limit = final_top_k or self.final_top_k
        document_counts: dict[str, int] = {}
        selected_text_keys: list[str] = []
        selected: list[str] = []
        for chunk_id in sorted(
            candidate_ids, key=lambda item: rank_scores.get(item, float("-inf")), reverse=True
        ):
            row = by_id.get(chunk_id)
            if row is None:
                continue
            document_id = str(row["document_id"])
            if document_counts.get(document_id, 0) >= max_per_document:
                continue
            text_key = normalized_text_key(str(row["text"]))
            if is_near_duplicate(text_key, selected_text_keys):
                continue
            selected.append(chunk_id)
            selected_text_keys.append(text_key)
            document_counts[document_id] = document_counts.get(document_id, 0) + 1
            if len(selected) >= result_limit:
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

    def _exact_search(
        self,
        *,
        knowledge_base_id: str,
        query: str,
        allowed_document_ids: set[str] | None,
    ) -> list[str]:
        """Short-circuit semantic search for identifiable literal phrases."""
        terms = _exact_terms(query)
        if not terms or (allowed_document_ids is not None and not allowed_document_ids):
            return []
        match_params: list[object] = []
        match_clauses: list[str] = []
        score_clauses: list[str] = []
        for term in terms:
            pattern = _like_pattern(term)
            match_clauses.append(
                "(chunks.text LIKE ? ESCAPE '\\' OR chunks_fts.text LIKE ? ESCAPE '\\')"
            )
            match_params.extend((pattern, pattern))
            score_clauses.append(
                "CASE WHEN chunks.text LIKE ? ESCAPE '\\' "
                "OR chunks_fts.text LIKE ? ESCAPE '\\' THEN ? ELSE 0 END"
            )
        score_params: list[object] = []
        for term in terms:
            pattern = _like_pattern(term)
            score_params.extend((pattern, pattern, float(len(term))))
        table_priority = (
            " + CASE WHEN chunks.content_type = 'table' THEN 100 ELSE 0 END"
            if _is_table_operation_query(query)
            else ""
        )
        params: list[object] = [*score_params, knowledge_base_id, *match_params]
        acl_clause = ""
        if allowed_document_ids is not None:
            placeholders = ",".join("?" for _ in allowed_document_ids)
            acl_clause = f" AND documents.id IN ({placeholders})"
            params.extend(sorted(allowed_document_ids))
        params.append(self.fts_top_k)
        rows = self.connection.execute(
            f"""
            SELECT chunks.id AS chunk_id,
                ({' + '.join(score_clauses)}{table_priority}) AS exact_score
            FROM chunks
            JOIN chunks_fts ON chunks_fts.chunk_id = chunks.id
            JOIN document_versions ON document_versions.id = chunks.document_version_id
            JOIN documents ON documents.active_version_id = document_versions.id
            WHERE documents.knowledge_base_id = ?
                AND documents.visibility_state = 'visible'
                AND ({' OR '.join(match_clauses)})
                {acl_clause}
            ORDER BY exact_score DESC, chunks.chunk_index
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
        query_vector = _QUERY_VECTOR_CACHE.get_or_create(
            provider=self.embedding_provider,
            query=query,
        )
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
                chunks.page_range,
                chunks.section_path,
                chunks.bbox,
                chunks.bbox_list,
                chunks.content_type,
                chunks.source_type,
                chunks.ocr_confidence,
                chunks.block_types,
                chunks.table_markdown,
                chunks.image_path,
                chunks.caption,
                chunks.image_metadata,
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


def _parse_page_range(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    loaded = json.loads(value)
    if not isinstance(loaded, list) or len(loaded) != 2:
        return None
    return (int(loaded[0]), int(loaded[1]))


def _parse_bbox_list(
    value: str | None,
) -> tuple[tuple[float, float, float, float], ...]:
    if value is None:
        return ()
    loaded = json.loads(value)
    if not isinstance(loaded, list):
        return ()
    boxes = []
    for item in loaded:
        if isinstance(item, list) and len(item) == 4:
            boxes.append(tuple(float(coordinate) for coordinate in item))
    return tuple(boxes)


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
    terms.extend(_mixed_identifier_terms(query))
    terms.extend(_table_operation_terms(query))
    for token in re.findall(r"[A-Za-z0-9_]+", query):
        if len(token) >= 2:
            terms.append(token)
    for sequence in re.findall(r"[\u4e00-\u9fff]+", query):
        terms.extend(_chinese_query_phrases(sequence))
    return list(dict.fromkeys(terms))[:16]


def _contains_terms(query: str) -> list[tuple[str, float]]:
    terms: list[tuple[str, float]] = []

    def add(value: str, weight: float) -> None:
        if value and value not in {item[0] for item in terms}:
            terms.append((value, weight))

    for phrase in _quoted_phrases(query):
        add(phrase, 100.0)

    for identifier in _mixed_identifier_terms(query):
        # A label such as 产品B / 型号A1 identifies a row far more precisely
        # than the surrounding natural-language question.
        add(identifier, 80.0)
    for table_term in _table_operation_terms(query):
        add(table_term, 60.0)

    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]*", query):
        if len(token) >= 2:
            add(token, 20.0)
    for sequence in re.findall(r"[\u4e00-\u9fff]+", query):
        if len(sequence) > 8 and sequence not in _quoted_phrases(query):
            add(sequence, 40.0)
        for phrase in _chinese_query_phrases(sequence):
            add(phrase, 20.0 + len(phrase) * 2.0)
        for index in range(len(sequence) - 1):
            add(sequence[index : index + 2], 2.0)
    return terms[:24]


def _exact_terms(query: str) -> list[str]:
    terms: list[str] = []

    def add(value: str) -> None:
        value = " ".join(value.split()).strip()
        if len(value) >= 3 and value not in terms:
            terms.append(value)

    for phrase in _quoted_phrases(query):
        add(phrase)
    for identifier in _mixed_identifier_terms(query):
        add(identifier)
    for table_term in _table_operation_terms(query):
        if table_term not in terms:
            terms.append(table_term)
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,}", query):
        add(token)
    question_words = re.compile(
        r"(?:\u662f\u4ec0\u4e48|\u662f\u591a\u5c11|\u6709\u54ea\u4e9b|\u54ea\u4e9b|"
        r"\u600e\u4e48\u7406\u89e3|\u600e\u4e48\u529e|\u5982\u4f55|\u4e3a\u4ec0\u4e48|"
        r"\u662f\u5426|\u9700\u8981|\u591a\u5c11)$"
    )
    for sequence in re.findall(r"[\u4e00-\u9fff]+", query):
        cleaned = question_words.sub("", sequence)
        for part in re.split(r"[\u7684\u4e2d\u91cc\u4e0e\u53ca\u5bf9\u4e8e]", cleaned):
            add(part)
    return terms[:8]


def _mixed_identifier_terms(query: str) -> list[str]:
    """Keep labels such as 产品B and 型号A1 intact during Chinese tokenization."""
    matches = re.findall(
        r"(?:[\u4e00-\u9fff]+[A-Za-z0-9]+|[A-Za-z0-9]+[\u4e00-\u9fff]+)",
        query,
    )
    return list(dict.fromkeys(item for item in matches if len(item) >= 3))


def _table_operation_terms(query: str) -> list[str]:
    """Extract a short table field immediately before a numeric-operation word."""
    values: list[str] = []
    pattern = re.compile(r"([\u4e00-\u9fff]{2,}?)(?:数量|销量|金额|价格|最高|最大|最低|最小)")
    for match in pattern.finditer(query):
        candidate = match.group(1).split("的")[-1]
        if len(candidate) >= 2:
            values.append(candidate[-2:])
    return list(dict.fromkeys(values))


def _is_table_operation_query(query: str) -> bool:
    normalized = re.sub(r"\s+", "", query)
    return bool(_table_operation_terms(query)) and any(
        marker in normalized
        for marker in (
            "数量",
            "销量",
            "金额",
            "价格",
            "合计",
            "总和",
            "最高",
            "最大",
            "最低",
            "最小",
        )
    )


def _quoted_phrases(query: str) -> list[str]:
    phrases = re.findall(r'[“"]([^”"]{2,120})[”"]', query)
    return list(dict.fromkeys(" ".join(phrase.split()) for phrase in phrases if phrase.strip()))


def is_complex_query(query: str) -> bool:
    normalized = "".join(query.split())
    markers = (
        "\u54ea\u4e9b",
        "\u5168\u90e8",
        "\u539f\u5219",
        "\u6b65\u9aa4",
        "\u5408\u8ba1",
        "\u603b\u548c",
        "\u8be6\u7ec6",
        "\u5305\u62ec",
        "\u5206\u522b",
        "\u6761\u4ef6",
        "\u8981\u6c42",
        "\u662f\u5426",
        "\u533a\u522b",
        "\u5982\u4f55",
        "\u4e3a\u4ec0\u4e48",
        "\u9700\u8981",
        "\u505a\u4ec0\u4e48",
    )
    return len(normalized) >= 20 or any(marker in normalized for marker in markers)


def _chinese_query_phrases(sequence: str) -> list[str]:
    question_suffixes = (
        r"(?:\u662f\u4ec0\u4e48|\u6709\u54ea\u4e9b|"
        r"\u600e\u4e48\u529e|\u5982\u4f55)$"
    )
    trimmed = re.sub(question_suffixes, "", sequence).lstrip("的中里")
    phrases: list[str] = []

    def add(value: str) -> None:
        if 2 <= len(value) <= 12 and value not in phrases:
            phrases.append(value)

    add(trimmed)
    for part in re.split(r"[\u7684\u4e2d\u91cc]", trimmed):
        add(part)
    return phrases


def _focused_quote(*, text: str, query: str, max_chars: int = 900) -> str:
    cleaned = clean_display_text(text)
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", cleaned) if item.strip()]
    if len(paragraphs) <= 3:
        return cleaned
    terms = _contains_terms(query)
    scores = [
        sum(weight for term, weight in terms if term in paragraph) for paragraph in paragraphs
    ]
    best_index = max(range(len(paragraphs)), key=scores.__getitem__)
    selected = paragraphs[max(0, best_index - 2) : best_index + 3]
    focused = "\n\n".join(selected)
    return focused if len(focused) <= max_chars else focused[:max_chars].rstrip() + "..."


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


def _structure_relevance_bonus(query: str, row: sqlite3.Row) -> float:
    bonus = 0.0
    section_path = str(row["section_path"] or "")
    if section_path:
        bonus += min(_lexical_overlap_score(query, section_path), 1.5)
    content_type = str(row["content_type"] or "text")
    normalized = query.casefold()
    table_operation_markers = (
        "数量",
        "销量",
        "金额",
        "价格",
        "合计",
        "总和",
        "最高",
        "最大",
        "最低",
        "最小",
    )
    if content_type == "table" and any(marker in normalized for marker in table_operation_markers):
        # A table is the primary evidence for a direct lookup or calculation;
        # prose mentioning the same business term should not outrank it.
        bonus += 20.0
    if content_type == "table" and any(
        marker in normalized
        for marker in ("表", "表格", "型号", "参数", "合计", "总计", "table", "total")
    ):
        bonus += 2.0
    if content_type == "image" and any(
        marker in normalized
        for marker in ("图", "流程图", "架构", "曲线", "figure", "diagram", "chart")
    ):
        bonus += 2.0
    return bonus


def _like_pattern(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return "%" + escaped + "%"
