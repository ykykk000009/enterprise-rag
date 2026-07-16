"""Build a reproducible, source-verified retrieval benchmark for the active corpus."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enterprise_document_rag.config import Settings
from enterprise_document_rag.db import sqlite_connection
from enterprise_document_rag.text_utils import clean_display_text

SEED = 20260716
CASE_COUNT = 100
MAX_CASES_PER_DOCUMENT = 8

CATEGORY_QUOTAS = {
    "filename_lookup": 18,
    "exact_fact": 22,
    "numeric_parameter": 18,
    "procedure_process": 18,
    "structured_content": 10,
    "fuzzy_semantic": 14,
}

NUMERIC_MARKERS = re.compile(r"\d|MPa|mm|kg|℃|%|型号|参数|规格", re.IGNORECASE)
PROCEDURE_MARKERS = (
    "流程",
    "步骤",
    "操作",
    "使用",
    "安装",
    "维护",
    "处理",
    "执行",
    "应当",
    "需要",
    "不得",
)
STRUCTURED_MARKERS = (
    "序号",
    "字段",
    "工作表",
    "表格",
    "列名",
    "行数",
    "SQL",
    "型号",
    "参数",
    "报价",
)
LOW_VALUE_MARKERS = (
    "关注公众号",
    "版权所有",
    "扫码关注",
    "本页完",
    "目录",
)


@dataclass(frozen=True)
class Candidate:
    document_id: str
    document_path: str
    relative_path: str
    chunk_id: str
    chunk_index: int
    page: int | None
    section: str
    evidence: str
    phrase: str


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value.lower())


def _display(value: str) -> str:
    return " ".join(value.split())


def _relative_path(path: str, roots: list[Path]) -> str:
    candidate = Path(path)
    for root in roots:
        try:
            return str(candidate.relative_to(root))
        except ValueError:
            continue
    return str(candidate)


def _segments(text: str) -> list[str]:
    cleaned = clean_display_text(text)
    raw_segments = re.split(r"\n+|(?<=[。！？；;!?])\s*", cleaned)
    segments: list[str] = []
    for raw in raw_segments:
        segment = _display(raw).strip("-—_·•◆■✎ ")
        normalized = _normalized(segment)
        if not 18 <= len(normalized) <= 180:
            continue
        if any(marker in segment for marker in LOW_VALUE_MARKERS):
            continue
        useful_chars = sum(char.isalnum() or "\u4e00" <= char <= "\u9fff" for char in segment)
        if useful_chars / max(len(segment), 1) < 0.68:
            continue
        segments.append(segment)
    return segments


def _unique_phrase(segment: str, document_texts: dict[str, str], document_id: str) -> str | None:
    compact = _display(segment)
    for length in (24, 32, 40, 56, 72):
        if len(compact) < length:
            windows = [compact]
        else:
            offsets = (0, max(0, len(compact) // 3), max(0, len(compact) - length))
            windows = [compact[offset : offset + length] for offset in offsets]
        for window in windows:
            phrase = window.strip("，。；：、,.!?！？:;()（）[]【】 ")
            normalized = _normalized(phrase)
            if len(normalized) < 16:
                continue
            matching_documents = [
                candidate_id
                for candidate_id, text in document_texts.items()
                if normalized in text
            ]
            if matching_documents == [document_id]:
                return phrase
    return None


def _load_candidates(
    *, knowledge_base_id: str, database: str
) -> tuple[list[Candidate], dict[str, dict[str, str]]]:
    settings = Settings(database_url=database)
    with sqlite_connection(settings) as connection:
        roots = [
            Path(row["root_path"])
            for row in connection.execute(
                "SELECT root_path FROM sources WHERE knowledge_base_id = ? ORDER BY root_path",
                (knowledge_base_id,),
            ).fetchall()
        ]
        rows = connection.execute(
            """
            SELECT documents.id AS document_id, documents.canonical_path,
                chunks.id AS chunk_id, chunks.chunk_index, chunks.page_no,
                chunks.section_path, chunks.text
            FROM documents
            JOIN document_versions ON document_versions.id = documents.active_version_id
            JOIN chunks ON chunks.document_version_id = document_versions.id
            WHERE documents.knowledge_base_id = ?
                AND documents.visibility_state = 'visible'
                AND document_versions.state = 'ready'
            ORDER BY documents.canonical_path, chunks.chunk_index
            """,
            (knowledge_base_id,),
        ).fetchall()

    document_chunks: dict[str, list[str]] = defaultdict(list)
    metadata: dict[str, dict[str, str]] = {}
    for row in rows:
        document_chunks[row["document_id"]].append(clean_display_text(str(row["text"])))
        metadata[row["document_id"]] = {
            "document_path": row["canonical_path"],
            "relative_path": _relative_path(row["canonical_path"], roots),
        }
    document_texts = {
        document_id: _normalized("\n".join(chunks))
        for document_id, chunks in document_chunks.items()
    }

    candidates: list[Candidate] = []
    seen_phrases: set[str] = set()
    per_document_count: dict[str, int] = defaultdict(int)
    for row in rows:
        if per_document_count[row["document_id"]] >= 80:
            continue
        for segment in _segments(str(row["text"])):
            phrase = _unique_phrase(segment, document_texts, row["document_id"])
            if phrase is None:
                continue
            phrase_key = _normalized(phrase)
            if phrase_key in seen_phrases:
                continue
            seen_phrases.add(phrase_key)
            item = metadata[row["document_id"]]
            candidates.append(
                Candidate(
                    document_id=row["document_id"],
                    document_path=item["document_path"],
                    relative_path=item["relative_path"],
                    chunk_id=row["chunk_id"],
                    chunk_index=row["chunk_index"],
                    page=row["page_no"],
                    section=row["section_path"] or "",
                    evidence=segment[:500],
                    phrase=phrase,
                )
            )
            per_document_count[row["document_id"]] += 1
            if per_document_count[row["document_id"]] >= 80:
                break
    return candidates, metadata


def _matches_category(candidate: Candidate, category: str) -> bool:
    content = f"{candidate.phrase} {candidate.evidence}"
    if category == "numeric_parameter":
        return bool(NUMERIC_MARKERS.search(content))
    if category == "procedure_process":
        return any(marker in content for marker in PROCEDURE_MARKERS)
    if category == "structured_content":
        return any(marker in content for marker in STRUCTURED_MARKERS)
    return True


def _fuzzy_key(phrase: str) -> str:
    value = phrase
    for stopword in ("如果", "可以", "需要", "进行", "以及", "一个", "这种", "内容"):
        value = value.replace(stopword, "")
    value = re.sub(r"[，。；：、,.!?！？:;()（）]", " ", value)
    words = [word for word in value.split() if word]
    compact = " ".join(words) or phrase
    return compact[:28]


def _question(candidate: Candidate, category: str) -> str:
    templates = {
        "exact_fact": "请检索包含原文“{phrase}”的资料文件，并返回路径。",
        "numeric_parameter": "请定位包含数值或参数“{phrase}”的资料文件，并返回路径。",
        "procedure_process": "请查找说明“{phrase}”相关操作或流程的资料文件，并返回路径。",
        "structured_content": "请查找结构化内容中出现“{phrase}”的资料文件，并返回路径。",
        "fuzzy_semantic": "请检索讨论“{phrase}”相关主题的资料文件，并返回路径。",
    }
    phrase = _fuzzy_key(candidate.phrase) if category == "fuzzy_semantic" else candidate.phrase
    return templates[category].format(phrase=phrase)


def _source(candidate: Candidate) -> dict[str, object]:
    return {
        "document_path": candidate.relative_path,
        "document_id": candidate.document_id,
        "chunk_id": candidate.chunk_id,
        "page": candidate.page,
        "section": candidate.section,
        "evidence_text": candidate.evidence,
    }


def _case(*, index: int, category: str, candidate: Candidate) -> dict[str, object]:
    return {
        "id": f"current-kb-{index:04d}",
        "question": _question(candidate, category),
        "category": category,
        "expected_answer": candidate.relative_path,
        "answer_aliases": [Path(candidate.relative_path).name],
        "relevant_sources": [_source(candidate)],
        "difficulty": "medium" if category == "fuzzy_semantic" else "easy",
        "requires_multiple_sources": False,
        "validated": False,
        "validation_method": "active_chunk_exact_evidence_and_document_uniqueness",
        "notes": (
            "Evidence was read directly from the active SQLite chunk and its normalized "
            "phrase was verified to occur in exactly one active document. Human wording "
            "review is still required for official T09 scoring."
        ),
    }


def build(*, knowledge_base_id: str, database: str) -> tuple[list[dict[str, object]], dict]:
    candidates, metadata = _load_candidates(
        knowledge_base_id=knowledge_base_id,
        database=database,
    )
    by_document: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_document[candidate.document_id].append(candidate)
    if len(by_document) < 2:
        raise ValueError("At least two indexed documents are required")

    cases: list[dict[str, object]] = []
    used_candidates: set[tuple[str, str]] = set()
    document_counts: dict[str, int] = defaultdict(int)
    index = 1

    for document_id, item in sorted(metadata.items(), key=lambda pair: pair[1]["relative_path"]):
        candidate = by_document[document_id][0]
        cases.append(
            {
                "id": f"current-kb-{index:04d}",
                "question": (
                    f"请查找文件名为“{Path(item['relative_path']).name}”的资料，并返回路径。"
                ),
                "category": "filename_lookup",
                "expected_answer": item["relative_path"],
                "answer_aliases": [Path(item["relative_path"]).name],
                "relevant_sources": [_source(candidate)],
                "difficulty": "easy",
                "requires_multiple_sources": False,
                "validated": False,
                "validation_method": "active_document_metadata",
                "notes": (
                    "Filename and document ID were verified against the active corpus. "
                    "Human wording review is still required for official T09 scoring."
                ),
            }
        )
        index += 1
        document_counts[document_id] += 1
        used_candidates.add((candidate.document_id, candidate.chunk_id))

    selection_order = (
        "structured_content",
        "numeric_parameter",
        "procedure_process",
        "fuzzy_semantic",
        "exact_fact",
    )
    for category in selection_order:
        required = CATEGORY_QUOTAS[category]
        selected = 0
        while selected < required:
            choice: Candidate | None = None
            for document_id in sorted(
                by_document,
                key=lambda value: (document_counts[value], metadata[value]["relative_path"]),
            ):
                if document_counts[document_id] >= MAX_CASES_PER_DOCUMENT:
                    continue
                choice = next(
                    (
                        candidate
                        for candidate in by_document[document_id]
                        if (candidate.document_id, candidate.chunk_id) not in used_candidates
                        and _matches_category(candidate, category)
                    ),
                    None,
                )
                if choice is not None:
                    break
            if choice is None:
                raise ValueError(
                    f"Insufficient unique candidates for {category}: selected {selected}/{required}"
                )
            cases.append(_case(index=index, category=category, candidate=choice))
            index += 1
            selected += 1
            document_counts[choice.document_id] += 1
            used_candidates.add((choice.document_id, choice.chunk_id))

    if len(cases) != CASE_COUNT:
        raise AssertionError(f"Expected {CASE_COUNT} cases, built {len(cases)}")
    summary = {
        "knowledge_base_id": knowledge_base_id,
        "case_count": len(cases),
        "document_count": len(metadata),
        "documents_used": len(
            {
                source["document_id"]
                for case in cases
                for source in case["relevant_sources"]
            }
        ),
        "category_counts": {
            category: sum(case["category"] == category for case in cases)
            for category in CATEGORY_QUOTAS
        },
        "seed": SEED,
        "validation_method": "active_chunk_exact_evidence_and_document_uniqueness",
        "t09_official": False,
        "official_blocker": "Human review of all 100 generated questions is pending.",
    }
    return cases, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge-base-id", required=True)
    parser.add_argument("--database", default="data/agent.db")
    parser.add_argument("--output", default="evaluation/dataset.jsonl")
    parser.add_argument("--summary-output", default="evaluation/dataset_summary.json")
    args = parser.parse_args()
    cases, summary = build(knowledge_base_id=args.knowledge_base_id, database=args.database)
    output = Path(args.output)
    with output.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    Path(args.summary_output).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({**summary, "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
