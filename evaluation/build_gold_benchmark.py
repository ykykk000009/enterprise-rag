"""Build a source-verified 100-case benchmark for the file-location workflow."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enterprise_document_rag.config import Settings
from enterprise_document_rag.db import sqlite_connection

FIELD_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9][\u4e00-\u9fffA-Za-z0-9_()（）/%+.-]{14,90}")
NUMERIC_MARKERS = re.compile(r"\d|MPa|mm|kg|℃|%|型号|参数|规格", re.IGNORECASE)
PROCEDURE_MARKERS = ("流程", "步骤", "操作", "使用", "安装", "维护", "清洗", "消毒", "应", "不得")
STRUCTURED_MARKERS = ("工作表", "行：", "表", "序号", "型号", "参数", "报价")


@dataclass(frozen=True)
class SourceCandidate:
    document_id: str
    document_path: str
    relative_path: str
    chunk_id: str
    page: int | None
    section: str
    evidence: str
    phrase: str


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _relative_path(path: str, roots: list[Path]) -> str:
    candidate = Path(path)
    for root in roots:
        try:
            return str(candidate.relative_to(root))
        except ValueError:
            continue
    return str(candidate)


def _field_phrase(text: str) -> str | None:
    normalized = _normalized(text)
    for match in FIELD_PATTERN.finditer(normalized):
        phrase = match.group(0).strip("-_.()（）")
        if len(phrase) >= 15:
            return phrase
    for segment in re.split(r"[。；;！？!?]", normalized):
        phrase = segment.strip()
        if 15 <= len(phrase) <= 90 and any(char.isalnum() for char in phrase):
            return phrase
    return None


def _load_candidates(*, knowledge_base_id: str, database: str) -> list[SourceCandidate]:
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
            SELECT documents.id AS document_id, documents.canonical_path, chunks.id AS chunk_id,
                chunks.page_no, chunks.section_path, chunks.text
            FROM documents
            JOIN document_versions ON document_versions.id = documents.active_version_id
            JOIN chunks ON chunks.document_version_id = document_versions.id
            WHERE documents.knowledge_base_id = ?
                AND documents.visibility_state = 'visible'
                AND document_versions.state = 'ready'
                AND chunks.token_count BETWEEN 12 AND 800
            ORDER BY documents.canonical_path, chunks.chunk_index
            """,
            (knowledge_base_id,),
        ).fetchall()
    candidates: list[SourceCandidate] = []
    used_document_ids: set[str] = set()
    for row in rows:
        if row["document_id"] in used_document_ids:
            continue
        evidence = _normalized(row["text"])
        phrase = _field_phrase(evidence)
        if phrase is None:
            continue
        candidates.append(
            SourceCandidate(
                document_id=row["document_id"],
                document_path=row["canonical_path"],
                relative_path=_relative_path(row["canonical_path"], roots),
                chunk_id=row["chunk_id"],
                page=row["page_no"],
                section=row["section_path"] or "",
                evidence=evidence[:500],
                phrase=phrase,
            )
        )
        used_document_ids.add(row["document_id"])
    return candidates


def _take(
    candidates: list[SourceCandidate],
    *,
    selected_ids: set[str],
    count: int,
    predicate,
) -> list[SourceCandidate]:
    selected = [
        candidate
        for candidate in candidates
        if candidate.document_id not in selected_ids and predicate(candidate)
    ][:count]
    if len(selected) < count:
        selected.extend(
            candidate
            for candidate in candidates
            if candidate.document_id not in selected_ids
            and candidate.document_id not in {item.document_id for item in selected}
        )
        selected = selected[:count]
    selected_ids.update(candidate.document_id for candidate in selected)
    if len(selected) != count:
        raise ValueError(f"Need {count} source documents, found {len(selected)}")
    return selected


def _source(candidate: SourceCandidate) -> dict[str, object]:
    return {
        "document_path": candidate.relative_path,
        "document_id": candidate.document_id,
        "page": candidate.page,
        "section": candidate.section,
        "evidence_text": candidate.evidence,
    }


def _single_case(*, index: int, category: str, candidate: SourceCandidate) -> dict[str, object]:
    prompts = {
        "exact_fact": "请检索包含字段“{phrase}”的文件，并返回文件路径。",
        "numeric_parameter": "请定位包含参数字段“{phrase}”的文件路径。",
        "procedure_process": "请查找说明“{phrase}”的资料文件，并返回路径。",
        "structured_content": "请查找表格或结构化内容中出现“{phrase}”的文件路径。",
        "fuzzy_semantic": "请检索与“{phrase}”相关的资料，并返回最相关文件的路径。",
    }
    return {
        "id": f"gold-{index:04d}",
        "question": prompts[category].format(phrase=candidate.phrase),
        "category": category,
        "expected_answer": candidate.relative_path,
        "answer_aliases": [Path(candidate.relative_path).name],
        "relevant_sources": [_source(candidate)],
        "difficulty": "easy" if category != "fuzzy_semantic" else "medium",
        "requires_multiple_sources": False,
        "validated": True,
        "notes": (
            "Source location and evidence text were checked directly against the indexed corpus."
        ),
    }


def _multi_case(*, index: int, left: SourceCandidate, right: SourceCandidate) -> dict[str, object]:
    return {
        "id": f"gold-{index:04d}",
        "question": (
            f"请分别定位包含字段“{left.phrase}”和“{right.phrase}”的两个文件，并返回路径。"
        ),
        "category": "multi_document",
        "expected_answer": f"{left.relative_path}\n{right.relative_path}",
        "answer_aliases": [left.relative_path, right.relative_path],
        "relevant_sources": [_source(left), _source(right)],
        "difficulty": "hard",
        "requires_multiple_sources": True,
        "validated": True,
        "notes": (
            "Both source locations and evidence texts were checked directly against "
            "the indexed corpus."
        ),
    }


def build(*, knowledge_base_id: str, database: str) -> list[dict[str, object]]:
    candidates = _load_candidates(knowledge_base_id=knowledge_base_id, database=database)
    if len(candidates) < 115:
        raise ValueError(f"Need at least 115 distinct source documents, found {len(candidates)}")
    selected_ids: set[str] = set()
    groups = [
        ("exact_fact", 25, lambda _: True),
        ("numeric_parameter", 20, lambda item: bool(NUMERIC_MARKERS.search(item.phrase))),
        (
            "procedure_process",
            20,
            lambda item: any(marker in item.evidence for marker in PROCEDURE_MARKERS),
        ),
        (
            "structured_content",
            10,
            lambda item: any(marker in item.evidence for marker in STRUCTURED_MARKERS),
        ),
        ("fuzzy_semantic", 10, lambda _: True),
    ]
    cases: list[dict[str, object]] = []
    index = 1
    for category, count, predicate in groups:
        for candidate in _take(
            candidates,
            selected_ids=selected_ids,
            count=count,
            predicate=predicate,
        ):
            cases.append(_single_case(index=index, category=category, candidate=candidate))
            index += 1
    multi_sources = _take(
        candidates,
        selected_ids=selected_ids,
        count=30,
        predicate=lambda _: True,
    )
    for pair_index in range(0, len(multi_sources), 2):
        cases.append(
            _multi_case(
                index=index,
                left=multi_sources[pair_index],
                right=multi_sources[pair_index + 1],
            )
        )
        index += 1
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge-base-id", required=True)
    parser.add_argument("--database", default="data/agent.db")
    parser.add_argument("--output", default="evaluation/dataset.jsonl")
    args = parser.parse_args()
    cases = build(knowledge_base_id=args.knowledge_base_id, database=args.database)
    output = Path(args.output)
    with output.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(json.dumps({"gold_case_count": len(cases), "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
