"""Build a 100-case low-overlap retrieval diagnostic without an LLM."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_gold_benchmark import Candidate, _load_candidates, _matches_category, _source
from build_no_leak_benchmark import CATEGORY_QUOTAS, DraftCase, _leakage_reasons

MAX_CASES_PER_DOCUMENT = 8
COMMON_TERMS = {
    "一个",
    "一种",
    "一些",
    "这个",
    "这些",
    "可以",
    "进行",
    "通过",
    "需要",
    "能够",
    "相关",
    "内容",
    "问题",
    "情况",
    "主要",
    "实现",
    "使用",
    "对于",
    "以及",
    "如果",
    "没有",
    "文件",
    "路径",
    "压缩",
}

TEMPLATES = {
    "conceptual_fact": "结合{first}与{second}两个线索，这里表达的核心认识是什么？",
    "numeric_parameter": "在{first}相关配置中，{second}对应的具体数值或规格是多少？",
    "procedure_process": "处理{first}事项时，{second}环节应采取怎样的做法？",
    "structured_content": "涉及{first}的数据记录中，应如何理解{second}相关字段？",
    "fuzzy_semantic": "从{first}和{second}的关联来看，这段内容主要说明什么问题？",
}


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value.lower())


def _term_document_frequency(candidates: list[Candidate]) -> dict[str, int]:
    document_texts: dict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        document_texts[candidate.document_id].append(_normalized(candidate.evidence))
    joined = {document_id: "".join(parts) for document_id, parts in document_texts.items()}
    frequencies: dict[str, int] = defaultdict(int)
    all_terms: set[str] = set()
    for candidate in candidates:
        all_terms.update(_candidate_terms(candidate.evidence))
    for term in all_terms:
        frequencies[term] = sum(term in text for text in joined.values())
    return dict(frequencies)


def _candidate_terms(evidence: str) -> set[str]:
    terms: set[str] = set()
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", evidence):
        for length in (4, 3, 2):
            for start in range(0, len(run) - length + 1):
                term = run[start : start + length]
                if term in COMMON_TERMS:
                    continue
                if any(common in term for common in COMMON_TERMS if len(common) >= 3):
                    continue
                terms.add(term)
    return terms


def _select_terms(candidate: Candidate, frequencies: dict[str, int]) -> tuple[str, str] | None:
    terms = sorted(
        _candidate_terms(candidate.evidence),
        key=lambda term: (
            frequencies.get(term, 99) == 1,
            abs(frequencies.get(term, 99) - 3),
            -len(term),
            term,
        ),
    )
    selected: list[str] = []
    for term in terms:
        frequency = frequencies.get(term, 99)
        if frequency > 8:
            continue
        if any(term in chosen or chosen in term for chosen in selected):
            continue
        selected.append(term)
        if len(selected) == 2:
            return selected[0], selected[1]
    return None


def _make_question(category: str, terms: tuple[str, str]) -> str:
    return TEMPLATES[category].format(first=terms[0], second=terms[1])


def build(*, knowledge_base_id: str, database: str) -> tuple[list[dict], dict]:
    candidates, _ = _load_candidates(
        knowledge_base_id=knowledge_base_id,
        database=database,
    )
    frequencies = _term_document_frequency(candidates)
    used: set[tuple[str, str]] = set()
    document_counts: dict[str, int] = defaultdict(int)
    cases: list[dict] = []
    case_index = 1
    category_order = (
        "structured_content",
        "numeric_parameter",
        "procedure_process",
        "fuzzy_semantic",
        "conceptual_fact",
    )
    for category in category_order:
        accepted = 0
        for candidate in candidates:
            if accepted >= CATEGORY_QUOTAS[category]:
                break
            key = (candidate.document_id, candidate.chunk_id)
            if key in used or document_counts[candidate.document_id] >= MAX_CASES_PER_DOCUMENT:
                continue
            if category not in {"conceptual_fact", "fuzzy_semantic"} and not _matches_category(
                candidate, category
            ):
                continue
            terms = _select_terms(candidate, frequencies)
            if terms is None:
                continue
            question = _make_question(category, terms)
            draft = DraftCase(
                id=f"no-leak-{case_index:04d}",
                category=category,
                candidate=candidate,
            )
            reasons = _leakage_reasons(question, draft)
            if reasons:
                continue
            cases.append(
                {
                    "id": draft.id,
                    "question": question,
                    "category": category,
                    "expected_answer": candidate.relative_path,
                    "answer_aliases": [Path(candidate.relative_path).name],
                    "relevant_sources": [_source(candidate)],
                    "difficulty": "hard" if category == "fuzzy_semantic" else "medium",
                    "requires_multiple_sources": False,
                    "validated": False,
                    "validation_method": "automatic_low_overlap_keyword_diagnostic",
                    "leakage_gate": {
                        "max_longest_common_substring_chars": 11,
                        "max_question_evidence_bigram_coverage": 0.649999,
                        "target_filename_allowed": False,
                        "answer_number_allowed": False,
                    },
                    "notes": (
                        "Automatic low-overlap diagnostic only; human semantic relevance "
                        "review is required before official Gold scoring."
                    ),
                }
            )
            accepted += 1
            case_index += 1
            used.add(key)
            document_counts[candidate.document_id] += 1
        if accepted != CATEGORY_QUOTAS[category]:
            raise ValueError(f"Only selected {accepted}/{CATEGORY_QUOTAS[category]} for {category}")

    summary = {
        "knowledge_base_id": knowledge_base_id,
        "case_count": len(cases),
        "documents_used": len(
            {source["document_id"] for case in cases for source in case["relevant_sources"]}
        ),
        "category_counts": {
            category: sum(case["category"] == category for case in cases)
            for category in CATEGORY_QUOTAS
        },
        "validated": False,
        "leakage_gate_passed": True,
        "human_review_pending": True,
        "construction": "automatic_low_overlap_keyword_diagnostic",
    }
    return cases, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge-base-id", required=True)
    parser.add_argument("--database", default="data/agent.db")
    parser.add_argument("--output", default="evaluation/dataset_no_leak.jsonl")
    parser.add_argument("--summary-output", default="evaluation/dataset_no_leak_summary.json")
    args = parser.parse_args()
    cases, summary = build(
        knowledge_base_id=args.knowledge_base_id,
        database=args.database,
    )
    with Path(args.output).open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    Path(args.summary_output).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
