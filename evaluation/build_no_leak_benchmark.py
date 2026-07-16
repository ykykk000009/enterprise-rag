"""Build a low-overlap semantic retrieval benchmark from the active corpus."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_dataset_leakage import (
    _coverage,
    _longest_common_substring_length,
    _normalize,
)
from build_gold_benchmark import Candidate, _load_candidates, _matches_category, _source

from enterprise_document_rag.config import Settings, configure_huggingface_cache

CASE_COUNT = 100
MAX_CASES_PER_DOCUMENT = 8
CATEGORY_QUOTAS = {
    "conceptual_fact": 25,
    "numeric_parameter": 20,
    "procedure_process": 20,
    "structured_content": 10,
    "fuzzy_semantic": 25,
}


@dataclass(frozen=True)
class DraftCase:
    id: str
    category: str
    candidate: Candidate


class LocalQuestionGenerator:
    def __init__(self, settings: Settings) -> None:
        configure_huggingface_cache(settings)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(settings.llm_model_id)
        self.model = AutoModelForCausalLM.from_pretrained(settings.llm_model_id)
        self.model.eval()
        self.last_raw_generation = ""

    def generate(
        self,
        cases: list[DraftCase],
        *,
        feedback: dict[str, str] | None = None,
    ) -> dict[str, str]:
        items = []
        for case in cases:
            feedback_text = (feedback or {}).get(case.id, "")
            items.append(
                {
                    "id": case.id,
                    "category": case.category,
                    "section": case.candidate.section,
                    "evidence": case.candidate.evidence[:360],
                    "previous_rejection": feedback_text,
                }
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "你是企业知识库检索测试集设计员。根据证据为每项编写一个自然、独立、"
                    "可由该证据回答的中文问题。问题用于测试语义检索，不是原文查重。禁止"
                    "出现文件名、路径、引号和‘查找包含某原文’等措辞；禁止连续复制证据"
                    "中的六个以上汉字；必须改用同义表达和真实用户问法。数值题询问数值，"
                    "但不要把答案数值写进问题。流程题询问如何做或有哪些要求；结构化题"
                    "询问字段、表格或数据关系；概念题询问含义、原因、作用或区别。每行"
                    "严格输出 ID|问题，不要输出解释、序号、Markdown 或其他内容。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(items, ensure_ascii=False),
            },
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(
            **inputs,
            max_new_tokens=max(80, len(cases) * 24),
            do_sample=False,
        )
        generated_tokens = output[0][inputs["input_ids"].shape[1] :]
        generated = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        self.last_raw_generation = generated
        return _parse_questions(generated, {case.id for case in cases})


def _parse_questions(generated: str, expected_ids: set[str]) -> dict[str, str]:
    questions: dict[str, str] = {}
    for line in generated.splitlines():
        cleaned = line.strip().strip("`*- ")
        match = re.match(
            r"^\|?\s*(no-leak-\d{4})\s*[|｜:：\t]\s*(.+?)\s*\|?$",
            cleaned,
        )
        if match and match.group(1) in expected_ids:
            question = match.group(2).strip().strip('"“”')
            questions[match.group(1)] = question
    return questions


def _select_candidates(*, knowledge_base_id: str, database: str) -> list[DraftCase]:
    candidates, metadata = _load_candidates(
        knowledge_base_id=knowledge_base_id,
        database=database,
    )
    by_document: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_document[candidate.document_id].append(candidate)
    used: set[tuple[str, str]] = set()
    document_counts: dict[str, int] = defaultdict(int)
    selected: list[DraftCase] = []
    index = 1
    category_order = (
        "structured_content",
        "numeric_parameter",
        "procedure_process",
        "fuzzy_semantic",
        "conceptual_fact",
    )
    for category in category_order:
        count = 0
        while count < CATEGORY_QUOTAS[category]:
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
                        if (candidate.document_id, candidate.chunk_id) not in used
                        and (
                            category in {"conceptual_fact", "fuzzy_semantic"}
                            or _matches_category(candidate, category)
                        )
                    ),
                    None,
                )
                if choice is not None:
                    break
            if choice is None:
                raise ValueError(f"Insufficient candidates for {category}: {count}")
            selected.append(
                DraftCase(
                    id=f"no-leak-{index:04d}",
                    category=category,
                    candidate=choice,
                )
            )
            index += 1
            count += 1
            used.add((choice.document_id, choice.chunk_id))
            document_counts[choice.document_id] += 1
    if len(selected) != CASE_COUNT:
        raise AssertionError(f"Expected {CASE_COUNT} cases, selected {len(selected)}")
    return selected


def _leakage_reasons(question: str, case: DraftCase) -> list[str]:
    reasons: list[str] = []
    evidence = case.candidate.evidence
    filename = Path(case.candidate.relative_path).name
    if not 8 <= len(question) <= 100:
        reasons.append("question_length_out_of_range")
    if len(re.findall(r"[\u4e00-\u9fff]", question)) < 4:
        reasons.append("insufficient_chinese_question_text")
    if _normalize(filename) and _normalize(filename) in _normalize(question):
        reasons.append("contains_target_filename")
    longest = _longest_common_substring_length(question, evidence)
    if longest >= 12:
        reasons.append(f"longest_common_substring={longest}")
    coverage = _coverage(question, evidence)
    if coverage >= 0.65:
        reasons.append(f"bigram_coverage={coverage:.3f}")
    if re.search(r"[“”\"]", question):
        reasons.append("contains_quotation")
    if case.category == "numeric_parameter":
        evidence_numbers = set(re.findall(r"\d+(?:\.\d+)?", evidence))
        question_numbers = set(re.findall(r"\d+(?:\.\d+)?", question))
        leaked_values = {
            value
            for value in question_numbers & evidence_numbers
            if not re.fullmatch(r"20\d{2}", value)
        }
        if leaked_values:
            reasons.append("contains_answer_number=" + ",".join(sorted(leaked_values)))
    return reasons


def _build_case(draft: DraftCase, question: str) -> dict[str, object]:
    candidate = draft.candidate
    return {
        "id": draft.id,
        "question": question.rstrip("。") + "？",
        "category": draft.category,
        "expected_answer": candidate.relative_path,
        "answer_aliases": [Path(candidate.relative_path).name],
        "relevant_sources": [_source(candidate)],
        "difficulty": "medium" if draft.category != "fuzzy_semantic" else "hard",
        "requires_multiple_sources": False,
        "validated": False,
        "validation_method": "synthetic_paraphrase_with_lexical_leakage_gate",
        "leakage_gate": {
            "max_longest_common_substring_chars": 11,
            "max_question_evidence_bigram_coverage": 0.649999,
            "target_filename_allowed": False,
        },
        "notes": "Human semantic relevance review is required before official T09 scoring.",
    }


def build(
    *,
    knowledge_base_id: str,
    database: str,
    batch_size: int,
) -> tuple[list[dict[str, object]], dict, list[dict]]:
    drafts = _select_candidates(knowledge_base_id=knowledge_base_id, database=database)
    settings = Settings(database_url=database)
    generator = LocalQuestionGenerator(settings)
    questions: dict[str, str] = {}
    raw_generations: list[dict] = []

    for start in range(0, len(drafts), batch_size):
        batch = drafts[start : start + batch_size]
        generated = generator.generate(batch)
        questions.update(generated)
        raw_generations.append(
            {"round": 1, "ids": [case.id for case in batch], "questions": generated}
        )
        print(f"generated {min(start + batch_size, len(drafts))}/{len(drafts)}", flush=True)

    for retry in range(1, 2):
        failures: list[DraftCase] = []
        feedback: dict[str, str] = {}
        for case in drafts:
            question = questions.get(case.id, "")
            reasons = _leakage_reasons(question, case)
            if reasons:
                failures.append(case)
                feedback[case.id] = "; ".join(reasons)
        if not failures:
            break
        print(f"retry {retry}: {len(failures)} rejected questions", flush=True)
        for start in range(0, len(failures), batch_size):
            batch = failures[start : start + batch_size]
            regenerated = generator.generate(batch, feedback=feedback)
            questions.update(regenerated)
            raw_generations.append(
                {
                    "round": retry + 1,
                    "ids": [case.id for case in batch],
                    "questions": regenerated,
                    "feedback": {case.id: feedback[case.id] for case in batch},
                }
            )

    final_failures = {
        case.id: _leakage_reasons(questions.get(case.id, ""), case) for case in drafts
    }
    final_failures = {case_id: reasons for case_id, reasons in final_failures.items() if reasons}
    if final_failures:
        raise ValueError(
            f"Leakage gate rejected {len(final_failures)} cases after retries: {final_failures}"
        )
    cases = [_build_case(case, questions[case.id]) for case in drafts]
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
    }
    return cases, summary, raw_generations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge-base-id", required=True)
    parser.add_argument("--database", default="data/agent.db")
    parser.add_argument("--output", default="evaluation/dataset_no_leak.jsonl")
    parser.add_argument("--summary-output", default="evaluation/dataset_no_leak_summary.json")
    parser.add_argument(
        "--raw-output",
        default="evaluation/results/no_leak_question_generation.jsonl",
    )
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()
    cases, summary, raw_generations = build(
        knowledge_base_id=args.knowledge_base_id,
        database=args.database,
        batch_size=args.batch_size,
    )
    with Path(args.output).open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    Path(args.summary_output).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with Path(args.raw_output).open("w", encoding="utf-8") as handle:
        for row in raw_generations:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
