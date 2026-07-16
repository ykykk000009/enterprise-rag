"""Build low-overlap questions from LLM-extracted short key terms."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_gold_benchmark import Candidate, _load_candidates, _matches_category, _source
from build_low_overlap_benchmark import COMMON_TERMS, TEMPLATES
from build_no_leak_benchmark import DraftCase, _leakage_reasons

from enterprise_document_rag.config import Settings, configure_huggingface_cache

TARGET_QUOTAS = {
    "conceptual_fact": 16,
    "numeric_parameter": 20,
    "procedure_process": 20,
    "structured_content": 10,
    "fuzzy_semantic": 34,
}

OVERSAMPLE = {
    "conceptual_fact": 50,
    "numeric_parameter": 40,
    "procedure_process": 40,
    "structured_content": 25,
    "fuzzy_semantic": 45,
}


@dataclass(frozen=True)
class TermDraft:
    id: str
    category: str
    candidate: Candidate


class LocalTermGenerator:
    def __init__(self, settings: Settings) -> None:
        configure_huggingface_cache(settings)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(settings.llm_model_id)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(settings.llm_model_id)
        self.model.eval()
        self.last_raw_generation = ""

    def generate(self, drafts: list[TermDraft]) -> dict[str, tuple[str, str]]:
        instruction = (
            "从中文中提取两个自然、具体、能代表主题的关键词。每个关键词必须是原文中"
            "连续出现的2至6个汉字，优先选择概念、对象、操作或业务名词；不要选一个、"
            "可以、进行、相关、内容、问题等泛词，不要数字。严格输出"
            "ID|关键词1|关键词2，不要解释。"
        )
        prompts = []
        for draft in drafts:
            messages = [
                {"role": "system", "content": instruction},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"id": draft.id, "text": draft.candidate.evidence[:240]},
                        ensure_ascii=False,
                    ),
                },
            ]
            prompts.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        output = self.model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=False,
        )
        input_length = inputs["input_ids"].shape[1]
        generated_rows = self.tokenizer.batch_decode(
            output[:, input_length:],
            skip_special_tokens=True,
        )
        self.last_raw_generation = "\n".join(generated_rows)
        return _parse_terms(self.last_raw_generation, {draft.id for draft in drafts})


def _parse_terms(generated: str, expected_ids: set[str]) -> dict[str, tuple[str, str]]:
    parsed: dict[str, tuple[str, str]] = {}
    for line in generated.splitlines():
        cleaned = line.strip().strip("`*- ")
        match = re.match(
            r"^\|?\s*(term-\d{4})\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|?$",
            cleaned,
        )
        if match and match.group(1) in expected_ids:
            parsed[match.group(1)] = (match.group(2).strip(), match.group(3).strip())
    return parsed


def _valid_terms(terms: tuple[str, str], evidence: str) -> bool:
    if terms[0] == terms[1] or terms[0] in terms[1] or terms[1] in terms[0]:
        return False
    for term in terms:
        if not re.fullmatch(r"[\u4e00-\u9fff]{2,6}", term):
            return False
        if term not in evidence or term in COMMON_TERMS:
            return False
        if any(common in term for common in COMMON_TERMS if len(common) >= 3):
            return False
    return True


def _select_drafts(candidates: list[Candidate]) -> list[TermDraft]:
    selected: list[TermDraft] = []
    index = 1
    for category in (
        "structured_content",
        "numeric_parameter",
        "procedure_process",
        "fuzzy_semantic",
        "conceptual_fact",
    ):
        used: set[tuple[str, str]] = set()
        document_counts: dict[str, int] = defaultdict(int)
        count = 0
        for candidate in candidates:
            if count >= OVERSAMPLE[category]:
                break
            key = (candidate.document_id, candidate.chunk_id)
            if key in used or document_counts[candidate.document_id] >= 16:
                continue
            if len(re.findall(r"[\u4e00-\u9fff]", candidate.evidence)) < 24:
                continue
            if category not in {"conceptual_fact", "fuzzy_semantic"} and not _matches_category(
                candidate, category
            ):
                continue
            selected.append(TermDraft(f"term-{index:04d}", category, candidate))
            used.add(key)
            document_counts[candidate.document_id] += 1
            count += 1
            index += 1
        if count < TARGET_QUOTAS[category]:
            raise ValueError(f"Insufficient candidates for {category}: {count}")
    return selected


def build(
    *, knowledge_base_id: str, database: str, batch_size: int
) -> tuple[list[dict], dict]:
    candidates, _ = _load_candidates(
        knowledge_base_id=knowledge_base_id,
        database=database,
    )
    drafts = _select_drafts(candidates)
    generator = LocalTermGenerator(Settings(database_url=database))
    extracted: dict[str, tuple[str, str]] = {}
    for start in range(0, len(drafts), batch_size):
        batch = drafts[start : start + batch_size]
        extracted.update(generator.generate(batch))
        print(f"extracted {min(start + batch_size, len(drafts))}/{len(drafts)}", flush=True)

    accepted_counts: dict[str, int] = defaultdict(int)
    cases: list[dict] = []
    for draft in drafts:
        if accepted_counts[draft.category] >= TARGET_QUOTAS[draft.category]:
            continue
        terms = extracted.get(draft.id)
        if terms is None or not _valid_terms(terms, draft.candidate.evidence):
            continue
        question = TEMPLATES[draft.category].format(first=terms[0], second=terms[1])
        gate_draft = DraftCase(
            id=f"no-leak-{len(cases) + 1:04d}",
            category=draft.category,
            candidate=draft.candidate,
        )
        if _leakage_reasons(question, gate_draft):
            continue
        candidate = draft.candidate
        cases.append(
            {
                "id": gate_draft.id,
                "question": question,
                "category": draft.category,
                "expected_answer": candidate.relative_path,
                "answer_aliases": [Path(candidate.relative_path).name],
                "relevant_sources": [_source(candidate)],
                "difficulty": "hard" if draft.category == "fuzzy_semantic" else "medium",
                "requires_multiple_sources": False,
                "validated": False,
                "validation_method": "llm_keyterms_with_low_overlap_gate",
                "leakage_gate": {
                    "max_longest_common_substring_chars": 11,
                    "max_question_evidence_bigram_coverage": 0.649999,
                    "target_filename_allowed": False,
                    "answer_number_allowed": False,
                },
                "notes": "Human semantic relevance review is required before Gold scoring.",
            }
        )
        accepted_counts[draft.category] += 1

    missing = {
        category: TARGET_QUOTAS[category] - accepted_counts[category]
        for category in TARGET_QUOTAS
        if accepted_counts[category] < TARGET_QUOTAS[category]
    }
    if len(cases) < 80:
        raise ValueError(f"Only {len(cases)} valid key-term questions: {missing}")
    summary = {
        "knowledge_base_id": knowledge_base_id,
        "case_count": len(cases),
        "documents_used": len(
            {source["document_id"] for case in cases for source in case["relevant_sources"]}
        ),
        "category_counts": dict(accepted_counts),
        "validated": False,
        "leakage_gate_passed": True,
        "human_review_pending": True,
        "construction": "llm_keyterms_with_low_overlap_gate",
        "target_case_count": sum(TARGET_QUOTAS.values()),
        "missing_after_quality_gate": missing,
    }
    return cases, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge-base-id", required=True)
    parser.add_argument("--database", default="data/agent.db")
    parser.add_argument("--output", default="evaluation/dataset_no_leak.jsonl")
    parser.add_argument("--summary-output", default="evaluation/dataset_no_leak_summary.json")
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    cases, summary = build(
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
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
