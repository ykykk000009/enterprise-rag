"""Audit retrieval benchmark leakage and lexical shortcut risk."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value)


def _quoted_payload(question: str) -> str:
    match = re.search(r"[“\"]([^”\"]+)[”\"]", question)
    return match.group(1).strip() if match else ""


def _ngrams(value: str, size: int = 2) -> set[str]:
    normalized = _normalize(value)
    if len(normalized) < size:
        return {normalized} if normalized else set()
    return {normalized[index : index + size] for index in range(len(normalized) - size + 1)}


def _coverage(source: str, target: str) -> float:
    source_grams = _ngrams(source)
    if not source_grams:
        return 0.0
    return len(source_grams & _ngrams(target)) / len(source_grams)


def _longest_common_substring_length(left: str, right: str) -> int:
    left_normalized = _normalize(left)
    right_normalized = _normalize(right)
    if not left_normalized or not right_normalized:
        return 0
    previous = [0] * (len(right_normalized) + 1)
    longest = 0
    for left_char in left_normalized:
        current = [0]
        for index, right_char in enumerate(right_normalized, start=1):
            value = previous[index - 1] + 1 if left_char == right_char else 0
            current.append(value)
            longest = max(longest, value)
        previous = current
    return longest


def audit(*, dataset: Path, retrieval_results: Path) -> dict:
    cases = {
        case["id"]: case
        for case in (
            json.loads(line)
            for line in dataset.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    rows = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in retrieval_results.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    category_rows: dict[str, list[dict]] = defaultdict(list)
    method_counts: Counter[str] = Counter()
    case_audits: list[dict] = []

    for case_id, case in cases.items():
        source = case["relevant_sources"][0]
        evidence = str(source.get("evidence_text", ""))
        payload = _quoted_payload(case["question"])
        filename = Path(str(source.get("document_path", ""))).name
        payload_normalized = _normalize(payload)
        evidence_normalized = _normalize(evidence)
        filename_normalized = _normalize(filename)
        direct_evidence_leak = (
            bool(payload_normalized) and payload_normalized in evidence_normalized
        )
        filename_leak = bool(filename_normalized) and filename_normalized in _normalize(
            case["question"]
        )
        payload_coverage = round(_coverage(payload, evidence), 6)
        question_coverage = round(_coverage(case["question"], evidence), 6)
        longest_common_substring = _longest_common_substring_length(
            case["question"],
            evidence,
        )
        result = rows.get(case_id, {})
        rank1 = (result.get("results") or [{}])[0]
        methods = rank1.get("retrieval_method") or []
        for method in methods:
            method_counts[str(method)] += 1
        audited = {
            "id": case_id,
            "category": case.get("category", "unknown"),
            "direct_evidence_leak": direct_evidence_leak,
            "filename_leak": filename_leak,
            "payload_evidence_bigram_coverage": payload_coverage,
            "question_evidence_bigram_coverage": question_coverage,
            "longest_common_substring_chars": longest_common_substring,
            "rank1_methods": methods,
            "rank1_relevant": bool((result.get("relevance") or [False])[0]),
        }
        case_audits.append(audited)
        category_rows[audited["category"]].append(audited)

    def summarize(items: list[dict]) -> dict:
        count = len(items)
        return {
            "case_count": count,
            "direct_evidence_leak_rate": round(
                sum(item["direct_evidence_leak"] for item in items) / count, 6
            )
            if count
            else 0.0,
            "filename_leak_rate": round(sum(item["filename_leak"] for item in items) / count, 6)
            if count
            else 0.0,
            "mean_payload_evidence_bigram_coverage": round(
                sum(item["payload_evidence_bigram_coverage"] for item in items) / count,
                6,
            )
            if count
            else 0.0,
            "mean_question_evidence_bigram_coverage": round(
                sum(item["question_evidence_bigram_coverage"] for item in items) / count,
                6,
            )
            if count
            else 0.0,
            "max_longest_common_substring_chars": max(
                (item["longest_common_substring_chars"] for item in items),
                default=0,
            ),
            "rank1_relevance_rate": round(sum(item["rank1_relevant"] for item in items) / count, 6)
            if count
            else 0.0,
        }

    overall = summarize(case_audits)
    shortcut_rate = round(
        sum(
            item["direct_evidence_leak"]
            or item["filename_leak"]
            or item["payload_evidence_bigram_coverage"] >= 0.8
            or item["question_evidence_bigram_coverage"] >= 0.65
            or item["longest_common_substring_chars"] >= 12
            for item in case_audits
        )
        / len(case_audits),
        6,
    )
    return {
        "status": "LEAKAGE_CONFIRMED" if shortcut_rate >= 0.5 else "NO_MATERIAL_LEAKAGE_FOUND",
        "overall": overall,
        "lexical_shortcut_case_rate": shortcut_rate,
        "rank1_method_case_counts": dict(sorted(method_counts.items())),
        "by_category": {
            category: summarize(items) for category, items in sorted(category_rows.items())
        },
        "case_audits": case_audits,
        "interpretation": (
            "Questions containing source filenames, verbatim evidence, or near-verbatim "
            "payloads measure lexical source lookup rather than held-out semantic retrieval."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evaluation/dataset.jsonl")
    parser.add_argument(
        "--retrieval-results",
        default="evaluation/results/retrieval_results.jsonl",
    )
    parser.add_argument("--output", default="evaluation/results/leakage_audit.json")
    args = parser.parse_args()
    result = audit(
        dataset=Path(args.dataset),
        retrieval_results=Path(args.retrieval_results),
    )
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {key: value for key, value in result.items() if key != "case_audits"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
