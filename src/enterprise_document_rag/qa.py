import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from .config import Settings
from .retrieval import HybridRetriever, SearchResult


@dataclass(frozen=True)
class Citation:
    citation_id: int
    document_id: str
    file_name: str
    canonical_path: str
    page_no: int | None
    section_path: str | None
    quote: str
    chunk_id: str
    bbox: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class Answer:
    answer: str
    confidence: str
    insufficient_evidence: bool
    citations: tuple[Citation, ...]


class LLMProvider(Protocol):
    def answer_from_evidence(self, *, question: str, evidence: list[SearchResult]) -> str: ...


class ExtractiveLLMProvider:
    """Evidence-only generator for the MVP path."""

    def answer_from_evidence(self, *, question: str, evidence: list[SearchResult]) -> str:
        del question
        best = evidence[0]
        return f"{_first_sentence(best.quote)} [1]"


class LocalQwenProvider:
    """Local CPU Qwen generator constrained to retrieved evidence."""

    def __init__(self, *, model_id: str, max_new_tokens: int) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def answer_from_evidence(self, *, question: str, evidence: list[SearchResult]) -> str:
        model, tokenizer = self._get_model()
        context = "\n\n".join(
            f"[{index}] {item.quote}" for index, item in enumerate(evidence, start=1)
        )
        messages = [
            {
                "role": "system",
                "content": "仅依据给定证据回答；证据不足时回答‘证据不足’；每个结论附引用编号。",
            },
            {"role": "user", "content": f"问题：{question}\n\n证据：\n{context}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated_tokens = output[0][inputs["input_ids"].shape[1] :]
        generated = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        answer = generated.strip()
        return answer if answer and "[" in answer else ExtractiveLLMProvider().answer_from_evidence(
            question=question,
            evidence=evidence,
        )

    def _get_model(self):
        if self._model is None or self._tokenizer is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id)
            self._model.eval()
        return self._model, self._tokenizer


@lru_cache
def _qwen_provider(model_id: str, max_new_tokens: int) -> LocalQwenProvider:
    return LocalQwenProvider(model_id=model_id, max_new_tokens=max_new_tokens)


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_backend == "extractive":
        return ExtractiveLLMProvider()
    if settings.llm_backend == "qwen_transformers":
        return _qwen_provider(settings.llm_model_id, settings.llm_max_new_tokens)
    raise ValueError(f"unsupported LLM backend: {settings.llm_backend}")


class RAGAnswerer:
    def __init__(
        self,
        *,
        connection: sqlite3.Connection,
        retriever: HybridRetriever,
        llm_provider: LLMProvider | None = None,
        final_top_k: int = 6,
    ) -> None:
        self.connection = connection
        self.retriever = retriever
        self.llm_provider = llm_provider or ExtractiveLLMProvider()
        self.final_top_k = final_top_k

    def answer(
        self,
        *,
        knowledge_base_id: str,
        question: str,
        allowed_document_ids: set[str] | None = None,
    ) -> Answer:
        evidence = self.retriever.search(
            knowledge_base_id=knowledge_base_id,
            query=question,
            allowed_document_ids=allowed_document_ids,
        )[: self.final_top_k]
        if not self._has_sufficient_evidence(question=question, evidence=evidence):
            return Answer(
                answer="在已授权资料中未找到足够的相关内容，无法可靠作答。",
                confidence="low",
                insufficient_evidence=True,
                citations=(),
            )
        citations = tuple(
            Citation(
                citation_id=index,
                document_id=result.document_id,
                file_name=result.file_name,
                canonical_path=result.canonical_path,
                page_no=result.page_no,
                section_path=result.section_path,
                quote=_short_quote(result.quote),
                chunk_id=result.chunk_id,
                bbox=result.bbox,
            )
            for index, result in enumerate(evidence, start=1)
        )
        self._validate_citations(knowledge_base_id=knowledge_base_id, citations=citations)
        table_total_answer = _answer_from_table_total(question=question, evidence=evidence)
        return Answer(
            answer=table_total_answer
            or self.llm_provider.answer_from_evidence(question=question, evidence=evidence),
            confidence="medium" if len(citations) == 1 else "high",
            insufficient_evidence=False,
            citations=citations,
        )

    def _has_sufficient_evidence(self, *, question: str, evidence: list[SearchResult]) -> bool:
        if not evidence:
            return False
        if "fts" in evidence[0].sources:
            return True
        question_terms = _content_terms(question)
        if not question_terms:
            return False
        evidence_terms = _content_terms(" ".join(item.quote for item in evidence[:2]))
        overlap = question_terms & evidence_terms
        return len(overlap) >= 1

    def _validate_citations(
        self,
        *,
        knowledge_base_id: str,
        citations: tuple[Citation, ...],
    ) -> None:
        for citation in citations:
            row = self.connection.execute(
                """
                SELECT chunks.id
                FROM chunks
                JOIN document_versions ON document_versions.id = chunks.document_version_id
                JOIN documents ON documents.active_version_id = document_versions.id
                WHERE documents.knowledge_base_id = ?
                    AND documents.visibility_state = 'visible'
                    AND documents.id = ?
                    AND chunks.id = ?
                """,
                (knowledge_base_id, citation.document_id, citation.chunk_id),
            ).fetchone()
            if row is None:
                raise ValueError(f"invalid or unauthorized citation: {citation.chunk_id}")


def _first_sentence(text: str) -> str:
    normalized = " ".join(text.split())
    parts = re.split(r"(?<=[。.!?])\s+", normalized, maxsplit=1)
    return _short_quote(parts[0])


def _short_quote(text: str, *, max_chars: int = 240) -> str:
    total_line = next((line for line in text.splitlines() if "序号：合计" in line), None)
    if total_line is not None:
        return total_line
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1].rstrip() + "..."


def _answer_from_table_total(*, question: str, evidence: list[SearchResult]) -> str | None:
    if not any(term in question for term in ("总价", "总和", "合计", "总额", "总报价")):
        return None
    for citation_id, result in enumerate(evidence, start=1):
        total_line = next(
            (line for line in result.quote.splitlines() if "序号：合计" in line),
            None,
        )
        if total_line is None:
            continue
        total_price = _excel_calculated_value(label="总价", text=total_line)
        hospital_total = _excel_calculated_value(label="医院总价", text=total_line)
        values = []
        if total_price is not None:
            values.append(f"总价为 {total_price}")
        if hospital_total is not None:
            values.append(f"医院总价为 {hospital_total}")
        if values:
            return f"根据《{result.file_name}》的合计行，{'，'.join(values)}。[{citation_id}]"
    return None


def _excel_calculated_value(*, label: str, text: str) -> str | None:
    match = re.search(
        rf"{re.escape(label)}：[^|\n]*?计算值：([0-9][0-9,]*(?:\.[0-9]+)?)",
        text,
    )
    return match.group(1) if match else None


def _content_terms(text: str) -> set[str]:
    terms = {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_.-]*", text.lower())
        if len(token) >= 2
    }
    for sequence in re.findall(r"[\u4e00-\u9fff]+", text):
        terms.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    stopwords = {
        "what",
        "which",
        "where",
        "when",
        "who",
        "the",
        "and",
        "for",
        "with",
        "请问",
        "什么",
        "如何",
        "哪些",
        "是否",
        "关键",
        "要求",
        "关于",
        "多少",
    }
    return terms - stopwords
