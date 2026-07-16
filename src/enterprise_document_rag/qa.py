import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from .config import Settings
from .retrieval import HybridRetriever, SearchResult
from .text_utils import merge_context_texts


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
        context = "\n\n".join(
            f"[{index}] {item.quote}" for index, item in enumerate(evidence, start=1)
        )
        focused_context = "\n\n".join(
            f"[{index}] {_focus_answer_context(question=question, text=item.quote)}"
            for index, item in enumerate(evidence, start=1)
        )
        outlines: list[tuple[int, list[tuple[int, str]]]] = []
        for index, item in enumerate(evidence, start=1):
            outline = _numbered_outline(item.quote)
            if len(outline) >= 2:
                outlines.append((index, outline))
        outline_context = ""
        if _is_collection_question(question):
            outline_context = "\n".join(
                f"[{citation_id}] 原文编号条目："
                + "；".join(f"{number}. {title}" for number, title in outline)
                for citation_id, outline in outlines
            )
        draft_messages = [
            {
                "role": "system",
                "content": (
                    "你是企业文档问答助手。仅依据给定证据回答，证据不足时明确回答"
                    "‘证据不足’。只输出最终答案，根据问题使用简洁段落或编号列表，"
                    "不超过 400 个汉字。不要逐段复述证据，不要抄写文档标题，不要输出"
                    "思考过程，也不要用引用编号开头。每个结论句末使用 [1]、[2] 形式"
                    "标注证据编号。若原文列出多条原则、步骤、条件或结论，答案必须覆盖"
                    "所有条目，不得只回答第一条。每条证据包含命中块及其前后连续文本，"
                    "邻接文本只用于补全跨块内容，不要把相邻的无关小节混入答案。若提供"
                    "‘原文编号条目’，必须逐项回答清单中的每一项，不得减少条目数量。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"问题：{question}\n\n核心证据（优先据此回答）：\n{focused_context}\n\n"
                    f"完整邻接证据（仅用于补全核心证据）：\n{context}\n\n"
                    + (f"结构化提示：\n{outline_context}\n\n" if outline_context else "")
                    + "请概括证据并直接回答，不要重复粘贴上面的证据原文。"
                ),
            },
        ]
        generated = self._generate(draft_messages)
        draft = _finalize_generated_answer(generated, evidence_count=len(evidence))
        if draft is None:
            draft = ExtractiveLLMProvider().answer_from_evidence(
                question=question,
                evidence=evidence,
            )

        review_messages = [
            {
                "role": "system",
                "content": (
                    "你是企业文档答案证据审查器。请仅根据问题和核心证据独立重写最终"
                    "答案。必须完成以下检查：第一，答案是否直接回答问题；"
                    "第二，是否遗漏证据中与问题直接相关的定义、条件、步骤、数值、例外或"
                    "后续内容；第三，是否混入前后邻接块中与问题无关的信息；第四，每个"
                    "结论是否都能由证据支持。删除无关或无依据内容，补齐遗漏，保持简洁。"
                    "不得加入核心证据中没有出现的其他章节、条目或示例。"
                    "只输出最终答案，不要说明审查过程，不要输出思考过程。引用必须使用"
                    "证据编号 [1]、[2]，不得编造编号。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"问题：{question}\n\n核心证据（优先据此审查）：\n{focused_context}\n\n"
                    + (f"结构化提示：\n{outline_context}\n\n" if outline_context else "")
                    + "请独立输出经证据审查后的最终答案。"
                ),
            },
        ]
        reviewed_generated = self._generate(review_messages)
        reviewed = _finalize_generated_answer(
            reviewed_generated,
            evidence_count=len(evidence),
        )
        answer = _choose_reviewed_answer(draft=draft, reviewed=reviewed)
        return _ensure_numbered_outline_coverage(
            question=question,
            answer=answer,
            outlines=outlines,
        )

    def _generate(self, messages: list[dict[str, str]]) -> str:
        model, tokenizer = self._get_model()
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated_tokens = output[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True)

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
        context_radius: int = 2,
    ) -> None:
        if context_radius < 0:
            raise ValueError("context_radius must be non-negative")
        self.connection = connection
        self.retriever = retriever
        self.llm_provider = llm_provider or ExtractiveLLMProvider()
        self.final_top_k = final_top_k
        self.context_radius = context_radius

    def answer(
        self,
        *,
        knowledge_base_id: str,
        question: str,
        allowed_document_ids: set[str] | None = None,
    ) -> Answer:
        evidence = _select_answer_evidence(
            self.retriever.search(
                knowledge_base_id=knowledge_base_id,
                query=question,
                allowed_document_ids=allowed_document_ids,
            )[: self.final_top_k]
        )
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
        generation_evidence = _expand_answer_evidence(
            connection=self.connection,
            knowledge_base_id=knowledge_base_id,
            evidence=evidence,
            radius=self.context_radius,
        )
        table_total_answer = _answer_from_table_total(
            question=question,
            evidence=generation_evidence,
        )
        return Answer(
            answer=table_total_answer
            or self.llm_provider.answer_from_evidence(
                question=question,
                evidence=generation_evidence,
            ),
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


def _finalize_generated_answer(answer: str, *, evidence_count: int) -> str | None:
    cleaned = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.replace("<think>", "").replace("</think>", "").strip()
    cleaned = re.sub(r"^(?:修正后的最终答案|最终答案|答案)\s*[:：]\s*", "", cleaned)
    if not cleaned or evidence_count < 1:
        return None

    valid_citations: list[int] = []

    def normalize_citation(match: re.Match[str]) -> str:
        citation_id = int(match.group(1))
        if 1 <= citation_id <= evidence_count:
            valid_citations.append(citation_id)
            return f"[{citation_id}]"
        return ""

    cleaned = re.sub(r"\[(\d+)\]", normalize_citation, cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned).strip()
    if not cleaned:
        return None
    if not valid_citations:
        cleaned = f"{cleaned.rstrip()} [1]"
    return cleaned


def _choose_reviewed_answer(*, draft: str, reviewed: str | None) -> str:
    if reviewed is None:
        return draft
    if "证据不足" in reviewed and "证据不足" not in draft:
        return draft
    return reviewed


def _focus_answer_context(*, question: str, text: str, max_chars: int = 2200) -> str:
    lines = text.splitlines()
    meaningful = [(index, line.strip()) for index, line in enumerate(lines) if line.strip()]
    if not meaningful:
        return text
    question_terms = _content_terms(question)
    best_index, best_line = max(
        meaningful,
        key=lambda item: _line_relevance_score(
            line=item[1],
            question=question,
            question_terms=question_terms,
        ),
    )
    heading_level = _heading_level(best_line)
    if heading_level is not None:
        end_index = len(lines)
        for index, line in meaningful:
            if index <= best_index:
                continue
            candidate_level = _heading_level(line)
            if candidate_level is not None and candidate_level <= heading_level:
                end_index = index
                break
        focused = "\n".join(lines[best_index:end_index]).strip()
    else:
        paragraph_index = next(
            (
                index
                for index, paragraph in enumerate(re.split(r"\n\s*\n", text))
                if best_line in paragraph
            ),
            0,
        )
        paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
        focused = "\n\n".join(
            paragraphs[max(0, paragraph_index - 1) : paragraph_index + 2]
        )
    return focused if len(focused) <= max_chars else focused[:max_chars].rstrip() + "..."


def _line_relevance_score(*, line: str, question: str, question_terms: set[str]) -> float:
    line_terms = _content_terms(line)
    score = float(len(question_terms & line_terms))
    normalized_line = re.sub(r"\s+", "", line)
    normalized_question = re.sub(r"[\s？?]", "", question)
    for suffix in ("是什么", "有哪些", "包括哪些", "怎么做", "如何"):
        normalized_question = normalized_question.removesuffix(suffix)
    if normalized_line and normalized_line in normalized_question:
        score += len(normalized_line) * 2.0
    return score


def _heading_level(line: str) -> int | None:
    stripped = line.strip()
    if re.match(r"^第[一二三四五六七八九十百0-9]+[章节篇]", stripped):
        return 0
    if re.match(r"^[一二三四五六七八九十]+、", stripped):
        return 1
    if stripped.startswith(("✎", "◆", "■")) and len(stripped) <= 80:
        return 2
    if re.match(r"^\d{1,2}[.．、)]", stripped) and len(stripped) <= 80:
        return 3
    return None


def _select_answer_evidence(evidence: list[SearchResult]) -> list[SearchResult]:
    if len(evidence) < 2:
        return evidence
    top_score = evidence[0].score
    if top_score <= 0:
        return evidence
    score_floor = top_score * 0.5
    return [item for item in evidence if item.score >= score_floor]


def _numbered_outline(text: str) -> list[tuple[int, str]]:
    matches = [
        (int(number), title.strip())
        for number, title in re.findall(
            r"(?m)^\s*(\d{1,2})[.．、]\s*([^\n]{2,48})$",
            text,
        )
    ]
    longest: list[tuple[int, str]] = []
    current: list[tuple[int, str]] = []
    for item in matches:
        number, _ = item
        if number == 1:
            if len(current) > len(longest):
                longest = current
            current = [item]
        elif current and number == current[-1][0] + 1:
            current.append(item)
        elif current:
            if len(current) > len(longest):
                longest = current
            current = []
    if len(current) > len(longest):
        longest = current
    return longest


def _ensure_numbered_outline_coverage(
    *,
    question: str,
    answer: str,
    outlines: list[tuple[int, list[tuple[int, str]]]],
) -> str:
    if not outlines or not _is_collection_question(question):
        return answer
    citation_id, outline = outlines[0]
    items = "；".join(f"{number}. {title}" for number, title in outline)
    return f"依据原文，共有 {len(outline)} 项：{items}。[{citation_id}]"


def _is_collection_question(question: str) -> bool:
    collection_terms = (
        "有哪些",
        "哪几",
        "包括哪些",
        "分别是",
        "列出",
        "所有",
        "全部",
        "基本原则",
        "主要原则",
        "主要步骤",
        "具体步骤",
        "主要方法",
        "基本要求",
    )
    return any(term in question for term in collection_terms)


def _expand_answer_evidence(
    *,
    connection: sqlite3.Connection,
    knowledge_base_id: str,
    evidence: list[SearchResult],
    radius: int = 2,
) -> list[SearchResult]:
    expanded: list[SearchResult] = []
    for result in evidence:
        rows = connection.execute(
            """
            SELECT neighbor.chunk_index, anchor.chunk_index AS anchor_chunk_index,
                neighbor.text
            FROM chunks AS anchor
            JOIN document_versions
                ON document_versions.id = anchor.document_version_id
            JOIN documents
                ON documents.active_version_id = document_versions.id
            JOIN chunks AS neighbor
                ON neighbor.document_version_id = anchor.document_version_id
            WHERE documents.knowledge_base_id = ?
                AND documents.id = ?
                AND documents.visibility_state = 'visible'
                AND anchor.id = ?
                AND neighbor.chunk_index BETWEEN anchor.chunk_index - ?
                    AND anchor.chunk_index + ?
            ORDER BY neighbor.chunk_index
            """,
            (
                knowledge_base_id,
                result.document_id,
                result.chunk_id,
                radius,
                radius,
            ),
        ).fetchall()
        labeled_texts = [
            f"【{_context_position_label(_chunk_offset(row))}】"
            f"\n{row['text']}"
            for row in rows
        ]
        context = merge_context_texts(labeled_texts)
        expanded.append(
            SearchResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                file_name=result.file_name,
                canonical_path=result.canonical_path,
                page_no=result.page_no,
                section_path=result.section_path,
                quote=context or result.quote,
                bbox=result.bbox,
                score=result.score,
                sources=result.sources,
            )
        )
    return expanded


def _context_position_label(offset: int) -> str:
    if offset == 0:
        return "命中块"
    if offset < 0:
        return f"前文第 {abs(offset)} 块"
    return f"后文第 {offset} 块"


def _chunk_offset(row: sqlite3.Row) -> int:
    return int(row["chunk_index"]) - int(row["anchor_chunk_index"])


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
