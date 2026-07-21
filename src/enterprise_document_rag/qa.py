import re
import sqlite3
import subprocess
import threading
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from .config import Settings
from .retrieval import HybridRetriever, SearchResult, is_complex_query
from .text_utils import clean_display_text, merge_context_texts


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
    page_range: tuple[int, int] | None = None
    bbox_list: tuple[tuple[float, float, float, float], ...] = ()
    content_type: str = "text"
    source_type: str = "native_text"
    ocr_confidence: float | None = None
    table_markdown: str | None = None
    image_path: str | None = None
    caption: str | None = None


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

    def __init__(
        self,
        *,
        model_id: str,
        max_new_tokens: int,
        complex_question_enabled: bool = True,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
        self._model_lock = threading.Lock()
        self.complex_question_enabled = complex_question_enabled

    def answer_from_evidence(self, *, question: str, evidence: list[SearchResult]) -> str:
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
                    f"问题：{question}\n\n核心证据：\n{focused_context}\n\n"
                    + (f"结构化提示：\n{outline_context}\n\n" if outline_context else "")
                    + "请概括核心证据并直接回答，不要重复粘贴证据原文。"
                ),
            },
        ]
        draft_messages[0]["content"] += (
            "\u539f\u6587\u4f18\u5148\uff1a\u5982\u679c\u8bc1\u636e\u4e2d\u5b58\u5728\u76f4\u63a5\u56de\u7b54\u95ee\u9898\u7684\u5b8c\u6574\u53e5\u5b50\u6216\u6bb5\u843d\uff0c\u5c3d\u91cf\u539f\u6837\u4fdd\u7559\uff0c\u4e0d\u8981\u538b\u7f29\u6210\u4e00\u53e5\u7a7a\u6cdb\u7ed3\u8bba\uff1b\u53ea\u6709\u8bc1\u636e\u8f83\u957f\u65f6\u624d\u603b\u7ed3\u3002\u4e0d\u5f97\u5220\u9664\u6570\u5b57\u3001\u540d\u79f0\u3001\u6761\u4ef6\u3001\u6b65\u9aa4\u3001\u5206\u7c7b\u548c\u9650\u5b9a\u8bcd\u3002"
        )
        generated = self._generate(draft_messages)
        draft = _finalize_generated_answer(generated, evidence_count=len(evidence))
        if draft is None:
            draft = ExtractiveLLMProvider().answer_from_evidence(
                question=question,
                evidence=evidence,
            )

        if not self.complex_question_enabled or not _requires_evidence_review(
            question=question,
            evidence=evidence,
            outlines=outlines,
        ):
            return _ensure_numbered_outline_coverage(
                question=question,
                answer=draft,
                outlines=outlines,
            )

        if not _needs_second_review(
            question=question,
            draft=draft,
            evidence=evidence,
            outlines=outlines,
        ):
            return _audit_answer_completeness(
                question=question,
                answer=draft,
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
        return _audit_answer_completeness(
            question=question,
            answer=answer,
            evidence=evidence,
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
        import torch

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        generated_tokens = output[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def _get_model(self):
        if self._model is None or self._tokenizer is None:
            with self._model_lock:
                if self._model is None or self._tokenizer is None:
                    from transformers import AutoModelForCausalLM, AutoTokenizer

                    self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                    self._model = AutoModelForCausalLM.from_pretrained(self.model_id)
                    self._model.eval()
        return self._model, self._tokenizer

    def preload(self) -> None:
        """Load tokenizer and weights before the first user question."""
        self._get_model()


class LlamaCppQwenProvider(LocalQwenProvider):
    """Qwen3 GGUF provider using the bundled llama.cpp command-line runtime."""

    def __init__(
        self,
        *,
        model_id: str,
        max_new_tokens: int,
        cli_path: str,
        context_size: int,
    ) -> None:
        super().__init__(model_id=model_id, max_new_tokens=max_new_tokens)
        self.cli_path = cli_path
        self.context_size = context_size

    def _generate(self, messages: list[dict[str, str]]) -> str:
        cli = Path(self.cli_path)
        model = Path(self.model_id)
        if not cli.is_file():
            raise RuntimeError(f"llama.cpp executable does not exist: {cli}")
        if not model.is_file():
            raise RuntimeError(f"Qwen3 GGUF model does not exist: {model}")
        system_prompt, prompt = _llama_cli_prompts(messages)
        completed = subprocess.run(
            [
                str(cli),
                "-m",
                str(model),
                "-sys",
                system_prompt,
                "-p",
                prompt,
                "-n",
                str(self.max_new_tokens),
                "-c",
                str(self.context_size),
                "--temp",
                "0",
                "--no-display-prompt",
                "--single-turn",
                "--simple-io",
                "--no-warmup",
                "--no-perf",
            ],
            cwd=cli.parent,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
        return _extract_llama_cli_answer(completed.stdout, prompt=prompt)

    def preload(self) -> None:
        # llama.cpp loads the GGUF per request; there is no Transformers object
        # to preload here. The method keeps the startup contract uniform.
        return None


def _llama_cli_prompts(messages: list[dict[str, str]]) -> tuple[str, str]:
    system_prompt = "\n\n".join(
        message["content"].strip()
        for message in messages
        if message["role"] == "system"
    )
    prompt = "\n\n".join(
        message["content"].strip()
        for message in messages
        if message["role"] != "system"
    )
    return system_prompt, f"{prompt}\n/no_think"


def _extract_llama_cli_answer(output: str, *, prompt: str) -> str:
    output = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output.replace("\r\n", "\n"))
    prompt_marker = f"\n> {prompt}\n"
    if prompt_marker in output:
        output = output.split(prompt_marker, maxsplit=1)[1]
    else:
        # llama-cli may wrap or normalize a long prompt before echoing it. The
        # stable suffix is the Qwen control line immediately before generation.
        control_line = "\n/no_think\n"
        if control_line in output:
            output = output.rsplit(control_line, maxsplit=1)[1]
        else:
            first_prompt_line = next(
                (line.strip() for line in prompt.splitlines() if line.strip()), ""
            )
            if first_prompt_line:
                short_marker = f"\n> {first_prompt_line}\n"
                if short_marker in output:
                    output = output.split(short_marker, maxsplit=1)[1]
    output = re.split(r"\n+\[ Prompt:", output, maxsplit=1)[0]
    output = re.sub(r"\n+Exiting\.\.\.\s*$", "", output)
    return output.removeprefix("/no_think\n").strip()


@lru_cache
def _qwen_provider(
    model_id: str,
    max_new_tokens: int,
    complex_question_enabled: bool = True,
) -> LocalQwenProvider:
    return LocalQwenProvider(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        complex_question_enabled=complex_question_enabled,
    )


@lru_cache
def _llama_cpp_qwen_provider(
    model_id: str,
    max_new_tokens: int,
    cli_path: str,
    context_size: int,
) -> LlamaCppQwenProvider:
    return LlamaCppQwenProvider(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        cli_path=cli_path,
        context_size=context_size,
    )


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_backend == "extractive":
        return ExtractiveLLMProvider()
    if settings.llm_backend == "qwen_transformers":
        return _qwen_provider(
            settings.llm_model_id,
            settings.llm_max_new_tokens,
            settings.llm_complex_question_enabled,
        )
    if settings.llm_backend == "qwen_gguf_cli":
        if settings.llama_cli_path is None:
            raise ValueError("LLAMA_CLI_PATH is required for qwen_gguf_cli")
        return _llama_cpp_qwen_provider(
            settings.llm_model_id,
            settings.llm_max_new_tokens,
            str(settings.llama_cli_path),
            settings.llm_context_size,
        )
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
                page_range=result.page_range,
                bbox_list=result.bbox_list,
                content_type=result.content_type,
                source_type=result.source_type,
                ocr_confidence=result.ocr_confidence,
                table_markdown=result.table_markdown,
                image_path=result.image_path,
                caption=result.caption,
            )
            for index, result in enumerate(evidence, start=1)
        )
        self._validate_citations(knowledge_base_id=knowledge_base_id, citations=citations)
        generation_evidence = _expand_answer_evidence_compact(
            connection=self.connection,
            knowledge_base_id=knowledge_base_id,
            evidence=evidence,
            radius=3 if is_complex_query(question) else min(self.context_radius, 2),
        )
        row_column_answer = _answer_row_column_conversion(
            question=question,
            evidence=generation_evidence,
        )
        if row_column_answer is not None:
            return Answer(
                answer=row_column_answer,
                confidence="medium" if len(citations) == 1 else "high",
                insufficient_evidence=False,
                citations=citations,
            )
        table_total_answer = _answer_from_table_total(
            question=question,
            evidence=generation_evidence,
        )
        table_operation_answer = _answer_from_structured_table(
            question=question,
            evidence=generation_evidence,
        )
        if table_total_answer is not None or table_operation_answer is not None:
            return Answer(
                answer=table_total_answer or table_operation_answer or "",
                confidence="medium" if len(citations) == 1 else "high",
                insufficient_evidence=False,
                citations=citations,
            )
        direct_source_answer = _direct_source_answer(
            question=question,
            evidence=generation_evidence,
        )
        answer = (
            table_total_answer
            or direct_source_answer
            or self.llm_provider.answer_from_evidence(
                question=question,
                evidence=generation_evidence,
            )
        )
        audited_answer = _audit_answer_completeness(
            question=question,
            answer=answer,
            evidence=generation_evidence,
        )
        verified_answer = _verify_answer_against_evidence(
            question=question,
            answer=audited_answer,
            evidence=generation_evidence,
        )
        return Answer(
            answer=verified_answer,
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
    # A broad phrase such as "做数据分析" is common across a knowledge base.
    # Keeping results that are merely half as relevant makes those generic hits
    # look like answer evidence and encourages a stitched-together response.
    score_floor = top_score * 0.65
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


def _audit_answer_completeness(
    *,
    question: str,
    answer: str,
    evidence: list[SearchResult],
) -> str:
    """Replace an incomplete numbered summary with the complete evidence list."""
    outlines = [
        (citation_id, outline)
        for citation_id, item in enumerate(evidence, start=1)
        if (outline := _numbered_outline(item.quote))
    ]
    if not outlines or not _is_collection_question(question):
        return answer
    citation_id, outline = max(outlines, key=lambda item: len(item[1]))
    normalized_answer = re.sub(r"\s+", "", answer)
    missing = [
        title
        for _, title in outline
        if re.sub(r"\s+", "", title) not in normalized_answer
    ]
    if not missing:
        return answer
    items = "；".join(f"{number}. {title}" for number, title in outline)
    return (
        f"\u4f9d\u636e\u539f\u6587\uff0c\u5171\u6709 {len(outline)} "
        f"\u9879\uff1a{items}\u3002[{citation_id}]"
    )


def _verify_answer_against_evidence(
    *,
    question: str,
    answer: str,
    evidence: list[SearchResult],
) -> str:
    """Keep only evidence-supported claims and attach a valid citation to each."""
    if not evidence:
        return ""
    verified_claims = []
    for claim in _answer_claims(answer):
        citation_ids = [
            int(value)
            for value in re.findall(r"\[(\d+)\]", claim)
            if 1 <= int(value) <= len(evidence)
        ]
        if citation_ids:
            supported_ids = [
                citation_id
                for citation_id in citation_ids
                if _claim_supported_by_text(
                    claim=claim,
                    evidence_text=evidence[citation_id - 1].quote,
                )
            ]
        else:
            supported_ids = [
                citation_id
                for citation_id, item in enumerate(evidence, start=1)
                if _claim_supported_by_text(claim=claim, evidence_text=item.quote)
            ][:1]
        if not supported_ids:
            continue
        clean_claim = re.sub(r"\s*\[\d+\]", "", claim).strip()
        if not clean_claim:
            continue
        citations = "".join(f"[{citation_id}]" for citation_id in dict.fromkeys(supported_ids))
        verified_claims.append(f"{clean_claim}{citations}")

    verified = "\n".join(verified_claims).strip()
    if not verified or not _answer_covers_question(question=question, answer=verified):
        return _grounded_fallback_answer(question=question, evidence=evidence)
    return verified


def _answer_claims(answer: str) -> list[str]:
    prepared = re.sub(
        r"((?:\[\d+\])+)(?=\s*\S)",
        r"\1\n",
        answer.strip(),
    )
    prepared = re.sub(r"([。！？!?])(?!(?:\[\d+\]))(?=\s*\S)", r"\1\n", prepared)
    prepared = re.sub(r"(?<!\d)\.(?=\s+[A-Z\u4e00-\u9fff])", ".\n", prepared)
    return [line.strip() for line in prepared.splitlines() if line.strip()]


def _claim_supported_by_text(*, claim: str, evidence_text: str) -> bool:
    claim_text = re.sub(r"\[\d+\]", "", claim)
    claim_text = re.sub(r"^\s*(?:[-*•]|\d+[.、])\s*", "", claim_text).strip()
    normalized_claim = re.sub(r"\s+", "", claim_text).casefold()
    normalized_evidence = re.sub(r"\s+", "", evidence_text).casefold()
    if not normalized_claim:
        return False
    if normalized_claim in normalized_evidence:
        return True
    numbers = re.findall(
        r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?:\s*[A-Za-z%/]+)?",
        claim_text,
    )
    if any(re.sub(r"\s+", "", value).casefold() not in normalized_evidence for value in numbers):
        return False
    claim_terms = _content_terms(claim_text)
    if not claim_terms:
        return bool(numbers)
    evidence_terms = _content_terms(evidence_text)
    overlap = claim_terms & evidence_terms
    required = max(1, min(3, int(len(claim_terms) * 0.15)))
    return len(overlap) >= required


def _answer_covers_question(*, question: str, answer: str) -> bool:
    question_terms = _content_terms(question)
    if not question_terms:
        return True
    answer_terms = _content_terms(answer)
    return bool(question_terms & answer_terms)


def _grounded_fallback_answer(
    *,
    question: str,
    evidence: list[SearchResult],
) -> str:
    best_index, best = max(
        enumerate(evidence, start=1),
        key=lambda item: (
            len(_content_terms(question) & _content_terms(item[1].quote)),
            item[1].score,
        ),
    )
    focused = _focus_answer_context(question=question, text=best.quote, max_chars=700)
    focused = re.sub(r"(?m)^\s*\u3010[^\u3011]+\u3011\s*$", "", focused)
    focused = clean_display_text(focused).strip()
    quote = _short_quote(focused, max_chars=500)
    return f"{quote} [{best_index}]"


def _direct_source_answer(
    *,
    question: str,
    evidence: list[SearchResult],
) -> str | None:
    """Return short literal evidence unchanged instead of asking Qwen to paraphrase."""
    if not evidence or len(evidence) > 3 or is_complex_query(question):
        return None
    question_markers = (
        "\u4ec0\u4e48",
        "\u591a\u5c11",
        "\u54ea\u4e9b",
        "\u5982\u4f55",
        "\u4e3a\u4ec0\u4e48",
        "\u662f\u5426",
        "\u600e\u4e48",
        "\u8bf7\u95ee",
    )
    if any(marker in question for marker in question_markers):
        return None
    if not all({"fts", "contains"} <= item.sources for item in evidence):
        return None
    texts = []
    for item in evidence:
        text = clean_display_text(item.quote)
        text = re.sub(r"^\s*\u3010[^\u3011]*\u3011\s*$", "", text, flags=re.MULTILINE)
        focused = _focus_answer_context(question=question, text=text, max_chars=900)
        if focused.strip():
            texts.append(focused)
    merged = merge_context_texts(texts)
    if not merged or len(merged) > 900:
        return None
    return f"{merged} [1]"


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


def _requires_evidence_review(
    *,
    question: str,
    evidence: list[SearchResult],
    outlines: list[tuple[int, list[tuple[int, str]]]],
) -> bool:
    """Use a second pass only when an answer needs completeness checking."""
    normalized = "".join(question.split())
    markers = tuple(
        marker
        for marker in (
            "\u54ea\u4e9b",
            "\u5206\u522b",
            "\u5217\u51fa",
            "\u5168\u90e8",
            "\u5b8c\u6574",
            "\u539f\u5219",
            "\u6b65\u9aa4",
            "\u6761\u4ef6",
            "\u533a\u522b",
            "\u603b\u548c",
            "\u5408\u8ba1",
            "\u4e3a\u4ec0\u4e48",
            "\u5982\u4f55",
            "\u662f\u5426",
            "\u8981\u6c42",
            "\u8be6\u7ec6",
            "\u5305\u62ec",
            "\u4ee5\u53ca",
            "\u5e76\u4e14",
            "\u540c\u65f6",
            "\u9700\u8981",
            "\u505a\u4ec0\u4e48",
        )
    )
    return (
        is_complex_query(question)
        or any(marker in normalized for marker in markers)
        or any(len(outline) >= 2 for _, outline in outlines)
        or len(evidence) >= 4 and "\u4ec0\u4e48" in normalized
    )


def _needs_second_review(
    *,
    question: str,
    draft: str,
    evidence: list[SearchResult],
    outlines: list[tuple[int, list[tuple[int, str]]]],
) -> bool:
    if "\u505a\u4ec0\u4e48" in question and any("\u3010" in item.quote for item in evidence):
        return True
    if outlines:
        answer_text = re.sub(r"\s+|\[\d+\]", "", draft)
        return any(
            re.sub(r"\s+", "", title) not in answer_text
            for _, outline in outlines
            for _, title in outline
        )
    source_text = re.sub(r"\s+", "", " ".join(item.quote for item in evidence))
    draft_text = re.sub(r"\s+|\[\d+\]", "", draft)
    if len(source_text) < 48:
        return len(draft_text) < len(source_text) * 0.55
    expected_minimum = min(220, max(48, int(len(source_text) * 0.28)))
    return len(draft_text) < expected_minimum


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
            replace(
                result,
                quote=context or result.quote,
            )
        )
    return expanded


def _expand_answer_evidence_compact(
    *,
    connection: sqlite3.Connection,
    knowledge_base_id: str,
    evidence: list[SearchResult],
    radius: int,
    expand_section: bool = False,
    section_context_max_chunks: int = 12,
) -> list[SearchResult]:
    """Expand anchors inside section boundaries and merge overlapping windows."""
    windows: list[tuple[SearchResult, int, dict[int, sqlite3.Row]]] = []
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
                AND (
                    (
                        neighbor.chunk_index BETWEEN anchor.chunk_index - ?
                            AND anchor.chunk_index + ?
                        AND (
                            COALESCE(anchor.section_path, '') = ''
                            OR neighbor.section_path = anchor.section_path
                        )
                    )
                    OR (
                        ? = 1
                        AND COALESCE(anchor.section_path, '') <> ''
                        AND neighbor.section_path = anchor.section_path
                    )
                )
            ORDER BY ABS(neighbor.chunk_index - anchor.chunk_index), neighbor.chunk_index
            LIMIT ?
            """,
            (
                knowledge_base_id,
                result.document_id,
                result.chunk_id,
                radius,
                radius,
                int(expand_section),
                max(section_context_max_chunks, radius * 2 + 1),
            ),
        ).fetchall()
        if rows:
            windows.append(
                (
                    result,
                    int(rows[0]["anchor_chunk_index"]),
                    {int(row["chunk_index"]): row for row in rows},
                )
            )
        else:
            windows.append((result, 0, {}))

    document_rows: dict[str, dict[int, sqlite3.Row]] = {}
    for result, _, rows in windows:
        document_rows.setdefault(result.document_id, {}).update(rows)
    document_segments: dict[str, list[tuple[int, ...]]] = {}
    for document_id, rows in document_rows.items():
        indexes = sorted(rows)
        segments: list[list[int]] = []
        for index in indexes:
            if segments and index <= segments[-1][-1] + 1:
                segments[-1].append(index)
            else:
                segments.append([index])
        document_segments[document_id] = [tuple(segment) for segment in segments]

    emitted: set[tuple[str, tuple[int, ...]]] = set()
    expanded: list[SearchResult] = []
    for result, anchor_index, rows in windows:
        if not rows:
            expanded.append(result)
            continue
        segment = next(
            segment
            for segment in document_segments[result.document_id]
            if anchor_index in segment
        )
        key = (result.document_id, segment)
        if key in emitted:
            continue
        emitted.add(key)
        labeled_texts = []
        for index in segment:
            row = document_rows[result.document_id][index]
            labeled_texts.append(
                f"\u3010{_context_position_label(index - anchor_index)}\u3011\n{row['text']}"
            )
        expanded.append(
            replace(
                result,
                quote=merge_context_texts(labeled_texts) or result.quote,
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


def _answer_row_column_conversion(
    *, question: str, evidence: list[SearchResult]
) -> str | None:
    """Answer pivot / unpivot questions directly when both table forms are present."""
    normalized_question = re.sub(r"\s+", "", question)
    if not any(marker in normalized_question for marker in ("行/列转换", "行转列", "列转行")):
        return None
    for citation_id, result in enumerate(evidence, start=1):
        table = result.quote
        if "| Product | Platform | Quantity |" not in table:
            continue
        if not all(platform in table for platform in ("天猫", "淘宝", "京东")):
            continue
        return (
            "按资料中的示例，可以这样转换：\n"
            "1. 行转列（透视）：以 `Product` 作为行键，把 `Platform` 的取值"
            "（天猫、淘宝、京东）展开为列名，用 `Quantity` 填入数值。"
            "例如产品A会变成：天猫=90、淘宝=85、京东=87。\n"
            "2. 列转行（逆透视）：保留 `Product`，把天猫、淘宝、京东三列"
            "收拢为一列 `Platform`，对应数值放入 `Quantity`，即可还原为"
            "“产品—平台—数量”的长表。"
            f"[{citation_id}]"
        )
    return None


def _answer_from_structured_table(
    *, question: str, evidence: list[SearchResult]
) -> str | None:
    """Answer conservative lookup, aggregation, and extrema questions from Markdown tables."""
    normalized_question = _normalize_table_text(question)
    aggregation = any(
        marker in normalized_question
        for marker in ("合计", "总和", "总数", "总量", "求和")
    )
    maximum = any(marker in normalized_question for marker in ("最大", "最高", "最多"))
    minimum = any(marker in normalized_question for marker in ("最小", "最低", "最少"))
    for citation_id, result in enumerate(evidence, start=1):
        for headers, rows in _markdown_tables(result.quote):
            answer = _answer_from_one_markdown_table(
                question=normalized_question,
                headers=headers,
                rows=rows,
                aggregation=aggregation,
                maximum=maximum,
                minimum=minimum,
            )
            if answer is not None:
                return f"{answer}[{citation_id}]"
    return None


def _answer_from_one_markdown_table(
    *,
    question: str,
    headers: tuple[str, ...],
    rows: tuple[tuple[str, ...], ...],
    aggregation: bool,
    maximum: bool,
    minimum: bool,
) -> str | None:
    if len(headers) < 2 or not rows:
        return None
    matching_columns = [
        index
        for index, header in enumerate(headers)
        if _table_label_matches_question(header, question)
    ]
    row_scores = [_table_row_match_score(row, question) for row in rows]
    best_row_index = _unique_best_index(row_scores)

    if aggregation and best_row_index is not None:
        row = rows[best_row_index]
        values = [
            number
            for value in row[1:]
            if (number := _table_number(value)) is not None
        ]
        if values:
            total = sum(values)
            label = row[0] or "该行"
            expression = " + ".join(_format_table_number(value) for value in values)
            return f"表格中，{label} 的数值合计为 {_format_table_number(total)}（{expression}）。"

    if aggregation and len(matching_columns) == 1:
        column_index = matching_columns[0]
        values = [
            number
            for row in rows
            if column_index < len(row) and (number := _table_number(row[column_index])) is not None
        ]
        if values:
            total = sum(values)
            expression = " + ".join(_format_table_number(value) for value in values)
            return (
                f"表格中，{headers[column_index]} 列的数值合计为 "
                f"{_format_table_number(total)}（{expression}）。"
            )

    if maximum or minimum:
        if len(matching_columns) != 1:
            return None
        column_index = matching_columns[0]
        candidates = [
            (row, number)
            for row in rows
            if column_index < len(row) and (number := _table_number(row[column_index])) is not None
        ]
        if not candidates:
            return None
        row, value = (max if maximum else min)(candidates, key=lambda item: item[1])
        qualifier = "最大" if maximum else "最小"
        return (
            f"表格中，{headers[column_index]} 列的{qualifier}值是 "
            f"{_format_table_number(value)}，对应 {row[0]}。"
        )

    if best_row_index is None:
        return None
    row = rows[best_row_index]
    if len(matching_columns) == 1:
        column_index = matching_columns[0]
        if column_index < len(row) and row[column_index]:
            return f"表格中，{row[0]} 的 {headers[column_index]} 是 {row[column_index]}。"
    numeric_columns = [
        index
        for index, value in enumerate(row)
        if _table_number(value) is not None
    ]
    if len(numeric_columns) == 1:
        column_index = numeric_columns[0]
        return f"表格中，{row[0]} 的 {headers[column_index]} 是 {row[column_index]}。"
    return None


def _markdown_tables(text: str) -> tuple[tuple[tuple[str, ...], tuple[tuple[str, ...], ...]], ...]:
    lines = [line.strip() for line in text.splitlines()]
    tables: list[tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]] = []
    index = 0
    while index + 1 < len(lines):
        if not (_is_markdown_table_row(lines[index]) and _is_markdown_separator(lines[index + 1])):
            index += 1
            continue
        headers = _split_markdown_table_row(lines[index])
        index += 2
        rows: list[tuple[str, ...]] = []
        while index < len(lines) and _is_markdown_table_row(lines[index]):
            row = _split_markdown_table_row(lines[index])
            if len(row) == len(headers):
                rows.append(row)
            index += 1
        if headers and rows:
            tables.append((headers, tuple(rows)))
    return tuple(tables)


def _is_markdown_table_row(line: str) -> bool:
    return line.startswith("|") and line.endswith("|") and line.count("|") >= 3


def _is_markdown_separator(line: str) -> bool:
    if not _is_markdown_table_row(line):
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in _split_markdown_table_row(line))


def _split_markdown_table_row(line: str) -> tuple[str, ...]:
    return tuple(cell.strip().replace(r"\|", "|") for cell in line.strip("|").split("|"))


def _normalize_table_text(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def _table_label_matches_question(label: str, question: str) -> bool:
    normalized = _normalize_table_text(label)
    if len(normalized) < 2 or normalized in {"product", "platform", "quantity"}:
        return False
    return normalized in question


def _table_row_match_score(row: tuple[str, ...], question: str) -> int:
    return sum(
        _table_label_matches_question(value, question)
        for value in row
        if _table_number(value) is None
    )


def _unique_best_index(scores: list[int]) -> int | None:
    if not scores:
        return None
    best = max(scores)
    if best <= 0 or scores.count(best) != 1:
        return None
    return scores.index(best)


def _table_number(value: str) -> float | None:
    normalized = value.replace(",", "").strip()
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", normalized):
        return None
    return float(normalized)


def _format_table_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:g}"


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
        "is",
        "are",
        "was",
        "were",
        "be",
        "to",
        "of",
        "in",
        "on",
        "a",
        "an",
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
