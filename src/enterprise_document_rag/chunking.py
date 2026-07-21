import hashlib
import re
from dataclasses import dataclass, replace

from .parsing import ParsedBlock, ParsedDocument
from .text_utils import sanitize_unicode

TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]")


@dataclass(frozen=True)
class Chunk:
    chunk_index: int
    text: str
    page_no: int | None
    section_path: tuple[str, ...]
    bbox: tuple[float, float, float, float] | None
    token_count: int
    text_hash: str
    deduplicable: bool = True
    previous_chunk_index: int | None = None
    next_chunk_index: int | None = None
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


class StructureAwareChunker:
    def __init__(
        self,
        *,
        target_tokens: int = 500,
        overlap_tokens: int = 60,
        min_tokens: int = 300,
        max_tokens: int = 800,
    ) -> None:
        if overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens")
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def chunk(self, document: ParsedDocument) -> tuple[Chunk, ...]:
        chunks: list[Chunk] = []
        pending: list[ParsedBlock] = []
        pending_tokens = 0
        pending_from_overlap = False

        for raw_block in document.blocks:
            for block in self._split_oversized_block(raw_block):
                chunks, pending, pending_tokens, pending_from_overlap = self._append_block(
                    chunks=chunks,
                    pending=pending,
                    pending_tokens=pending_tokens,
                    pending_from_overlap=pending_from_overlap,
                    block=block,
                )

        if pending and (not pending_from_overlap or not self._contains_split_block(pending)):
            if (
                chunks
                and pending_tokens < self.min_tokens
                and chunks[-1].token_count + pending_tokens <= self.max_tokens
            ):
                merged = self._merge_chunk_with_blocks(chunks.pop(), pending)
                chunks.append(self._with_index(merged, len(chunks)))
            else:
                chunks.append(self._build_chunk(chunks, pending))

        return self._link_chunks(self._deduplicate_exact_chunks(chunks))

    def _append_block(
        self,
        *,
        chunks: list[Chunk],
        pending: list[ParsedBlock],
        pending_tokens: int,
        pending_from_overlap: bool,
        block: ParsedBlock,
    ) -> tuple[list[Chunk], list[ParsedBlock], int, bool]:
        block_tokens = count_tokens(block.text)
        base_type = block.block_type.split(":", 1)[0]
        if base_type in {"table", "table_row", "image"}:
            if pending and not pending_from_overlap:
                chunks.append(self._build_chunk(chunks, pending))
            chunks.append(self._build_chunk(chunks, [block]))
            return chunks, [], 0, False
        if (
            pending
            and block.section_path != pending[-1].section_path
            and block.block_type == "heading"
        ):
            if not pending_from_overlap:
                chunks.append(self._build_chunk(chunks, pending))
            pending = []
            pending_tokens = 0
            pending_from_overlap = False
        if pending and pending_tokens + block_tokens > self.max_tokens:
            if not pending_from_overlap:
                chunks.append(self._build_chunk(chunks, pending))
                pending = self._overlap_blocks(pending)
                pending_tokens = sum(count_tokens(item.text) for item in pending)
            pending = self._trim_overlap_to_fit(
                blocks=pending,
                available_tokens=self.max_tokens - block_tokens,
            )
            pending_tokens = sum(count_tokens(item.text) for item in pending)
        pending.append(block)
        pending_tokens += block_tokens
        pending_from_overlap = False
        if pending_tokens >= self.target_tokens:
            chunks.append(self._build_chunk(chunks, pending))
            pending = self._overlap_blocks(pending)
            pending_tokens = sum(count_tokens(item.text) for item in pending)
            pending_from_overlap = True
        return chunks, pending, pending_tokens, pending_from_overlap

    def _split_oversized_block(self, block: ParsedBlock) -> tuple[ParsedBlock, ...]:
        matches = list(TOKEN_PATTERN.finditer(block.text))
        if len(matches) <= self.max_tokens:
            return (block,)
        if block.block_type == "table" and block.table_markdown:
            table_pieces = self._split_markdown_table(block)
            if table_pieces:
                return table_pieces
        pieces: list[ParsedBlock] = []
        start_offset = 0
        for start_index in range(0, len(matches), self.max_tokens):
            end_index = min(start_index + self.max_tokens, len(matches))
            end_offset = matches[end_index - 1].end()
            text = block.text[start_offset:end_offset].strip()
            if text:
                pieces.append(
                    replace(
                        block,
                        text=text,
                        block_type=f"{block.block_type}:chunk_split",
                    )
                )
            start_offset = end_offset
        tail = block.text[start_offset:].strip()
        if tail:
            pieces.append(
                replace(
                    block,
                    text=tail,
                    block_type=f"{block.block_type}:chunk_split",
                )
            )
        return tuple(pieces)

    def _split_markdown_table(self, block: ParsedBlock) -> tuple[ParsedBlock, ...]:
        lines = [line for line in block.table_markdown.splitlines() if line.strip()]
        if len(lines) < 3:
            return ()
        header = lines[:2]
        header_tokens = count_tokens("\n".join(header))
        if header_tokens >= self.max_tokens:
            return ()
        groups: list[list[str]] = []
        current = list(header)
        current_tokens = header_tokens
        for row in lines[2:]:
            row_tokens = count_tokens(row)
            if len(current) > 2 and current_tokens + row_tokens > self.max_tokens:
                groups.append(current)
                current = list(header)
                current_tokens = header_tokens
            current.append(row)
            current_tokens += row_tokens
        if len(current) > 2:
            groups.append(current)
        return tuple(
            replace(
                block,
                text="\n".join(group),
                block_type="table:chunk_split",
                table_markdown="\n".join(group),
            )
            for group in groups
        )

    def _trim_overlap_to_fit(
        self,
        *,
        blocks: list[ParsedBlock],
        available_tokens: int,
    ) -> list[ParsedBlock]:
        if available_tokens <= 0:
            return []
        selected: list[ParsedBlock] = []
        total = 0
        for block in reversed(blocks):
            block_tokens = count_tokens(block.text)
            if total + block_tokens > available_tokens:
                break
            selected.insert(0, block)
            total += block_tokens
        return selected

    def _build_chunk(self, chunks: list[Chunk], blocks: list[ParsedBlock]) -> Chunk:
        return self._chunk_from_blocks(chunk_index=len(chunks), blocks=blocks)

    def _chunk_from_blocks(self, *, chunk_index: int, blocks: list[ParsedBlock]) -> Chunk:
        text = sanitize_unicode("\n\n".join(block.text for block in blocks)).strip()
        first = blocks[0]
        metadata = _metadata_from_blocks(blocks)
        return Chunk(
            chunk_index=chunk_index,
            text=text,
            page_no=first.page_no,
            section_path=first.section_path,
            bbox=first.bbox,
            token_count=count_tokens(text),
            text_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            deduplicable=not self._contains_split_block(blocks),
            **metadata,
        )

    def _merge_chunk_with_blocks(self, chunk: Chunk, blocks: list[ParsedBlock]) -> Chunk:
        extra_text = "\n\n".join(block.text for block in blocks).strip()
        text = sanitize_unicode("\n\n".join(item for item in [chunk.text, extra_text] if item))
        extra = _metadata_from_blocks(blocks)
        page_range = _merge_page_ranges(chunk.page_range, extra["page_range"])
        bbox_list = tuple(dict.fromkeys((*chunk.bbox_list, *extra["bbox_list"])))
        content_type = _merge_label(chunk.content_type, str(extra["content_type"]))
        source_type = _merge_label(chunk.source_type, str(extra["source_type"]))
        confidences = [
            value
            for value in (chunk.ocr_confidence, extra["ocr_confidence"])
            if isinstance(value, (int, float))
        ]
        return replace(
            chunk,
            text=text,
            token_count=count_tokens(text),
            text_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            deduplicable=chunk.deduplicable and not self._contains_split_block(blocks),
            page_range=page_range,
            bbox_list=bbox_list,
            content_type=content_type,
            source_type=source_type,
            ocr_confidence=sum(confidences) / len(confidences) if confidences else None,
            block_types=tuple(
                dict.fromkeys((*chunk.block_types, *extra["block_types"]))
            ),
            table_markdown=(
                chunk.table_markdown
                if chunk.table_markdown == extra["table_markdown"]
                else None
            ),
            image_path=(
                chunk.image_path
                if chunk.image_path == extra["image_path"]
                else None
            ),
            caption=(
                chunk.caption
                if chunk.caption == extra["caption"]
                else None
            ),
            image_metadata=(
                chunk.image_metadata
                if chunk.image_metadata == extra["image_metadata"]
                else None
            ),
        )

    def _with_index(self, chunk: Chunk, chunk_index: int) -> Chunk:
        return replace(
            chunk,
            chunk_index=chunk_index,
            previous_chunk_index=None,
            next_chunk_index=None,
        )

    def _overlap_blocks(self, blocks: list[ParsedBlock]) -> list[ParsedBlock]:
        if self.overlap_tokens <= 0:
            return []
        selected: list[ParsedBlock] = []
        total = 0
        for block in reversed(blocks):
            selected.insert(0, block)
            total += count_tokens(block.text)
            if total >= self.overlap_tokens:
                break
        return selected

    def _deduplicate_exact_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        seen_hashes: set[str] = set()
        unique_chunks: list[Chunk] = []
        for chunk in chunks:
            if chunk.deduplicable and chunk.text_hash in seen_hashes:
                continue
            if chunk.deduplicable:
                seen_hashes.add(chunk.text_hash)
            unique_chunks.append(chunk)
        return unique_chunks

    @staticmethod
    def _contains_split_block(blocks: list[ParsedBlock]) -> bool:
        return any(block.block_type.endswith(":chunk_split") for block in blocks)

    def _link_chunks(self, chunks: list[Chunk]) -> tuple[Chunk, ...]:
        linked: list[Chunk] = []
        for index, chunk in enumerate(chunks):
            linked.append(
                replace(
                    chunk,
                    chunk_index=index,
                    previous_chunk_index=index - 1 if index > 0 else None,
                    next_chunk_index=index + 1 if index < len(chunks) - 1 else None,
                )
            )
        return tuple(linked)


def count_tokens(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def _metadata_from_blocks(blocks: list[ParsedBlock]) -> dict[str, object]:
    page_numbers = [block.page_no for block in blocks if block.page_no is not None]
    boxes = tuple(
        dict.fromkeys(block.bbox for block in blocks if block.bbox is not None)
    )
    block_types = tuple(
        dict.fromkeys(block.block_type.split(":", 1)[0] for block in blocks)
    )
    content_types = {
        "table"
        if block_type in {"table", "table_row"}
        else "image"
        if block_type == "image"
        else "text"
        for block_type in block_types
        if block_type not in {"header", "footer"}
    }
    source_types = {block.source_type for block in blocks}
    confidences = [
        float(block.confidence)
        for block in blocks
        if block.source_type == "ocr_text" and block.confidence is not None
    ]
    return {
        "page_range": (
            (min(page_numbers), max(page_numbers))
            if page_numbers
            else None
        ),
        "bbox_list": boxes,
        "content_type": (
            next(iter(content_types)) if len(content_types) == 1 else "mixed"
        ),
        "source_type": (
            next(iter(source_types)) if len(source_types) == 1 else "mixed"
        ),
        "ocr_confidence": (
            sum(confidences) / len(confidences) if confidences else None
        ),
        "block_types": block_types,
        "table_markdown": next(
            (
                block.table_markdown
                for block in blocks
                if block.table_markdown is not None
            ),
            None,
        ),
        "image_path": next(
            (block.image_path for block in blocks if block.image_path is not None),
            None,
        ),
        "caption": next(
            (block.caption for block in blocks if block.caption is not None),
            None,
        ),
        "image_metadata": next(
            (block.image_metadata for block in blocks if block.image_metadata is not None),
            None,
        ),
    }


def _merge_page_ranges(
    first: tuple[int, int] | None,
    second: object,
) -> tuple[int, int] | None:
    if not isinstance(second, tuple):
        return first
    if first is None:
        return second
    return (min(first[0], second[0]), max(first[1], second[1]))


def _merge_label(first: str, second: str) -> str:
    if first == second:
        return first
    return "mixed"
