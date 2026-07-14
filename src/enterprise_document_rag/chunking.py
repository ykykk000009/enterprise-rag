import hashlib
import re
from dataclasses import dataclass

from .parsing import ParsedBlock, ParsedDocument

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
        pieces: list[ParsedBlock] = []
        start_offset = 0
        for start_index in range(0, len(matches), self.max_tokens):
            end_index = min(start_index + self.max_tokens, len(matches))
            end_offset = matches[end_index - 1].end()
            text = block.text[start_offset:end_offset].strip()
            if text:
                pieces.append(
                    ParsedBlock(
                        text=text,
                        page_no=block.page_no,
                        section_path=block.section_path,
                        bbox=block.bbox,
                        block_type=f"{block.block_type}:chunk_split",
                        confidence=block.confidence,
                    )
                )
            start_offset = end_offset
        tail = block.text[start_offset:].strip()
        if tail:
            pieces.append(
                ParsedBlock(
                    text=tail,
                    page_no=block.page_no,
                    section_path=block.section_path,
                    bbox=block.bbox,
                    block_type=f"{block.block_type}:chunk_split",
                    confidence=block.confidence,
                )
            )
        return tuple(pieces)

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
        text = "\n\n".join(block.text for block in blocks).strip()
        first = blocks[0]
        return Chunk(
            chunk_index=chunk_index,
            text=text,
            page_no=first.page_no,
            section_path=first.section_path,
            bbox=first.bbox,
            token_count=count_tokens(text),
            text_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            deduplicable=not self._contains_split_block(blocks),
        )

    def _merge_chunk_with_blocks(self, chunk: Chunk, blocks: list[ParsedBlock]) -> Chunk:
        extra_text = "\n\n".join(block.text for block in blocks).strip()
        text = "\n\n".join(item for item in [chunk.text, extra_text] if item)
        return Chunk(
            chunk_index=chunk.chunk_index,
            text=text,
            page_no=chunk.page_no,
            section_path=chunk.section_path,
            bbox=chunk.bbox,
            token_count=count_tokens(text),
            text_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            deduplicable=chunk.deduplicable and not self._contains_split_block(blocks),
        )

    def _with_index(self, chunk: Chunk, chunk_index: int) -> Chunk:
        return Chunk(
            chunk_index=chunk_index,
            text=chunk.text,
            page_no=chunk.page_no,
            section_path=chunk.section_path,
            bbox=chunk.bbox,
            token_count=chunk.token_count,
            text_hash=chunk.text_hash,
            deduplicable=chunk.deduplicable,
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
                Chunk(
                    chunk_index=index,
                    text=chunk.text,
                    page_no=chunk.page_no,
                    section_path=chunk.section_path,
                    bbox=chunk.bbox,
                    token_count=chunk.token_count,
                    text_hash=chunk.text_hash,
                    deduplicable=chunk.deduplicable,
                    previous_chunk_index=index - 1 if index > 0 else None,
                    next_chunk_index=index + 1 if index < len(chunks) - 1 else None,
                )
            )
        return tuple(linked)


def count_tokens(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))
