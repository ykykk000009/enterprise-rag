"""Text normalization helpers shared by retrieval and preview rendering."""

import re

_METADATA_PREFIXES = (
    "\u6587\u4ef6\u540d\uff1a",
    "\u6587\u4ef6\u8def\u5f84\uff1a",
    "\u538b\u7f29\u5305\u5185\u6587\u4ef6\uff1a",
)


def sanitize_unicode(text: str) -> str:
    """Preserve valid Unicode while replacing malformed UTF-16 surrogates."""
    normalized: list[str] = []
    index = 0
    while index < len(text):
        codepoint = ord(text[index])
        if 0xD800 <= codepoint <= 0xDBFF:
            if index + 1 < len(text):
                next_codepoint = ord(text[index + 1])
                if 0xDC00 <= next_codepoint <= 0xDFFF:
                    normalized.append(
                        chr(0x10000 + ((codepoint - 0xD800) << 10) + next_codepoint - 0xDC00)
                    )
                    index += 2
                    continue
            normalized.append("\ufffd")
        elif 0xDC00 <= codepoint <= 0xDFFF:
            normalized.append("\ufffd")
        else:
            normalized.append(text[index])
        index += 1
    return "".join(normalized)


def clean_display_text(text: str) -> str:
    """Remove retrieval-only metadata and repeated paragraphs from displayed text."""
    text = sanitize_unicode(text)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(prefix) for prefix in _METADATA_PREFIXES):
            continue
        lines.append(line.rstrip())
    return _deduplicate_paragraphs("\n".join(lines))


def merge_context_texts(texts: list[str]) -> str:
    """Merge adjacent chunks while removing exact overlap paragraphs."""
    return _deduplicate_paragraphs("\n\n".join(clean_display_text(text) for text in texts))


def normalized_text_key(text: str) -> str:
    cleaned = clean_display_text(text).lower()
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", cleaned)


def is_near_duplicate(key: str, existing_keys: list[str]) -> bool:
    if not key:
        return True
    for existing in existing_keys:
        if key == existing:
            return True
        shorter, longer = sorted((key, existing), key=len)
        if len(shorter) >= 60 and len(shorter) / len(longer) >= 0.78 and shorter in longer:
            return True
    return False


def _deduplicate_paragraphs(text: str) -> str:
    seen: set[str] = set()
    paragraphs: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        cleaned = "\n".join(line.rstrip() for line in paragraph.strip().splitlines()).strip()
        key = re.sub(r"\s+", "", cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        paragraphs.append(cleaned)
    return "\n\n".join(paragraphs)
