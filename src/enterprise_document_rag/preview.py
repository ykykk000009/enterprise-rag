"""Safe, local HTML previews for documents returned by retrieval."""

from __future__ import annotations

import html
import re
from pathlib import Path

from docx import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from .text_utils import clean_display_text, merge_context_texts


def render_document_preview(
    *,
    source_path: Path,
    indexed_chunks: list[dict[str, object]],
    focus_chunk_id: str | None,
) -> str:
    """Render a read-only HTML view without exposing the source download."""
    focus_text = next(
        (
            str(chunk["text"])
            for chunk in indexed_chunks
            if chunk["id"] == focus_chunk_id
        ),
        "",
    )
    if source_path.suffix.lower() == ".docx":
        content = _render_docx(source_path=source_path, focus_text=focus_text)
    else:
        content = _render_indexed_chunks(
            indexed_chunks=indexed_chunks,
            focus_chunk_id=focus_chunk_id,
        )
    return _page(title=source_path.name, source_path=source_path, content=content)


def _render_docx(*, source_path: Path, focus_text: str) -> str:
    document = DocxDocument(str(source_path))
    parts: list[str] = []
    matched = False
    for index, element in enumerate(document.element.body.iterchildren()):
        if isinstance(element, CT_P):
            paragraph = Paragraph(element, document)
            text = " ".join(paragraph.text.split())
            if not text:
                continue
            level = _heading_level(paragraph.style.name if paragraph.style else "")
            is_match = not matched and _is_context_match(text=text, focus_text=focus_text)
            if is_match:
                matched = True
            attributes = ' id="matched-context" class="matched"' if is_match else ""
            if level is not None:
                parts.append(f"<h{level}{attributes}>{html.escape(text)}</h{level}>")
            else:
                parts.append(f"<p{attributes}>{html.escape(text)}</p>")
        elif isinstance(element, CT_Tbl):
            table = Table(element, document)
            rows = []
            for row in table.rows:
                cells = "".join(
                    f"<td>{html.escape(' '.join(cell.text.split()))}</td>"
                    for cell in row.cells
                )
                rows.append(f"<tr>{cells}</tr>")
            if rows:
                parts.append(f"<table><tbody>{''.join(rows)}</tbody></table>")
        del index
    if not parts:
        return '<p class="empty">该 DOCX 没有可预览的文本内容。</p>'
    if focus_text and not matched:
        parts.insert(0, _context_notice(focus_text))
    return "\n".join(parts)


def _render_indexed_chunks(
    *,
    indexed_chunks: list[dict[str, object]],
    focus_chunk_id: str | None,
) -> str:
    if not indexed_chunks:
        return '<p class="empty">该文件尚无可预览的索引内容。</p>'
    if focus_chunk_id is not None:
        focus_index = next(
            index
            for index, chunk in enumerate(indexed_chunks)
            if chunk["id"] == focus_chunk_id
        )
        selected = indexed_chunks[max(0, focus_index - 1) : focus_index + 2]
        focus = indexed_chunks[focus_index]
        position = []
        if focus["page_no"] is not None:
            position.append(f"第 {focus['page_no']} 页")
        if focus["section_path"]:
            position.append(str(focus["section_path"]))
        meta = (
            f'<div class="chunk-meta">{html.escape(" · ".join(position))}</div>'
            if position
            else ""
        )
        context = merge_context_texts([str(chunk["text"]) for chunk in selected])
        return (
            f"{meta}<h2>命中内容上下文</h2>"
            f'<p id="matched-context" class="matched">{html.escape(context)}</p>'
        )
    parts: list[str] = []
    for chunk in indexed_chunks:
        is_match = chunk["id"] == focus_chunk_id
        attributes = ' id="matched-context" class="matched"' if is_match else ""
        position = []
        if chunk["page_no"] is not None:
            position.append(f"第 {chunk['page_no']} 页")
        if chunk["section_path"]:
            position.append(str(chunk["section_path"]))
        if position:
            parts.append(f"<div class=\"chunk-meta\">{html.escape(' · '.join(position))}</div>")
        display_text = clean_display_text(str(chunk["text"]))
        parts.append(f"<p{attributes}>{html.escape(display_text)}</p>")
    return "\n".join(parts)


def _context_notice(focus_text: str) -> str:
    preview = " ".join(focus_text.split())[:280]
    return (
        '<aside id="matched-context" class="matched context-notice">'
        '<strong>检索命中上下文</strong><br>'
        f"{html.escape(preview)}"
        "</aside>"
    )


def _heading_level(style_name: str) -> int | None:
    match = re.search(r"(?:heading|标题)\s*([1-6])", style_name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _is_context_match(*, text: str, focus_text: str) -> bool:
    if not focus_text:
        return False
    normalized_text = "".join(text.split())
    normalized_focus = "".join(focus_text.split())
    if normalized_text and normalized_text in normalized_focus:
        return True
    for phrase in re.split(r"[。！？.!?\n]", normalized_focus):
        if len(phrase) >= 12 and phrase in normalized_text:
            return True
    return False


def _page(*, title: str, source_path: Path, content: str) -> str:
    escaped_title = html.escape(title)
    escaped_path = html.escape(str(source_path))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title} - 在线预览</title>
  <style>
    :root {{
      font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
      color: #17212b;
      background: #f4f6f8;
    }}
    body {{ margin: 0; }}
    header {{ padding: 20px 28px; background: #17212b; color: #fff; }}
    h1 {{ margin: 0; font-size: 20px; font-weight: 650; }}
    .path {{ margin-top: 7px; color: #c8d3dd; font-size: 13px; overflow-wrap: anywhere; }}
    main {{
      max-width: 920px; margin: 24px auto; padding: 28px 34px; background: #fff;
      border: 1px solid #d9e1e8; border-radius: 6px; line-height: 1.75;
    }}
    h2, h3, h4, h5, h6 {{ margin: 1.4em 0 .6em; line-height: 1.35; }}
    p {{ margin: 0 0 1em; white-space: pre-wrap; }}
    table {{ width: 100%; margin: 1em 0; border-collapse: collapse; }}
    td {{
      padding: 7px 9px; border: 1px solid #d9e1e8; vertical-align: top;
      white-space: pre-wrap;
    }}
    .matched {{
      scroll-margin-top: 20px; background: #fff6c8; border-left: 3px solid #c69214;
      padding: 8px 11px;
    }}
    .context-notice {{ margin: 0 0 20px; }}
    .chunk-meta {{ margin-top: 20px; color: #657383; font-size: 12px; }}
    .empty {{ color: #657383; }}
    @media (max-width: 780px) {{
      header {{ padding: 16px; }}
      main {{ margin: 0; border: 0; padding: 22px 16px; }}
    }}
  </style>
</head>
<body>
  <header><h1>{escaped_title}</h1><div class="path">{escaped_path}</div></header>
  <main>{content}</main>
</body>
</html>"""
