import gzip
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO
from itertools import zip_longest
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

import pymupdf
from docx import Document as DocxDocument
from openpyxl import load_workbook

from .ocr import OcrProvider, RapidOcrProvider

PARSER_VERSION = "parser-v4"
ARCHIVE_MEMBER_SUFFIXES = frozenset(
    {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xlsx", ".xlsm", ".xls", ".txt", ".md"}
)


@dataclass(frozen=True)
class ParsedBlock:
    text: str
    page_no: int | None
    section_path: tuple[str, ...] = ()
    bbox: tuple[float, float, float, float] | None = None
    block_type: str = "paragraph"
    confidence: float | None = None


@dataclass(frozen=True)
class ParsedPage:
    page_no: int | None
    blocks: tuple[ParsedBlock, ...]


@dataclass(frozen=True)
class ParsedDocument:
    source_path: str
    parser_version: str
    pages: tuple[ParsedPage, ...]

    @property
    def blocks(self) -> tuple[ParsedBlock, ...]:
        return tuple(block for page in self.pages for block in page.blocks)


class UnsupportedFileTypeError(ValueError):
    pass


class ArchiveLimitError(ValueError):
    pass


class OfficeParsingError(ValueError):
    pass


def normalize_text(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def parse_document(
    path: str | Path,
    *,
    ocr_enabled: bool = False,
    ocr_min_text_chars_per_page: int = 40,
    ocr_render_dpi: int = 150,
    ocr_provider: OcrProvider | None = None,
    archive_max_members: int = 500,
    archive_max_member_bytes: int = 50 * 1024 * 1024,
    archive_max_uncompressed_bytes: int = 200 * 1024 * 1024,
    archive_max_compression_ratio: int = 100,
) -> ParsedDocument:
    resolved = Path(path).resolve(strict=True)
    suffix = resolved.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(
            resolved,
            ocr_enabled=ocr_enabled,
            ocr_min_text_chars_per_page=ocr_min_text_chars_per_page,
            ocr_render_dpi=ocr_render_dpi,
            ocr_provider=ocr_provider,
        )
    if suffix == ".docx":
        return parse_docx(
            resolved,
            ocr_enabled=ocr_enabled,
            ocr_min_text_chars=ocr_min_text_chars_per_page,
            ocr_provider=ocr_provider,
        )
    if suffix == ".doc":
        return parse_doc(resolved)
    if suffix in {".ppt", ".pptx"}:
        return parse_presentation(resolved)
    if suffix in {".xlsx", ".xlsm"}:
        return parse_xlsx(resolved)
    if suffix == ".xls":
        return parse_xls(resolved)
    if suffix == ".zip":
        return parse_zip(
            resolved,
            ocr_enabled=ocr_enabled,
            ocr_min_text_chars_per_page=ocr_min_text_chars_per_page,
            ocr_render_dpi=ocr_render_dpi,
            ocr_provider=ocr_provider,
            max_members=archive_max_members,
            max_member_bytes=archive_max_member_bytes,
            max_uncompressed_bytes=archive_max_uncompressed_bytes,
            max_compression_ratio=archive_max_compression_ratio,
        )
    if suffix in {".tar", ".gz"}:
        return parse_tar_or_gzip(
            resolved,
            ocr_enabled=ocr_enabled,
            ocr_min_text_chars_per_page=ocr_min_text_chars_per_page,
            ocr_render_dpi=ocr_render_dpi,
            ocr_provider=ocr_provider,
            max_members=archive_max_members,
            max_member_bytes=archive_max_member_bytes,
            max_uncompressed_bytes=archive_max_uncompressed_bytes,
            max_compression_ratio=archive_max_compression_ratio,
        )
    if suffix in {".rar", ".7z"}:
        return parse_external_archive(
            resolved,
            ocr_enabled=ocr_enabled,
            ocr_min_text_chars_per_page=ocr_min_text_chars_per_page,
            ocr_render_dpi=ocr_render_dpi,
            ocr_provider=ocr_provider,
            max_members=archive_max_members,
            max_member_bytes=archive_max_member_bytes,
            max_uncompressed_bytes=archive_max_uncompressed_bytes,
            max_compression_ratio=archive_max_compression_ratio,
        )
    if suffix == ".txt":
        return parse_text_file(resolved, markdown=False)
    if suffix == ".md":
        return parse_text_file(resolved, markdown=True)
    raise UnsupportedFileTypeError(f"unsupported file type: {suffix}")


def parse_pdf(
    path: str | Path,
    *,
    ocr_enabled: bool = False,
    ocr_min_text_chars_per_page: int = 40,
    ocr_render_dpi: int = 150,
    ocr_provider: OcrProvider | None = None,
) -> ParsedDocument:
    resolved = Path(path).resolve(strict=True)
    with pymupdf.open(resolved) as document:
        return _parse_pdf_document(
            document=document,
            source_path=str(resolved),
            ocr_enabled=ocr_enabled,
            ocr_min_text_chars_per_page=ocr_min_text_chars_per_page,
            ocr_render_dpi=ocr_render_dpi,
            ocr_provider=ocr_provider,
        )


def _parse_pdf_document(
    *,
    document,
    source_path: str,
    ocr_enabled: bool,
    ocr_min_text_chars_per_page: int,
    ocr_render_dpi: int,
    ocr_provider: OcrProvider | None,
) -> ParsedDocument:
    pages: list[ParsedPage] = []
    provider = ocr_provider
    for page_index, page in enumerate(document, start=1):
        blocks: list[ParsedBlock] = []
        for block in page.get_text("blocks", sort=True):
            x0, y0, x1, y1, text, *_ = block
            normalized = normalize_text(text)
            if not normalized:
                continue
            blocks.append(
                ParsedBlock(
                    text=normalized,
                    page_no=page_index,
                    bbox=(float(x0), float(y0), float(x1), float(y1)),
                )
            )
        if ocr_enabled and _text_character_count(blocks) < ocr_min_text_chars_per_page:
            if provider is None:
                provider = RapidOcrProvider()
            for ocr_block in provider.extract_page(page=page, dpi=ocr_render_dpi):
                blocks.append(
                    ParsedBlock(
                        text=ocr_block.text,
                        page_no=page_index,
                        bbox=ocr_block.bbox,
                        block_type="ocr",
                        confidence=ocr_block.confidence,
                    )
                )
        pages.append(ParsedPage(page_no=page_index, blocks=tuple(blocks)))
    return ParsedDocument(
        source_path=source_path,
        parser_version=PARSER_VERSION,
        pages=tuple(pages),
    )


def parse_docx(
    path: str | Path,
    *,
    ocr_enabled: bool = False,
    ocr_min_text_chars: int = 40,
    ocr_provider: OcrProvider | None = None,
) -> ParsedDocument:
    resolved = Path(path).resolve(strict=True)
    return _parse_docx_document(
        document=DocxDocument(str(resolved)),
        source_path=str(resolved),
        ocr_enabled=ocr_enabled,
        ocr_min_text_chars=ocr_min_text_chars,
        ocr_provider=ocr_provider,
    )


def _parse_docx_document(
    *,
    document,
    source_path: str,
    ocr_enabled: bool,
    ocr_min_text_chars: int,
    ocr_provider: OcrProvider | None,
) -> ParsedDocument:
    blocks: list[ParsedBlock] = []
    section_stack: list[str] = []

    for paragraph in document.paragraphs:
        text = normalize_text(paragraph.text)
        if not text:
            continue
        style_name = paragraph.style.name if paragraph.style is not None else ""
        heading_level = _heading_level(style_name)
        if heading_level is not None:
            section_stack = section_stack[: heading_level - 1]
            section_stack.append(text)
            block_type = "heading"
        else:
            block_type = "paragraph"
        blocks.append(
            ParsedBlock(
                text=text,
                page_no=None,
                section_path=tuple(section_stack),
                block_type=block_type,
            )
        )

    for table in document.tables:
        rows: list[str] = []
        for row in table.rows:
            cells = [normalize_text(cell.text) for cell in row.cells]
            if any(cells):
                rows.append("\t".join(cells))
        table_text = "\n".join(rows)
        if table_text:
            blocks.append(
                ParsedBlock(
                    text=table_text,
                    page_no=None,
                    section_path=tuple(section_stack),
                    block_type="table",
                )
            )

    if ocr_enabled and _text_character_count(blocks) < ocr_min_text_chars:
        provider = ocr_provider or RapidOcrProvider()
        seen_parts: set[str] = set()
        for relation in document.part.rels.values():
            if not relation.reltype.endswith("/image"):
                continue
            image_part = relation.target_part
            part_name = str(image_part.partname)
            if part_name in seen_parts:
                continue
            seen_parts.add(part_name)
            for ocr_block in provider.extract_image(image_bytes=image_part.blob):
                blocks.append(
                    ParsedBlock(
                        text=ocr_block.text,
                        page_no=None,
                        section_path=tuple(section_stack),
                        bbox=None,
                        block_type="ocr",
                        confidence=ocr_block.confidence,
                    )
                )

    return ParsedDocument(
        source_path=source_path,
        parser_version=PARSER_VERSION,
        pages=(ParsedPage(page_no=None, blocks=tuple(blocks)),),
    )


def parse_doc(path: str | Path) -> ParsedDocument:
    """Extract legacy Word content through the locally installed Microsoft Word."""
    resolved = Path(path).resolve(strict=True)
    pythoncom, client = _office_com_modules("Microsoft Word")
    application = document = None
    try:
        application = client.DispatchEx("Word.Application")
        application.Visible = False
        application.DisplayAlerts = 0
        document = application.Documents.Open(
            str(resolved),
            ConfirmConversions=False,
            ReadOnly=True,
            AddToRecentFiles=False,
            Visible=False,
            OpenAndRepair=True,
        )
        blocks: list[ParsedBlock] = []
        for index in range(1, document.Paragraphs.Count + 1):
            text = normalize_text(str(document.Paragraphs(index).Range.Text))
            if text:
                blocks.append(ParsedBlock(text=text, page_no=None))
        for table_index in range(1, document.Tables.Count + 1):
            table = document.Tables(table_index)
            rows: list[str] = []
            for row_index in range(1, table.Rows.Count + 1):
                row = table.Rows(row_index)
                cells = [normalize_text(str(cell.Range.Text)) for cell in row.Cells]
                if any(cells):
                    rows.append("\t".join(cells))
            if rows:
                blocks.append(ParsedBlock(text="\n".join(rows), page_no=None, block_type="table"))
        return ParsedDocument(
            source_path=str(resolved),
            parser_version=PARSER_VERSION,
            pages=(ParsedPage(page_no=None, blocks=tuple(blocks)),),
        )
    except Exception as exc:
        raise OfficeParsingError(f"unable to parse legacy Word document: {resolved.name}") from exc
    finally:
        if document is not None:
            document.Close(False)
        if application is not None:
            application.Quit()
        pythoncom.CoUninitialize()


def parse_presentation(path: str | Path) -> ParsedDocument:
    """Extract presentation content without relying on interactive Office automation."""
    resolved = Path(path).resolve(strict=True)
    if resolved.suffix.lower() == ".pptx":
        return parse_pptx(resolved)
    return _parse_legacy_presentation(resolved)


def parse_pptx(path: str | Path) -> ParsedDocument:
    """Read PPTX slide text directly from its Open XML package without COM automation."""
    resolved = Path(path).resolve(strict=True)
    try:
        with ZipFile(resolved) as package:
            slide_names = sorted(
                (
                    name
                    for name in package.namelist()
                    if re.fullmatch(r"ppt/slides/slide[0-9]+\.xml", name)
                ),
                key=_pptx_slide_sort_key,
            )
            pages = []
            for slide_index, slide_name in enumerate(slide_names, start=1):
                root = ElementTree.fromstring(package.read(slide_name))
                text_parts = [
                    normalized
                    for node in root.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}t")
                    if (normalized := normalize_text(node.text or ""))
                ]
                blocks = (
                    (ParsedBlock(text="\n".join(text_parts), page_no=slide_index),)
                    if text_parts
                    else ()
                )
                pages.append(ParsedPage(page_no=slide_index, blocks=blocks))
    except (BadZipFile, ElementTree.ParseError, KeyError, OSError) as exc:
        raise OfficeParsingError(f"unable to parse PPTX presentation: {resolved.name}") from exc
    if not pages:
        raise OfficeParsingError(f"PPTX contains no slides: {resolved.name}")
    return ParsedDocument(
        source_path=str(resolved), parser_version=PARSER_VERSION, pages=tuple(pages)
    )


def _pptx_slide_sort_key(slide_name: str) -> int:
    match = re.search(r"slide([0-9]+)\.xml$", slide_name)
    return int(match.group(1)) if match else 0


def _parse_legacy_presentation(resolved: Path) -> ParsedDocument:
    """Extract legacy PPT text boxes and tables through local PowerPoint."""
    pythoncom, client = _office_com_modules("Microsoft PowerPoint")
    application = presentation = None
    try:
        with tempfile.TemporaryDirectory(prefix="enterprise-rag-ppt-") as directory:
            temporary_path = Path(directory) / "presentation.ppt"
            shutil.copyfile(resolved, temporary_path)
            application = client.DispatchEx("PowerPoint.Application")
            presentation = application.Presentations.Open(str(temporary_path), 1, 0, 0)
            pages: list[ParsedPage] = []
            for slide_index in range(1, presentation.Slides.Count + 1):
                slide = presentation.Slides(slide_index)
                blocks: list[ParsedBlock] = []
                for shape_index in range(1, slide.Shapes.Count + 1):
                    shape = slide.Shapes(shape_index)
                    if shape.HasTable:
                        table = shape.Table
                        rows = []
                        for row_index in range(1, table.Rows.Count + 1):
                            cells = [
                                normalize_text(
                                    str(
                                        table.Cell(row_index, col_index)
                                        .Shape.TextFrame.TextRange.Text
                                    )
                                )
                                for col_index in range(1, table.Columns.Count + 1)
                            ]
                            if any(cells):
                                rows.append("\t".join(cells))
                        if rows:
                            blocks.append(
                                ParsedBlock(
                                    text="\n".join(rows),
                                    page_no=slide_index,
                                    block_type="table",
                                )
                            )
                        continue
                    if shape.HasTextFrame and shape.TextFrame.HasText:
                        text = normalize_text(str(shape.TextFrame.TextRange.Text))
                        if text:
                            blocks.append(ParsedBlock(text=text, page_no=slide_index))
                pages.append(ParsedPage(page_no=slide_index, blocks=tuple(blocks)))
        return ParsedDocument(
            source_path=str(resolved), parser_version=PARSER_VERSION, pages=tuple(pages)
        )
    except Exception as exc:
        raise OfficeParsingError(f"unable to parse presentation: {resolved.name}") from exc
    finally:
        if presentation is not None:
            presentation.Close()
        if application is not None:
            application.Quit()
        pythoncom.CoUninitialize()


def _office_com_modules(application_name: str):
    try:
        import pythoncom
        import win32com.client
    except ImportError as exc:
        raise OfficeParsingError(f"{application_name} integration is unavailable") from exc
    pythoncom.CoInitialize()
    return pythoncom, win32com.client


def parse_text_file(path: str | Path, *, markdown: bool) -> ParsedDocument:
    resolved = Path(path).resolve(strict=True)
    return _parse_text_content(
        text=resolved.read_text(encoding="utf-8"),
        source_path=str(resolved),
        markdown=markdown,
    )


def _parse_text_content(*, text: str, source_path: str, markdown: bool) -> ParsedDocument:
    blocks: list[ParsedBlock] = []
    section_stack: list[str] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        text = normalize_text("\n".join(paragraph_lines))
        paragraph_lines.clear()
        if text:
            blocks.append(
                ParsedBlock(
                    text=text,
                    page_no=None,
                    section_path=tuple(section_stack),
                )
            )

    for raw_line in text.splitlines():
        if markdown:
            heading = re.match(r"^(#{1,6})\s+(.+)$", raw_line)
            if heading:
                flush_paragraph()
                level = len(heading.group(1))
                title = normalize_text(heading.group(2))
                section_stack = section_stack[: level - 1]
                section_stack.append(title)
                blocks.append(
                    ParsedBlock(
                        text=title,
                        page_no=None,
                        section_path=tuple(section_stack),
                        block_type="heading",
                    )
                )
                continue
        if raw_line.strip():
            paragraph_lines.append(raw_line)
        else:
            flush_paragraph()
    flush_paragraph()

    return ParsedDocument(
        source_path=source_path,
        parser_version=PARSER_VERSION,
        pages=(ParsedPage(page_no=None, blocks=tuple(blocks)),),
    )


def parse_xlsx(path: str | Path) -> ParsedDocument:
    """Parse an Excel workbook into row blocks with formula and cached value context."""
    resolved = Path(path).resolve(strict=True)
    formula_workbook = load_workbook(filename=resolved, read_only=True, data_only=False)
    value_workbook = load_workbook(filename=resolved, read_only=True, data_only=True)
    try:
        return _parse_xlsx_workbooks(
            formula_workbook=formula_workbook,
            value_workbook=value_workbook,
            source_path=str(resolved),
        )
    finally:
        formula_workbook.close()
        value_workbook.close()


def _parse_xlsx_workbooks(*, formula_workbook, value_workbook, source_path: str) -> ParsedDocument:
    pages = []
    value_sheets = {worksheet.title: worksheet for worksheet in value_workbook.worksheets}
    for formula_sheet in formula_workbook.worksheets:
        value_sheet = value_sheets.get(formula_sheet.title)
        if value_sheet is None:
            raise ValueError(f"cached-value worksheet is missing: {formula_sheet.title}")
        pages.append(
            ParsedPage(
                page_no=None,
                blocks=tuple(
                    _excel_row_blocks(
                        sheet_name=formula_sheet.title,
                        rows=_xlsx_rows(
                            formula_rows=formula_sheet.iter_rows(values_only=True),
                            value_rows=value_sheet.iter_rows(values_only=True),
                        ),
                    )
                ),
            )
        )
    return ParsedDocument(
        source_path=source_path,
        parser_version=PARSER_VERSION,
        pages=tuple(pages),
    )


def _xlsx_rows(*, formula_rows, value_rows) -> Iterable[tuple[int, list[Any]]]:
    for row_number, (formula_row, value_row) in enumerate(
        zip_longest(formula_rows, value_rows, fillvalue=()), start=1
    ):
        yield (
            row_number,
            [
                _formula_with_cached_value(formula_value, cached_value)
                for formula_value, cached_value in zip_longest(
                    formula_row, value_row, fillvalue=None
                )
            ],
        )


def _formula_with_cached_value(formula_value: Any, cached_value: Any) -> Any:
    if (
        isinstance(formula_value, str)
        and formula_value.startswith("=")
        and cached_value is not None
    ):
        return f"{formula_value}（计算值：{_display_excel_value(cached_value)}）"
    return formula_value


def parse_xls(path: str | Path) -> ParsedDocument:
    """Parse a legacy XLS workbook using xlrd when the source requires it."""
    import xlrd

    resolved = Path(path).resolve(strict=True)
    workbook = xlrd.open_workbook(str(resolved), on_demand=True)
    try:
        return _parse_xls_workbook(workbook=workbook, source_path=str(resolved))
    finally:
        workbook.release_resources()


def _parse_xls_workbook(*, workbook, source_path: str) -> ParsedDocument:
    pages = []
    for sheet_name in workbook.sheet_names():
        sheet = workbook.sheet_by_name(sheet_name)
        rows = (
            (row_index, sheet.row_values(row_index - 1))
            for row_index in range(1, sheet.nrows + 1)
        )
        pages.append(
            ParsedPage(
                page_no=None,
                blocks=tuple(_excel_row_blocks(sheet_name=sheet_name, rows=rows)),
            )
        )
    return ParsedDocument(
        source_path=source_path,
        parser_version=PARSER_VERSION,
        pages=tuple(pages),
    )


def parse_tar_or_gzip(
    path: str | Path,
    *,
    ocr_enabled: bool,
    ocr_min_text_chars_per_page: int,
    ocr_render_dpi: int,
    ocr_provider: OcrProvider | None,
    max_members: int,
    max_member_bytes: int,
    max_uncompressed_bytes: int,
    max_compression_ratio: int,
) -> ParsedDocument:
    resolved = Path(path).resolve(strict=True)
    if tarfile.is_tarfile(resolved):
        return _parse_tar_archive(
            resolved,
            ocr_enabled=ocr_enabled,
            ocr_min_text_chars_per_page=ocr_min_text_chars_per_page,
            ocr_render_dpi=ocr_render_dpi,
            ocr_provider=ocr_provider,
            max_members=max_members,
            max_member_bytes=max_member_bytes,
            max_uncompressed_bytes=max_uncompressed_bytes,
            max_compression_ratio=max_compression_ratio,
        )

    member_path = resolved.with_suffix("").name
    if not _is_supported_archive_member(member_path):
        raise ValueError("gzip file does not contain a supported document type")
    with gzip.open(resolved, "rb") as stream:
        contents = _read_limited(stream, max_member_bytes)
    if len(contents) > max_uncompressed_bytes:
        raise ArchiveLimitError("archive uncompressed size limit exceeded")
    return _archive_document_from_pages(
        resolved,
        _parse_archive_member_pages(
            member_path=member_path,
            contents=contents,
            ocr_enabled=ocr_enabled,
            ocr_min_text_chars_per_page=ocr_min_text_chars_per_page,
            ocr_render_dpi=ocr_render_dpi,
            ocr_provider=ocr_provider,
        ),
    )


def _parse_tar_archive(
    resolved: Path,
    *,
    ocr_enabled: bool,
    ocr_min_text_chars_per_page: int,
    ocr_render_dpi: int,
    ocr_provider: OcrProvider | None,
    max_members: int,
    max_member_bytes: int,
    max_uncompressed_bytes: int,
    max_compression_ratio: int,
) -> ParsedDocument:
    pages: list[ParsedPage] = []
    total_uncompressed_bytes = 0
    archive_size = max(resolved.stat().st_size, 1)
    with tarfile.open(resolved, mode="r:*") as archive:
        members = [item for item in archive.getmembers() if item.isfile()]
        if len(members) > max_members:
            raise ArchiveLimitError(f"archive contains more than {max_members} files")
        for member in members:
            if not _is_supported_archive_member(member.name):
                continue
            if member.size > max_member_bytes:
                raise ArchiveLimitError(
                    f"archive member exceeds {max_member_bytes} bytes: {member.name}"
                )
            total_uncompressed_bytes += member.size
            _validate_archive_size_limits(
                total_uncompressed_bytes=total_uncompressed_bytes,
                archive_size=archive_size,
                max_uncompressed_bytes=max_uncompressed_bytes,
                max_compression_ratio=max_compression_ratio,
            )
            stream = archive.extractfile(member)
            if stream is None:
                continue
            with stream:
                contents = _read_limited(stream, max_member_bytes)
            pages.extend(
                _parse_archive_member_pages(
                    member_path=member.name,
                    contents=contents,
                    ocr_enabled=ocr_enabled,
                    ocr_min_text_chars_per_page=ocr_min_text_chars_per_page,
                    ocr_render_dpi=ocr_render_dpi,
                    ocr_provider=ocr_provider,
                )
            )
    return _archive_document_from_pages(resolved, pages)


def parse_external_archive(
    path: str | Path,
    *,
    ocr_enabled: bool,
    ocr_min_text_chars_per_page: int,
    ocr_render_dpi: int,
    ocr_provider: OcrProvider | None,
    max_members: int,
    max_member_bytes: int,
    max_uncompressed_bytes: int,
    max_compression_ratio: int,
) -> ParsedDocument:
    """Read RAR and 7z members through the locally available libarchive tool."""
    resolved = Path(path).resolve(strict=True)
    bsdtar = _find_bsdtar()
    try:
        listed = subprocess.run(
            [bsdtar, "-tf", str(resolved)],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"unable to open archive: {resolved.name}") from exc

    member_names = _decode_external_archive_names(listed.stdout).splitlines()
    members = [line.strip() for line in member_names if line.strip()]
    if len(members) > max_members:
        raise ArchiveLimitError(f"archive contains more than {max_members} files")

    pages: list[ParsedPage] = []
    total_uncompressed_bytes = 0
    archive_size = max(resolved.stat().st_size, 1)
    for member_path in members:
        if not _is_supported_archive_member(member_path):
            continue
        contents = _read_external_archive_member(bsdtar, resolved, member_path, max_member_bytes)
        total_uncompressed_bytes += len(contents)
        _validate_archive_size_limits(
            total_uncompressed_bytes=total_uncompressed_bytes,
            archive_size=archive_size,
            max_uncompressed_bytes=max_uncompressed_bytes,
            max_compression_ratio=max_compression_ratio,
        )
        pages.extend(
            _parse_archive_member_pages(
                member_path=member_path,
                contents=contents,
                ocr_enabled=ocr_enabled,
                ocr_min_text_chars_per_page=ocr_min_text_chars_per_page,
                ocr_render_dpi=ocr_render_dpi,
                ocr_provider=ocr_provider,
            )
        )
    return _archive_document_from_pages(resolved, pages)


def _find_bsdtar() -> str:
    configured = os.environ.get("BSDTAR_PATH")
    candidate = configured or shutil.which("bsdtar") or shutil.which("tar")
    if candidate is None:
        raise ValueError("RAR and 7z parsing requires a local bsdtar executable")
    return candidate


def _decode_external_archive_names(contents: bytes) -> str:
    for encoding in ("utf-8", "mbcs", "gb18030"):
        try:
            return contents.decode(encoding)
        except UnicodeDecodeError:
            continue
    return contents.decode("gb18030", errors="replace")


def _read_external_archive_member(
    bsdtar: str,
    archive_path: Path,
    member_path: str,
    max_member_bytes: int,
) -> bytes:
    process = subprocess.Popen(
        [bsdtar, "-xOf", str(archive_path), member_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    try:
        contents = _read_limited(process.stdout, max_member_bytes)
        _, stderr = process.communicate()
    except Exception:
        process.kill()
        process.communicate()
        raise
    if process.returncode != 0:
        raise ValueError(stderr.decode("utf-8", errors="replace").strip() or "archive read failed")
    return contents


def _read_limited(stream, max_bytes: int) -> bytes:
    contents = stream.read(max_bytes + 1)
    if len(contents) > max_bytes:
        raise ArchiveLimitError(f"archive member exceeds {max_bytes} bytes")
    return contents


def _validate_archive_size_limits(
    *,
    total_uncompressed_bytes: int,
    archive_size: int,
    max_uncompressed_bytes: int,
    max_compression_ratio: int,
) -> None:
    if total_uncompressed_bytes > max_uncompressed_bytes:
        raise ArchiveLimitError("archive uncompressed size limit exceeded")
    if total_uncompressed_bytes > archive_size * max_compression_ratio:
        raise ArchiveLimitError("archive compression ratio limit exceeded")


def _is_supported_archive_member(member_path: str) -> bool:
    normalized = member_path.replace("\\", "/")
    return (
        Path(normalized).suffix.lower() in ARCHIVE_MEMBER_SUFFIXES
        and ".." not in Path(normalized).parts
        and not normalized.startswith(("/", "__MACOSX/"))
    )


def _parse_archive_member_pages(
    *,
    member_path: str,
    contents: bytes,
    ocr_enabled: bool,
    ocr_min_text_chars_per_page: int,
    ocr_render_dpi: int,
    ocr_provider: OcrProvider | None,
) -> list[ParsedPage]:
    try:
        parsed = _parse_archive_member(
            member_path=member_path,
            contents=contents,
            ocr_enabled=ocr_enabled,
            ocr_min_text_chars_per_page=ocr_min_text_chars_per_page,
            ocr_render_dpi=ocr_render_dpi,
            ocr_provider=ocr_provider,
        )
    except (BadZipFile, OSError, RuntimeError, ValueError):
        return []
    return _prefix_archive_member(parsed=parsed, member_path=member_path)


def _archive_document_from_pages(resolved: Path, pages: list[ParsedPage]) -> ParsedDocument:
    if not pages:
        raise ValueError("archive contains no readable supported documents")
    return ParsedDocument(
        source_path=str(resolved), parser_version=PARSER_VERSION, pages=tuple(pages)
    )


def parse_zip(
    path: str | Path,
    *,
    ocr_enabled: bool,
    ocr_min_text_chars_per_page: int,
    ocr_render_dpi: int,
    ocr_provider: OcrProvider | None,
    max_members: int,
    max_member_bytes: int,
    max_uncompressed_bytes: int,
    max_compression_ratio: int,
) -> ParsedDocument:
    """Read supported archive members in memory without extracting them to disk."""
    resolved = Path(path).resolve(strict=True)
    pages: list[ParsedPage] = []
    total_uncompressed_bytes = 0
    try:
        with ZipFile(resolved) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            if len(members) > max_members:
                raise ArchiveLimitError(f"archive contains more than {max_members} files")
            for member in members:
                member_path = member.filename.replace("\\", "/")
                if (
                    not _is_supported_archive_member(member_path)
                    or member.flag_bits & 0x1
                ):
                    continue
                if member.file_size > max_member_bytes:
                    raise ArchiveLimitError(
                        f"archive member exceeds {max_member_bytes} bytes: {member_path}"
                    )
                total_uncompressed_bytes += member.file_size
                if total_uncompressed_bytes > max_uncompressed_bytes:
                    raise ArchiveLimitError("archive uncompressed size limit exceeded")
                if (
                    member.file_size > 0
                    and (
                        member.compress_size == 0
                        or member.file_size > member.compress_size * max_compression_ratio
                    )
                ):
                    raise ArchiveLimitError(
                        f"archive compression ratio limit exceeded: {member_path}"
                    )
                contents = archive.read(member)
                pages.extend(
                    _parse_archive_member_pages(
                        member_path=member_path,
                        contents=contents,
                        ocr_enabled=ocr_enabled,
                        ocr_min_text_chars_per_page=ocr_min_text_chars_per_page,
                        ocr_render_dpi=ocr_render_dpi,
                        ocr_provider=ocr_provider,
                    )
                )
    except BadZipFile as exc:
        raise ValueError("invalid ZIP archive") from exc
    if not pages:
        raise ValueError("archive contains no readable supported documents")
    return ParsedDocument(
        source_path=str(resolved),
        parser_version=PARSER_VERSION,
        pages=tuple(pages),
    )


def _parse_archive_member(
    *,
    member_path: str,
    contents: bytes,
    ocr_enabled: bool,
    ocr_min_text_chars_per_page: int,
    ocr_render_dpi: int,
    ocr_provider: OcrProvider | None,
) -> ParsedDocument:
    suffix = Path(member_path).suffix.lower()
    source_path = f"archive://{member_path}"
    if suffix == ".pdf":
        with pymupdf.open(stream=contents, filetype="pdf") as document:
            return _parse_pdf_document(
                document=document,
                source_path=source_path,
                ocr_enabled=ocr_enabled,
                ocr_min_text_chars_per_page=ocr_min_text_chars_per_page,
                ocr_render_dpi=ocr_render_dpi,
                ocr_provider=ocr_provider,
            )
    if suffix == ".docx":
        return _parse_docx_document(
            document=DocxDocument(BytesIO(contents)),
            source_path=source_path,
            ocr_enabled=ocr_enabled,
            ocr_min_text_chars=ocr_min_text_chars_per_page,
            ocr_provider=ocr_provider,
        )
    if suffix == ".doc":
        return _parse_archive_office_member(
            member_path=member_path,
            contents=contents,
            parser=parse_doc,
        )
    if suffix in {".ppt", ".pptx"}:
        return _parse_archive_office_member(
            member_path=member_path,
            contents=contents,
            parser=parse_presentation,
        )
    if suffix in {".xlsx", ".xlsm"}:
        formula_workbook = load_workbook(
            filename=BytesIO(contents), read_only=True, data_only=False
        )
        value_workbook = load_workbook(filename=BytesIO(contents), read_only=True, data_only=True)
        try:
            return _parse_xlsx_workbooks(
                formula_workbook=formula_workbook,
                value_workbook=value_workbook,
                source_path=source_path,
            )
        finally:
            formula_workbook.close()
            value_workbook.close()
    if suffix == ".xls":
        import xlrd

        workbook = xlrd.open_workbook(file_contents=contents, on_demand=True)
        try:
            return _parse_xls_workbook(workbook=workbook, source_path=source_path)
        finally:
            workbook.release_resources()
    return _parse_text_content(
        text=_decode_text_member(contents),
        source_path=source_path,
        markdown=suffix == ".md",
    )


def _parse_archive_office_member(*, member_path: str, contents: bytes, parser) -> ParsedDocument:
    source_path = f"archive://{member_path}"
    suffix = Path(member_path).suffix.lower()
    with tempfile.TemporaryDirectory(prefix="enterprise-rag-office-") as directory:
        member_file = Path(directory) / f"document{suffix}"
        member_file.write_bytes(contents)
        parsed = parser(member_file)
    return ParsedDocument(
        source_path=source_path,
        parser_version=PARSER_VERSION,
        pages=parsed.pages,
    )


def _prefix_archive_member(*, parsed: ParsedDocument, member_path: str) -> list[ParsedPage]:
    prefix = f"压缩包内文件：{member_path}"
    return [
        ParsedPage(
            page_no=page.page_no,
            blocks=tuple(
                ParsedBlock(
                    text=block.text,
                    page_no=block.page_no,
                    section_path=(prefix, *block.section_path),
                    bbox=block.bbox,
                    block_type=block.block_type,
                    confidence=block.confidence,
                )
                for block in page.blocks
            ),
        )
        for page in parsed.pages
    ]


def _decode_text_member(contents: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return contents.decode(encoding)
        except UnicodeDecodeError:
            continue
    return contents.decode("utf-8", errors="replace")


def _excel_row_blocks(
    *,
    sheet_name: str,
    rows: Iterable[tuple[int, list[Any]]],
) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    titles: list[str] = []
    headers: list[str] | None = None
    table_title = ""
    section_path = (f"工作表：{sheet_name}",)

    for row_number, values in rows:
        cells = [_display_excel_value(value) for value in values]
        populated = [(index, value) for index, value in enumerate(cells) if value]
        if not populated:
            continue
        if headers is None:
            if len(populated) < 2:
                titles.append(" ".join(value for _, value in populated))
                continue
            headers = _excel_headers(cells)
            table_title = titles[0] if titles else ""
            title = "\n".join(titles)
            if title:
                blocks.append(
                    ParsedBlock(
                        text=f"工作表：{sheet_name}\n{title}",
                        page_no=None,
                        section_path=section_path,
                        block_type="heading",
                    )
                )
            continue

        fields = [
            f"{headers[index]}：{value}"
            for index, value in populated
            if index < len(headers)
        ]
        if fields:
            context = f"表格标题：{table_title}\n" if table_title else ""
            blocks.append(
                ParsedBlock(
                    text=f"工作表：{sheet_name}\n{context}行：{row_number}\n"
                    + " | ".join(fields),
                    page_no=None,
                    section_path=section_path,
                    block_type="table_row",
                )
            )

    if headers is None and titles:
        blocks.append(
            ParsedBlock(
                text=f"工作表：{sheet_name}\n" + "\n".join(titles),
                page_no=None,
                section_path=section_path,
                block_type="paragraph",
            )
        )
    return blocks


def _excel_headers(values: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    headers: list[str] = []
    for index, value in enumerate(values, start=1):
        base = value or f"列{index}"
        seen[base] = seen.get(base, 0) + 1
        headers.append(base if seen[base] == 1 else f"{base}（{seen[base]}）")
    return headers


def _display_excel_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (datetime, date)):
        return value.isoformat(sep=" ") if isinstance(value, datetime) else value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return normalize_text(str(value))


def _heading_level(style_name: str) -> int | None:
    match = re.match(r"Heading\s+([1-6])$", style_name)
    if not match:
        return None
    return int(match.group(1))


def _text_character_count(blocks: list[ParsedBlock]) -> int:
    return sum(sum(character.isalnum() for character in block.text) for block in blocks)
