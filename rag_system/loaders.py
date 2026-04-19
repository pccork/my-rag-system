from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from rag_system.models import DocumentPage


def find_pdfs(docs_dir: Path) -> list[Path]:
    if not docs_dir.exists():
        return []
    return sorted(path for path in docs_dir.rglob("*.pdf") if path.is_file())


def load_pdf(path: Path, docs_dir: Path | None = None) -> list[DocumentPage]:
    reader = PdfReader(str(path))
    source_path = str(path)
    source_name = str(path.relative_to(docs_dir)) if docs_dir else path.name
    document_type = infer_document_type(path.name)
    pages: list[DocumentPage] = []
    current_section: str | None = None

    for index, page in enumerate(reader.pages, start=1):
        raw_text = extract_page_text(page)
        structured_text, current_section = structure_text(raw_text, current_section)
        if not structured_text.strip():
            continue
        pages.append(
            DocumentPage(
                source_path=source_path,
                source_name=source_name,
                filename=path.name,
                page_number=index,
                text=structured_text,
                section_title=current_section,
                document_type=document_type,
                metadata={
                    "filename": path.name,
                    "page_number": index,
                    "section_title": current_section or "",
                    "document_type": document_type,
                    "total_pages": len(reader.pages),
                },
            )
        )

    return pages


def load_pdfs(docs_dir: Path) -> list[DocumentPage]:
    pages: list[DocumentPage] = []
    for pdf_path in find_pdfs(docs_dir):
        pages.extend(load_pdf(pdf_path, docs_dir=docs_dir))
    return pages


def extract_page_text(page: object) -> str:
    try:
        return page.extract_text(extraction_mode="layout") or ""
    except TypeError:
        return page.extract_text() or ""


def structure_text(text: str, current_section: str | None = None) -> tuple[str, str | None]:
    lines = [normalize_line(line) for line in text.splitlines()]
    structured: list[str] = []
    in_table = False
    in_warning = False

    for line in lines:
        if not line:
            if in_table:
                structured.append("[/TABLE]")
                in_table = False
            if in_warning:
                structured.append("[/WARNING]")
                in_warning = False
            append_blank(structured)
            continue

        if is_heading(line):
            if in_table:
                structured.append("[/TABLE]")
                in_table = False
            if in_warning:
                structured.append("[/WARNING]")
                in_warning = False
            current_section = clean_heading(line)
            append_blank(structured)
            structured.append(f"# {current_section}")
            append_blank(structured)
            continue

        if is_warning(line):
            if in_table:
                structured.append("[/TABLE]")
                in_table = False
            if not in_warning:
                structured.append("[WARNING]")
                in_warning = True
            structured.append(line)
            continue

        if is_table_row(line):
            if in_warning:
                structured.append("[/WARNING]")
                in_warning = False
            if not in_table:
                structured.append("[TABLE]")
                in_table = True
            structured.append(format_table_row(line))
            continue

        if in_table:
            structured.append("[/TABLE]")
            in_table = False
        if in_warning:
            structured.append("[/WARNING]")
            in_warning = False
        structured.append(line)

    if in_table:
        structured.append("[/TABLE]")
    if in_warning:
        structured.append("[/WARNING]")

    return clean_structured_text(structured), current_section


def infer_document_type(filename: str) -> str:
    normalized = filename.lower()
    if re.search(r"(^|[_\-\s])sop([_\-\s.]|$)", normalized):
        return "SOP"
    if re.search(r"(^|[_\-\s])ifu([_\-\s.]|$)", normalized):
        return "IFU"
    return "unknown"


def normalize_line(line: str) -> str:
    return " ".join(line.replace("\x00", "").split())


def is_heading(line: str) -> bool:
    if len(line) > 120 or line.endswith((".", ",", ";")):
        return False
    numbered = re.match(r"^\d+(\.\d+)*\.?\s+[A-Z][A-Za-z0-9 /():,&-]+$", line)
    title_case = re.match(r"^[A-Z][A-Za-z0-9 /():,&-]{3,}$", line)
    return bool(numbered or line.isupper() or title_case and len(line.split()) <= 10)


def clean_heading(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip(" :-")


def is_warning(line: str) -> bool:
    return bool(re.match(r"^(warning|caution|danger|important|note)\b[:\s-]*", line, re.I))


def is_table_row(line: str) -> bool:
    if "|" in line or "\t" in line:
        return True
    columns = re.split(r"\s{2,}", line)
    return len([column for column in columns if column.strip()]) >= 3


def format_table_row(line: str) -> str:
    columns = [column.strip() for column in re.split(r"\s{2,}|\t|\|", line) if column.strip()]
    return " | ".join(columns) if len(columns) > 1 else line


def append_blank(lines: list[str]) -> None:
    if lines and lines[-1] != "":
        lines.append("")


def clean_structured_text(lines: list[str]) -> str:
    cleaned: list[str] = []
    previous_blank = False

    for line in lines:
        is_blank = line == ""
        if is_blank and previous_blank:
            continue
        cleaned.append(line)
        previous_blank = is_blank

    return "\n".join(cleaned).strip()
