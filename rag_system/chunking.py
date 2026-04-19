from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass

from rag_system.models import DocumentChunk, DocumentPage


MIN_TOKENS = 300
MAX_TOKENS = 800
OVERSIZE_OVERLAP_TOKENS = 40

HEADING_RE = re.compile(r"^#\s+(.+)$")
STEP_RE = re.compile(r"^((?:\d+(?:\.\d+)*[\).]?)|(?:step\s+\d+[:.)-]?))\s+", re.I)
WARNING_START_RE = re.compile(r"^\[(WARNING|CAUTION)\]$", re.I)
WARNING_LINE_RE = re.compile(r"^(warning|caution|danger|important|note)\b[:\s-]*", re.I)
TABLE_START_RE = re.compile(r"^\[TABLE\]$", re.I)
MAINTENANCE_RE = re.compile(
    r"\b(maintenance|cleaning|calibration|service|servicing|inspection|troubleshooting)\b",
    re.I,
)


@dataclass(frozen=True)
class SemanticBlock:
    text: str
    source_path: str
    source_name: str
    filename: str
    page_start: int
    page_end: int
    section_title: str
    document_type: str
    block_type: str

    @property
    def token_count(self) -> int:
        return count_tokens(self.text)


def chunk_pages(
    pages: Iterable[DocumentPage],
    chunk_size: int = 650,
    chunk_overlap: int = 50,
) -> list[DocumentChunk]:
    target_tokens = clamp(chunk_size, MIN_TOKENS, MAX_TOKENS)
    overlap_tokens = min(max(chunk_overlap, 0), OVERSIZE_OVERLAP_TOKENS)
    chunks: list[DocumentChunk] = []

    for source_pages in group_pages_by_source(pages):
        blocks = build_semantic_blocks(source_pages)
        chunks.extend(pack_blocks(blocks, target_tokens, overlap_tokens, len(chunks)))

    return chunks


def group_pages_by_source(pages: Iterable[DocumentPage]) -> list[list[DocumentPage]]:
    grouped: dict[str, list[DocumentPage]] = {}
    for page in pages:
        grouped.setdefault(page.source_path, []).append(page)
    return [sorted(group, key=lambda page: page.page_number) for group in grouped.values()]


def build_semantic_blocks(pages: list[DocumentPage]) -> list[SemanticBlock]:
    blocks: list[SemanticBlock] = []
    current_section = ""
    pending_lines: list[str] = []
    pending_type = "body"
    pending_page_start: int | None = None
    pending_page_end: int | None = None

    def flush() -> None:
        nonlocal pending_lines, pending_type, pending_page_start, pending_page_end
        if not pending_lines or pending_page_start is None or pending_page_end is None:
            return
        first_page = pages[0]
        blocks.append(
            SemanticBlock(
                text="\n".join(pending_lines).strip(),
                source_path=first_page.source_path,
                source_name=first_page.source_name,
                filename=first_page.filename,
                page_start=pending_page_start,
                page_end=pending_page_end,
                section_title=current_section,
                document_type=first_page.document_type,
                block_type=pending_type,
            )
        )
        pending_lines = []
        pending_type = "body"
        pending_page_start = None
        pending_page_end = None

    def start_block(block_type: str, line: str, page_number: int) -> None:
        nonlocal pending_lines, pending_type, pending_page_start, pending_page_end
        pending_lines = [line]
        pending_type = block_type
        pending_page_start = page_number
        pending_page_end = page_number

    for page in pages:
        lines = page.text.splitlines()
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            index += 1

            if not line:
                continue

            heading_match = HEADING_RE.match(line)
            if heading_match:
                flush()
                current_section = heading_match.group(1).strip()
                start_block(section_block_type(current_section), line, page.page_number)
                flush()
                continue

            if WARNING_START_RE.match(line):
                flush()
                warning_lines = collect_marker_block(
                    first_line=line,
                    lines=lines,
                    start_index=index,
                    end_marker="[/WARNING]",
                )
                index += len(warning_lines.consumed)
                start_block("warning", "\n".join(warning_lines.text), page.page_number)
                flush()
                continue

            if WARNING_LINE_RE.match(line):
                flush()
                warning_lines = collect_until_boundary(
                    first_line=line,
                    lines=lines,
                    start_index=index,
                )
                index += len(warning_lines.consumed)
                start_block("warning", "\n".join(warning_lines.text), page.page_number)
                flush()
                continue

            if TABLE_START_RE.match(line):
                flush()
                table_lines = collect_marker_block(
                    first_line=line,
                    lines=lines,
                    start_index=index,
                    end_marker="[/TABLE]",
                )
                index += len(table_lines.consumed)
                start_block("table", "\n".join(table_lines.text), page.page_number)
                flush()
                continue

            if STEP_RE.match(line):
                flush()
                step_lines = [line]
                while index < len(lines):
                    next_line = lines[index].strip()
                    if is_boundary(next_line):
                        break
                    if next_line:
                        step_lines.append(next_line)
                    index += 1
                start_block("step", "\n".join(step_lines), page.page_number)
                flush()
                continue

            line_type = "maintenance" if MAINTENANCE_RE.search(current_section) else "body"
            if pending_lines and pending_type != line_type:
                flush()
            if not pending_lines:
                start_block(line_type, line, page.page_number)
            else:
                pending_lines.append(line)
                pending_page_end = page.page_number

    flush()
    return blocks


@dataclass(frozen=True)
class CollectedLines:
    text: list[str]
    consumed: list[str]


def collect_marker_block(
    first_line: str,
    lines: list[str],
    start_index: int,
    end_marker: str,
) -> CollectedLines:
    text = [first_line]
    consumed: list[str] = []
    for line in lines[start_index:]:
        stripped = line.strip()
        consumed.append(line)
        if stripped == end_marker:
            break
        if stripped:
            text.append(stripped)
    return CollectedLines(text=text, consumed=consumed)


def collect_until_boundary(
    first_line: str,
    lines: list[str],
    start_index: int,
) -> CollectedLines:
    text = [first_line]
    consumed: list[str] = []
    for line in lines[start_index:]:
        stripped = line.strip()
        if is_boundary(stripped):
            break
        consumed.append(line)
        if stripped:
            text.append(stripped)
    return CollectedLines(text=text, consumed=consumed)


def is_boundary(line: str) -> bool:
    return bool(
        not line
        or HEADING_RE.match(line)
        or STEP_RE.match(line)
        or WARNING_START_RE.match(line)
        or WARNING_LINE_RE.match(line)
        or TABLE_START_RE.match(line)
        or line == "[/TABLE]"
    )


def section_block_type(section_title: str) -> str:
    return "maintenance" if MAINTENANCE_RE.search(section_title) else "heading"


def pack_blocks(
    blocks: list[SemanticBlock],
    target_tokens: int,
    overlap_tokens: int,
    chunk_offset: int,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    buffer: list[SemanticBlock] = []

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        chunks.append(build_chunk(buffer, chunk_offset + len(chunks)))
        buffer = []

    for block in blocks:
        if block.token_count > MAX_TOKENS:
            flush()
            parts = split_oversized_block(block, target_tokens, overlap_tokens)
            for part in parts:
                chunks.append(build_chunk([part], chunk_offset + len(chunks)))
            continue

        projected = sum(item.token_count for item in buffer) + block.token_count
        boundary_forces_split = block.block_type in {"heading", "maintenance", "warning"}

        if buffer and projected > MAX_TOKENS:
            flush()
        elif buffer and boundary_forces_split and projected >= MIN_TOKENS:
            flush()

        buffer.append(block)

        if sum(item.token_count for item in buffer) >= target_tokens:
            flush()

    flush()
    return chunks


def split_oversized_block(
    block: SemanticBlock,
    target_tokens: int,
    overlap_tokens: int,
) -> list[SemanticBlock]:
    words = block.text.split()
    parts: list[SemanticBlock] = []
    step = max(1, target_tokens - overlap_tokens)

    for start in range(0, len(words), step):
        part_words = words[start : start + target_tokens]
        if not part_words:
            continue
        parts.append(
            SemanticBlock(
                text=" ".join(part_words),
                source_path=block.source_path,
                source_name=block.source_name,
                filename=block.filename,
                page_start=block.page_start,
                page_end=block.page_end,
                section_title=block.section_title,
                document_type=block.document_type,
                block_type=f"{block.block_type}_part",
            )
        )
        if start + target_tokens >= len(words):
            break

    return parts


def build_chunk(blocks: list[SemanticBlock], chunk_index: int) -> DocumentChunk:
    text = "\n\n".join(block.text for block in blocks).strip()
    first = blocks[0]
    last = blocks[-1]
    block_types = sorted({block.block_type for block in blocks})
    section_titles = [block.section_title for block in blocks if block.section_title]
    section_title = section_titles[-1] if section_titles else ""
    digest = hashlib.sha256(
        f"{first.source_name}:{first.page_start}:{last.page_end}:{chunk_index}:{text}".encode(
            "utf-8"
        )
    ).hexdigest()[:16]

    return DocumentChunk(
        id=f"{first.source_name}:{first.page_start}-{last.page_end}:{chunk_index}:{digest}",
        text=text,
        source_path=first.source_path,
        source_name=first.source_name,
        page_number=first.page_start,
        metadata={
            "filename": first.filename,
            "source_name": first.source_name,
            "page_number": first.page_start,
            "page_start": first.page_start,
            "page_end": last.page_end,
            "section_title": section_title,
            "document_type": first.document_type,
            "chunk_index": chunk_index,
            "chunk_type": classify_chunk(block_types),
            "block_types": ", ".join(block_types),
            "token_count": count_tokens(text),
            "contains_warning": any("warning" in block_type for block_type in block_types),
            "contains_steps": any("step" in block_type for block_type in block_types),
            "is_maintenance": any("maintenance" in block_type for block_type in block_types),
        },
    )


def classify_chunk(block_types: list[str]) -> str:
    if any("warning" in block_type for block_type in block_types):
        return "warning"
    if any("step" in block_type for block_type in block_types):
        return "procedure"
    if any("maintenance" in block_type for block_type in block_types):
        return "maintenance"
    return "section"


def count_tokens(text: str) -> int:
    return len(re.findall(r"\S+", text))


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))
