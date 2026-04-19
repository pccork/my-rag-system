from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentPage:
    source_path: str
    source_name: str
    filename: str
    page_number: int
    text: str
    section_title: str | None = None
    document_type: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    text: str
    source_path: str
    source_name: str
    page_number: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None


@dataclass(frozen=True)
class Citation:
    index: int
    filename: str
    page: int | str
    section: str
    chunk_id: str
    score: float | None = None


@dataclass(frozen=True)
class QueryResponse:
    question: str
    answer: str
    citations: list[Citation]
    results: list[SearchResult]
    audit_id: str | None = None
