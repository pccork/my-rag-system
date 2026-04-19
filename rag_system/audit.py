from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from rag_system.config import Settings
from rag_system.models import Citation, SearchResult
from rag_system.vector_store import MetadataFilter


AUDIT_SCHEMA_VERSION = "query-audit-v1"


def new_audit_id() -> str:
    return str(uuid4())


def write_query_audit_event(
    *,
    settings: Settings,
    audit_id: str,
    user_id: str,
    session_id: str | None,
    question: str,
    top_k: int | None,
    filters: MetadataFilter | None,
    results: list[SearchResult],
    citations: list[Citation],
    answer: str | None,
    status: str,
    latency_ms: int,
    error: str | None = None,
    request_metadata: dict[str, Any] | None = None,
) -> None:
    if not settings.audit_log_enabled:
        return

    event = build_query_audit_event(
        settings=settings,
        audit_id=audit_id,
        user_id=user_id,
        session_id=session_id,
        question=question,
        top_k=top_k,
        filters=filters,
        results=results,
        citations=citations,
        answer=answer,
        status=status,
        latency_ms=latency_ms,
        error=error,
        request_metadata=request_metadata,
    )
    append_hash_chained_jsonl(settings.audit_log_path, event)


def build_query_audit_event(
    *,
    settings: Settings,
    audit_id: str,
    user_id: str,
    session_id: str | None,
    question: str,
    top_k: int | None,
    filters: MetadataFilter | None,
    results: list[SearchResult],
    citations: list[Citation],
    answer: str | None,
    status: str,
    latency_ms: int,
    error: str | None = None,
    request_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    include_text = settings.audit_include_text
    event: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "event_type": "rag.query",
        "audit_id": audit_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "actor": {
            "user_id": user_id,
            "session_id": session_id or "",
        },
        "request": {
            "question": question if include_text else "",
            "question_sha256": sha256_text(question),
            "top_k": top_k or settings.retrieval_top_k,
            "filters": filters or {},
            "metadata": request_metadata or {},
        },
        "pipeline": {
            "embedding_backend": settings.embedding_backend,
            "embedding_model": settings.embedding_model,
            "vector_store": "chromadb",
            "chroma_collection": settings.chroma_collection,
            "llm_backend": settings.llm_backend,
            "llm_model": settings.llm_model,
        },
        "retrieval": {
            "chunk_count": len(results),
            "chunks": [serialize_result(result, include_text=include_text) for result in results],
        },
        "response": {
            "answer": answer if include_text and answer is not None else "",
            "answer_sha256": sha256_text(answer or ""),
            "citations": [serialize_dataclass(citation) for citation in citations],
        },
        "outcome": {
            "status": status,
            "error": error or "",
            "latency_ms": latency_ms,
        },
    }
    return event


def append_hash_chained_jsonl(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_hash = read_last_record_hash(path)
    event["previous_record_hash"] = previous_hash
    event["record_hash"] = compute_record_hash(event)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, sort_keys=True, separators=(",", ":"), default=str) + "\n")


def read_last_record_hash(path: Path) -> str:
    if not path.exists():
        return ""

    last_line = ""
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                last_line = line.strip()

    if not last_line:
        return ""
    try:
        return str(json.loads(last_line).get("record_hash", ""))
    except json.JSONDecodeError:
        return ""


def compute_record_hash(event: dict[str, Any]) -> str:
    hashable = {key: value for key, value in event.items() if key != "record_hash"}
    payload = json.dumps(hashable, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_text(payload)


def verify_hash_chain(path: Path) -> tuple[bool, str]:
    previous_hash = ""
    if not path.exists():
        return True, "Audit log does not exist yet."

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                return False, f"Line {line_number} is not valid JSON: {exc}"

            if event.get("previous_record_hash", "") != previous_hash:
                return False, f"Line {line_number} has an invalid previous_record_hash."

            record_hash = str(event.get("record_hash", ""))
            if compute_record_hash(event) != record_hash:
                return False, f"Line {line_number} has an invalid record_hash."

            previous_hash = record_hash

    return True, "Audit hash chain is valid."


def serialize_result(result: SearchResult, *, include_text: bool) -> dict[str, Any]:
    return {
        "id": result.id,
        "text": result.text if include_text else "",
        "text_sha256": sha256_text(result.text),
        "metadata": result.metadata,
        "score": result.score,
    }


def serialize_dataclass(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    return dict(value)


def sha256_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()
