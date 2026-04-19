from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    docs_dir: Path
    chroma_dir: Path
    chroma_collection: str
    embedding_backend: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    llm_backend: str
    llm_model: str
    llm_base_url: str
    llm_api_key: str
    llm_temperature: float
    audit_log_enabled: bool
    audit_log_path: Path
    audit_include_text: bool


def get_settings() -> Settings:
    return Settings(
        docs_dir=Path(os.getenv("DOCS_DIR", "docs")),
        chroma_dir=Path(os.getenv("CHROMA_DIR", "data/chroma")),
        chroma_collection=os.getenv("CHROMA_COLLECTION", "local_rag_docs"),
        embedding_backend=os.getenv("EMBEDDING_BACKEND", "sentence-transformers"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "650")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
        llm_backend=os.getenv("LLM_BACKEND", "ollama"),
        llm_model=os.getenv("LLM_MODEL", "llama3.1"),
        llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434"),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        audit_log_enabled=parse_bool(os.getenv("AUDIT_LOG_ENABLED", "true")),
        audit_log_path=Path(os.getenv("AUDIT_LOG_PATH", "data/audit/query_audit.jsonl")),
        audit_include_text=parse_bool(os.getenv("AUDIT_INCLUDE_TEXT", "true")),
    )


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}
