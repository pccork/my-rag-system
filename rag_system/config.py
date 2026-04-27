from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    docs_dir: Path
    metadata_dir: Path
    chroma_dir: Path
    chroma_collection: str
    vector_store_backend: str
    postgres_dsn: str
    postgres_embedding_dimension: int
    postgres_rrf_k: int
    postgres_hnsw_m: int
    postgres_hnsw_ef_construction: int
    postgres_hnsw_ef_search: int
    postgres_candidate_multiplier: int
    postgres_effective_only: bool
    postgres_effective_status: str
    postgres_hnsw_iterative_scan: str
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
        docs_dir=Path(os.getenv("DOCS_DIR", "data/raw")),
        metadata_dir=Path(os.getenv("METADATA_DIR", "data/metadata")),
        chroma_dir=Path(os.getenv("CHROMA_DIR", "data/chroma")),
        chroma_collection=os.getenv("CHROMA_COLLECTION", "local_rag_docs"),
        vector_store_backend=os.getenv("VECTOR_STORE_BACKEND", "chroma"),
        postgres_dsn=os.getenv("POSTGRES_DSN", "postgresql://localhost:5432/rag_system"),
        postgres_embedding_dimension=int(os.getenv("POSTGRES_EMBEDDING_DIMENSION", "768")),
        postgres_rrf_k=int(os.getenv("POSTGRES_RRF_K", "60")),
        postgres_hnsw_m=int(os.getenv("POSTGRES_HNSW_M", "16")),
        postgres_hnsw_ef_construction=int(
            os.getenv("POSTGRES_HNSW_EF_CONSTRUCTION", "100")
        ),
        postgres_hnsw_ef_search=int(os.getenv("POSTGRES_HNSW_EF_SEARCH", "100")),
        postgres_candidate_multiplier=int(os.getenv("POSTGRES_CANDIDATE_MULTIPLIER", "5")),
        postgres_effective_only=parse_bool(os.getenv("POSTGRES_EFFECTIVE_ONLY", "true")),
        postgres_effective_status=os.getenv("POSTGRES_EFFECTIVE_STATUS", "Effective"),
        postgres_hnsw_iterative_scan=os.getenv(
            "POSTGRES_HNSW_ITERATIVE_SCAN", "strict_order"
        ),
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
