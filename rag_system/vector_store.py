from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from rag_system.models import DocumentChunk, SearchResult

if TYPE_CHECKING:
    from psycopg import Connection
    from psycopg.sql import Composed

    from rag_system.config import Settings


MetadataValue: TypeAlias = str | int | float | bool | list[str | int | float | bool]
MetadataFilter: TypeAlias = dict[str, MetadataValue]


class VectorStore(ABC):
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def add(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        raise NotImplementedError


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        import chromadb

        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        existing = self.collection.get(include=[])
        ids = existing.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)

    def add(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[chunk_metadata(chunk) for chunk in chunks],
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        response = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters,
            include=["documents", "metadatas", "distances"],
        )
        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: list[SearchResult] = []
        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=False
        ):
            score = 1 - distance if distance is not None else None
            results.append(
                SearchResult(
                    id=chunk_id,
                    text=document,
                    metadata=metadata or {},
                    score=score,
                )
            )
        return results


class PostgresHybridVectorStore(VectorStore):
    """PostgreSQL store using pgvector, full-text search, and RRF ranking."""

    def __init__(
        self,
        dsn: str,
        embedding_dimension: int = 768,
        rrf_k: int = 60,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 100,
        hnsw_ef_search: int = 100,
        candidate_multiplier: int = 5,
        effective_only: bool = True,
        effective_status: str = "Effective",
        hnsw_iterative_scan: str = "strict_order",
    ) -> None:
        """Connect to PostgreSQL and ensure the hybrid retrieval schema exists."""
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL vector storage requires psycopg. Install requirements first."
            ) from exc

        self.dsn = dsn
        self.embedding_dimension = embedding_dimension
        self.rrf_k = rrf_k
        self.hnsw_m = max(int(hnsw_m), 1)
        self.hnsw_ef_construction = max(int(hnsw_ef_construction), 1)
        self.hnsw_ef_search = max(int(hnsw_ef_search), 1)
        self.candidate_multiplier = max(int(candidate_multiplier), 1)
        self.effective_only = effective_only
        self.effective_status = effective_status.strip() or "Effective"
        self.hnsw_iterative_scan = hnsw_iterative_scan.strip()
        self.connection: Connection[Any] = psycopg.connect(dsn)
        self.connection.execute("SET timezone TO 'UTC'")
        self.ensure_schema()

    def reset(self) -> None:
        """Delete indexed documents and chunks from PostgreSQL."""
        with self.connection.transaction():
            self.connection.execute("TRUNCATE document_chunks, documents RESTART IDENTITY")

    def add(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert chunks, metadata, and embeddings into PostgreSQL."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        from psycopg.types.json import Jsonb

        with self.connection.transaction():
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                document_id = self.upsert_document(chunk)
                metadata = postgres_chunk_metadata(chunk)
                self.connection.execute(
                    """
                    INSERT INTO document_chunks (
                        id,
                        document_id,
                        chunk_index,
                        text,
                        embedding,
                        metadata,
                        filename,
                        page_start,
                        page_end,
                        section_title,
                        document_type,
                        status,
                        version,
                        source_name,
                        source_path
                    )
                    VALUES (
                        %(id)s,
                        %(document_id)s,
                        %(chunk_index)s,
                        %(text)s,
                        %(embedding)s::vector,
                        %(metadata)s,
                        %(filename)s,
                        %(page_start)s,
                        %(page_end)s,
                        %(section_title)s,
                        %(document_type)s,
                        %(status)s,
                        %(version)s,
                        %(source_name)s,
                        %(source_path)s
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        chunk_index = EXCLUDED.chunk_index,
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        filename = EXCLUDED.filename,
                        page_start = EXCLUDED.page_start,
                        page_end = EXCLUDED.page_end,
                        section_title = EXCLUDED.section_title,
                        document_type = EXCLUDED.document_type,
                        status = EXCLUDED.status,
                        version = EXCLUDED.version,
                        source_name = EXCLUDED.source_name,
                        source_path = EXCLUDED.source_path
                    """,
                    {
                        "id": chunk.id,
                        "document_id": document_id,
                        "chunk_index": int(metadata.get("chunk_index", 0)),
                        "text": chunk.text,
                        "embedding": vector_literal(embedding),
                        "metadata": Jsonb(metadata),
                        "filename": str(metadata.get("filename") or chunk.source_name),
                        "page_start": int(metadata.get("page_start") or chunk.page_number),
                        "page_end": int(metadata.get("page_end") or chunk.page_number),
                        "section_title": str(metadata.get("section_title") or ""),
                        "document_type": str(metadata.get("document_type") or "unknown"),
                        "status": metadata_status(metadata),
                        "version": metadata_version(metadata),
                        "source_name": chunk.source_name,
                        "source_path": chunk.source_path,
                    },
                )

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        """Run vector search, falling back to hybrid ranking without text terms."""
        return self.hybrid_search(
            query_text="",
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: int,
        filters: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        """Combine pgvector and PostgreSQL full-text results with RRF scoring."""
        from psycopg import sql

        effective_filters = self.effective_filters(filters)
        filter_sql, text_filter_sql, filter_params = build_postgres_filter_sql(
            effective_filters
        )
        params: dict[str, Any] = {
            "query_embedding": vector_literal(query_embedding),
            "query_text": query_text,
            "top_k": top_k,
            "candidate_limit": max(top_k * self.candidate_multiplier, top_k),
            "rrf_k": self.rrf_k,
        }
        params.update(filter_params)
        statement = sql.SQL(
            """
            WITH vector_results AS (
                SELECT
                    id,
                    row_number() OVER (ORDER BY embedding <=> %(query_embedding)s::vector) AS rank
                FROM document_chunks
                {filter_sql}
                ORDER BY embedding <=> %(query_embedding)s::vector
                LIMIT %(candidate_limit)s
            ),
            text_results AS (
                SELECT
                    id,
                    row_number() OVER (
                        ORDER BY ts_rank_cd(search_vector, websearch_to_tsquery('english', %(query_text)s)) DESC
                    ) AS rank
                FROM document_chunks
                WHERE search_vector @@ websearch_to_tsquery('english', %(query_text)s)
                {text_filter_sql}
                ORDER BY ts_rank_cd(search_vector, websearch_to_tsquery('english', %(query_text)s)) DESC
                LIMIT %(candidate_limit)s
            ),
            combined AS (
                SELECT
                    id,
                    sum(score) AS rrf_score
                FROM (
                    SELECT id, 1.0 / (%(rrf_k)s + rank) AS score FROM vector_results
                    UNION ALL
                    SELECT id, 1.0 / (%(rrf_k)s + rank) AS score FROM text_results
                ) ranked
                GROUP BY id
            )
            SELECT
                chunk.id,
                chunk.text,
                chunk.metadata,
                chunk.filename,
                chunk.page_start,
                chunk.page_end,
                chunk.section_title,
                chunk.document_type,
                chunk.status,
                chunk.version,
                chunk.source_name,
                chunk.source_path,
                combined.rrf_score
            FROM combined
            JOIN document_chunks chunk ON chunk.id = combined.id
            ORDER BY combined.rrf_score DESC, chunk.id
            LIMIT %(top_k)s
            """
        ).format(
            filter_sql=filter_sql,
            text_filter_sql=text_filter_sql,
        )
        with self.connection.transaction():
            self.connection.execute(
                "SELECT set_config('hnsw.ef_search', %s, true)",
                [str(self.hnsw_ef_search)],
            )
            if self.hnsw_iterative_scan:
                self.connection.execute(
                    "SELECT set_config('hnsw.iterative_scan', %s, true)",
                    [self.hnsw_iterative_scan],
                )
            rows = self.connection.execute(statement, params).fetchall()
        return [postgres_row_to_result(row) for row in rows]

    def ensure_schema(self) -> None:
        """Create tables and indexes required by the PostgreSQL backend."""
        with self.connection.transaction():
            self.connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id bigserial PRIMARY KEY,
                    source_path text NOT NULL UNIQUE,
                    filename text NOT NULL,
                    source_name text NOT NULL,
                    document_type text NOT NULL DEFAULT 'unknown',
                    status text NOT NULL DEFAULT 'Effective',
                    version text NOT NULL DEFAULT 'unknown',
                    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    updated_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            self.connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id text PRIMARY KEY,
                    document_id bigint NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index integer NOT NULL,
                    text text NOT NULL,
                    embedding vector({self.embedding_dimension}) NOT NULL,
                    metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                    filename text NOT NULL,
                    page_start integer NOT NULL,
                    page_end integer NOT NULL,
                    section_title text NOT NULL DEFAULT '',
                    document_type text NOT NULL DEFAULT 'unknown',
                    status text NOT NULL DEFAULT 'Effective',
                    version text NOT NULL DEFAULT 'unknown',
                    source_name text NOT NULL,
                    source_path text NOT NULL,
                    search_vector tsvector GENERATED ALWAYS AS (
                        setweight(to_tsvector('english', coalesce(section_title, '')), 'A') ||
                        setweight(to_tsvector('english', coalesce(text, '')), 'B') ||
                        setweight(to_tsvector('english', coalesce(filename, '')), 'C')
                    ) STORED
                )
                """
            )
            self.migrate_schema()
            self.connection.execute(
                """
                CREATE INDEX IF NOT EXISTS document_chunks_search_vector_idx
                ON document_chunks USING gin (search_vector)
                """
            )
            self.connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx
                ON document_chunks USING hnsw (embedding vector_cosine_ops)
                WITH (m = {self.hnsw_m}, ef_construction = {self.hnsw_ef_construction})
                """
            )
            self.connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS document_chunks_effective_embedding_hnsw_idx
                ON document_chunks USING hnsw (embedding vector_cosine_ops)
                WITH (m = {self.hnsw_m}, ef_construction = {self.hnsw_ef_construction})
                WHERE status = '{sql_literal(self.effective_status)}'
                """
            )
            self.connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS document_chunks_effective_search_vector_idx
                ON document_chunks USING gin (search_vector)
                WHERE status = '{sql_literal(self.effective_status)}'
                """
            )
            self.connection.execute(
                """
                CREATE INDEX IF NOT EXISTS document_chunks_metadata_idx
                ON document_chunks USING gin (metadata)
                """
            )
            self.connection.execute(
                """
                CREATE INDEX IF NOT EXISTS document_chunks_source_page_idx
                ON document_chunks (filename, page_start, page_end)
                """
            )
            self.connection.execute(
                """
                CREATE INDEX IF NOT EXISTS document_chunks_status_version_idx
                ON document_chunks (status, filename, version)
                """
            )

    def upsert_document(self, chunk: DocumentChunk) -> int:
        """Insert or update a source document row and return its database id."""
        from psycopg.types.json import Jsonb

        metadata = postgres_chunk_metadata(chunk)
        row = self.connection.execute(
            """
            INSERT INTO documents (
                source_path,
                filename,
                source_name,
                document_type,
                status,
                version,
                metadata,
                updated_at
            )
            VALUES (
                %(source_path)s,
                %(filename)s,
                %(source_name)s,
                %(document_type)s,
                %(status)s,
                %(version)s,
                %(metadata)s,
                now()
            )
            ON CONFLICT (source_path) DO UPDATE SET
                filename = EXCLUDED.filename,
                source_name = EXCLUDED.source_name,
                document_type = EXCLUDED.document_type,
                status = EXCLUDED.status,
                version = EXCLUDED.version,
                metadata = documents.metadata || EXCLUDED.metadata,
                updated_at = now()
            RETURNING id
            """,
            {
                "source_path": chunk.source_path,
                "filename": str(metadata.get("filename") or chunk.source_name),
                "source_name": chunk.source_name,
                "document_type": str(metadata.get("document_type") or "unknown"),
                "status": metadata_status(metadata),
                "version": metadata_version(metadata),
                "metadata": Jsonb(
                    {
                        key: value
                        for key, value in metadata.items()
                        if key
                        in {
                            "filename",
                            "source_name",
                            "document_type",
                            "manufacturer",
                            "product",
                            "product_family",
                            "document_code",
                            "part_number",
                            "version",
                            "revision",
                            "language",
                            "status",
                            "document_status",
                            "lifecycle_status",
                        }
                    }
                ),
            },
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Failed to upsert document for {chunk.source_name}")
        return int(row[0])

    def migrate_schema(self) -> None:
        """Add relational status and version columns to existing PostgreSQL tables."""
        self.connection.execute(
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'Effective'"
        )
        self.connection.execute(
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS version text NOT NULL DEFAULT 'unknown'"
        )
        self.connection.execute(
            "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'Effective'"
        )
        self.connection.execute(
            "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS version text NOT NULL DEFAULT 'unknown'"
        )

    def effective_filters(self, filters: MetadataFilter | None) -> MetadataFilter | None:
        """Apply the configured effective-document policy to retrieval filters."""
        merged: MetadataFilter = dict(filters or {})
        if self.effective_only and "status" not in merged and "document_status" not in merged:
            merged["status"] = self.effective_status
        return merged or None


def get_vector_store(settings: Settings) -> VectorStore:
    """Build the configured vector store backend."""
    backend = settings.vector_store_backend.strip().lower()
    if backend in {"chroma", "chromadb"}:
        return ChromaVectorStore(settings.chroma_dir, settings.chroma_collection)
    if backend in {"postgres", "postgresql", "pgvector"}:
        return PostgresHybridVectorStore(
            settings.postgres_dsn,
            embedding_dimension=settings.postgres_embedding_dimension,
            rrf_k=settings.postgres_rrf_k,
            hnsw_m=settings.postgres_hnsw_m,
            hnsw_ef_construction=settings.postgres_hnsw_ef_construction,
            hnsw_ef_search=settings.postgres_hnsw_ef_search,
            candidate_multiplier=settings.postgres_candidate_multiplier,
            effective_only=settings.postgres_effective_only,
            effective_status=settings.postgres_effective_status,
            hnsw_iterative_scan=settings.postgres_hnsw_iterative_scan,
        )
    raise ValueError(f"Unsupported vector store backend: {settings.vector_store_backend}")


def vector_literal(embedding: list[float]) -> str:
    """Format an embedding as a pgvector literal."""
    return "[" + ",".join(str(float(value)) for value in embedding) + "]"


def build_postgres_filter_sql(
    filters: MetadataFilter | None,
) -> tuple[Composed, Composed, dict[str, Any]]:
    """Build safe SQL fragments for exact-match metadata filters."""
    from psycopg import sql
    from psycopg.types.json import Jsonb

    if not filters:
        return sql.SQL(""), sql.SQL(""), {}

    clauses: list[Composed] = []
    params: dict[str, Any] = {}
    column_filters = {
        "filename",
        "page_start",
        "page_end",
        "section_title",
        "document_type",
        "status",
        "version",
        "source_name",
        "source_path",
    }
    for index, (key, value) in enumerate(filters.items()):
        param_name = f"filter_{index}"
        column_key = "status" if key == "document_status" else key
        if key == "page_number":
            clauses.append(sql.SQL("page_start <= {param} AND page_end >= {param}").format(
                param=sql.Placeholder(param_name)
            ))
            params[param_name] = value
        elif column_key in column_filters:
            clauses.append(
                sql.SQL("{field} = {param}").format(
                    field=sql.Identifier(column_key),
                    param=sql.Placeholder(param_name),
                )
            )
            params[param_name] = value
        elif column_key in {"related_labs", "analysis_types"} and isinstance(value, str):
            clauses.append(
                sql.SQL("metadata @> {param}").format(param=sql.Placeholder(param_name))
            )
            params[param_name] = Jsonb({column_key: [value]})
        else:
            clauses.append(
                sql.SQL("metadata @> {param}").format(param=sql.Placeholder(param_name))
            )
            params[param_name] = Jsonb({key: value})

    joined = sql.SQL(" AND ").join(clauses)
    return sql.SQL("WHERE ") + joined, sql.SQL("AND ") + joined, params


def postgres_row_to_result(row: tuple[Any, ...]) -> SearchResult:
    """Convert a PostgreSQL result row into the shared search result model."""
    metadata = dict(row[2] or {})
    metadata.update(
        {
            "filename": row[3],
            "page_start": row[4],
            "page_end": row[5],
            "page_number": row[4],
            "section_title": row[6],
            "document_type": row[7],
            "status": row[8],
            "document_status": row[8],
            "version": row[9],
            "source_name": row[10],
            "source_path": row[11],
        }
    )
    return SearchResult(
        id=str(row[0]),
        text=str(row[1]),
        metadata=metadata,
        score=float(row[12]) if row[12] is not None else None,
    )


def metadata_status(metadata: dict[str, MetadataValue]) -> str:
    """Read a document lifecycle status from metadata."""
    return metadata_text(
        metadata,
        keys=("status", "document_status", "lifecycle_status", "effective_status"),
        default="Effective",
    )


def metadata_version(metadata: dict[str, MetadataValue]) -> str:
    """Read a document version or revision from metadata."""
    return metadata_text(
        metadata,
        keys=("version", "document_version", "revision"),
        default="unknown",
    )


def metadata_text(
    metadata: dict[str, MetadataValue],
    *,
    keys: tuple[str, ...],
    default: str,
) -> str:
    """Return the first non-empty metadata value for a list of keys."""
    for key in keys:
        value = metadata.get(key)
        if value not in {None, ""}:
            return str(value)
    return default


def sql_literal(value: str) -> str:
    """Escape a string for use in a SQL literal inside DDL."""
    return value.replace("'", "''")


def chunk_metadata(chunk: DocumentChunk) -> dict[str, MetadataValue]:
    metadata = dict(chunk.metadata)
    metadata.update(
        {
            "source_path": chunk.source_path,
            "source_name": chunk.source_name,
            "page_number": chunk.page_number,
        }
    )
    return {
        key: value
        for key, value in metadata.items()
        if is_chroma_metadata_value(value)
    }


def postgres_chunk_metadata(chunk: DocumentChunk) -> dict[str, MetadataValue]:
    """Build JSONB metadata for PostgreSQL, preserving scalar string lists."""
    metadata = dict(chunk.metadata)
    metadata.update(
        {
            "source_path": chunk.source_path,
            "source_name": chunk.source_name,
            "page_number": chunk.page_number,
        }
    )
    return {
        key: value
        for key, value in metadata.items()
        if is_postgres_metadata_value(value)
    }


def is_chroma_metadata_value(value: object) -> bool:
    return isinstance(value, str | int | float | bool)


def is_postgres_metadata_value(value: object) -> bool:
    if is_chroma_metadata_value(value):
        return True
    return isinstance(value, list) and all(
        isinstance(item, str | int | float | bool) for item in value
    )
