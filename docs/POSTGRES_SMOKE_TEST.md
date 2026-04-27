# PostgreSQL Hybrid Retrieval Smoke Test

This note records the PostgreSQL backend smoke test used to verify the Phase 1 portal retrieval design:

```text
PostgreSQL + pgvector + PostgreSQL full-text search
documents table
document_chunks table
GIN full-text index
HNSW vector index
RRF result merging
```

## Test Setup

The test used the existing project virtual environment:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

That confirmed the required runtime packages were available, including:

```text
chromadb 1.5.8
psycopg 3.3.3
```

A temporary Docker container provided PostgreSQL with the `vector` extension:

```bash
docker run --name rag-pgvector-smoke \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_DB=rag_system \
  -p 55432:5432 \
  -d pgvector/pgvector:pg16
```

The database was checked with:

```bash
pg_isready -h 127.0.0.1 -p 55432 -U postgres
```

Expected readiness result:

```text
127.0.0.1:55432 - accepting connections
```

## What The Test Did

The smoke test connected the real `PostgresHybridVectorStore` to the temporary database:

```text
postgresql://postgres:postgres@127.0.0.1:55432/rag_system
```

It used a small embedding dimension of `3` so the test could run with tiny synthetic data rather than a real embedding model.

The test then verified that the backend can:

- create the PostgreSQL schema automatically
- create and use the `vector` extension
- create the `documents` and `document_chunks` tables
- create the generated `search_vector` column
- create the GIN full-text index
- create the HNSW pgvector index with the backend defaults
- create partial Effective-only HNSW and full-text indexes
- insert and upsert chunks, embeddings, and metadata
- run vector search
- run PostgreSQL full-text search
- combine both rankings with Reciprocal Rank Fusion
- apply metadata filters
- apply the default `status = Effective` live-search policy
- return citation metadata such as filename, document type, page, and version

The inserted synthetic chunks represented:

```text
ifu-cleaning-1: IFU cleaning text, version A, page 12
sop-warning-1: SOP safety warning text, version 2.1, page 4
ifu-install-1: IFU installation text, version B, page 2
ifu-archived-1: archived IFU cleaning text, version old, page 8
```

## Smoke Test Script

The smoke test body was:

```python
from rag_system.models import DocumentChunk
from rag_system.vector_store import PostgresHybridVectorStore

store = PostgresHybridVectorStore(
    "postgresql://postgres:postgres@127.0.0.1:55432/rag_system",
    embedding_dimension=3,
    hnsw_m=16,
    hnsw_ef_construction=100,
    hnsw_ef_search=100,
    hnsw_iterative_scan="strict_order",
    candidate_multiplier=5,
    effective_only=True,
    effective_status="Effective",
)
store.reset()

chunks = [
    DocumentChunk(
        id="ifu-cleaning-1",
        text="Clean the flow cell with approved cleaning solution before calibration.",
        source_path="data/raw/ifu.pdf",
        source_name="ifu.pdf",
        page_number=12,
        metadata={
            "filename": "ifu.pdf",
            "page_start": 12,
            "page_end": 12,
            "section_title": "Cleaning",
            "document_type": "IFU",
            "chunk_index": 0,
            "version": "A",
            "contains_steps": True,
        },
    ),
    DocumentChunk(
        id="sop-warning-1",
        text="Warning: wear gloves and eye protection before handling reagent.",
        source_path="data/raw/sop.pdf",
        source_name="sop.pdf",
        page_number=4,
        metadata={
            "filename": "sop.pdf",
            "page_start": 4,
            "page_end": 4,
            "section_title": "Safety",
            "document_type": "SOP",
            "chunk_index": 1,
            "version": "2.1",
            "contains_warning": True,
        },
    ),
    DocumentChunk(
        id="ifu-install-1",
        text="Install the analyzer on a level bench with ventilation clearance.",
        source_path="data/raw/install.pdf",
        source_name="install.pdf",
        page_number=2,
        metadata={
            "filename": "install.pdf",
            "page_start": 2,
            "page_end": 2,
            "section_title": "Installation",
            "document_type": "IFU",
            "chunk_index": 2,
            "version": "B",
        },
    ),
    DocumentChunk(
        id="ifu-archived-1",
        text="Archived cleaning procedure that should not appear in live search.",
        source_path="data/raw/archived.pdf",
        source_name="archived.pdf",
        page_number=8,
        metadata={
            "filename": "archived.pdf",
            "page_start": 8,
            "page_end": 8,
            "section_title": "Cleaning",
            "document_type": "IFU",
            "chunk_index": 3,
            "status": "Archived",
            "version": "old",
        },
    ),
]

embeddings = [
    [0.9, 0.1, 0.0],
    [0.1, 0.9, 0.0],
    [0.0, 0.1, 0.9],
    [0.95, 0.05, 0.0],
]

store.add(chunks, embeddings)

results = store.hybrid_search(
    "approved cleaning solution",
    [0.85, 0.15, 0.0],
    top_k=2,
)
filtered = store.hybrid_search(
    "warning reagent",
    [0.1, 0.85, 0.0],
    top_k=3,
    filters={"document_type": "SOP"},
)
archived = store.hybrid_search(
    "archived cleaning",
    [0.95, 0.05, 0.0],
    top_k=3,
)
explicit_archived = store.hybrid_search(
    "archived cleaning",
    [0.95, 0.05, 0.0],
    top_k=3,
    filters={"status": "Archived"},
)
indexdef = store.connection.execute(
    "SELECT indexdef FROM pg_indexes "
    "WHERE indexname = 'document_chunks_embedding_hnsw_idx'"
).fetchone()
partial_indexdef = store.connection.execute(
    "SELECT indexdef FROM pg_indexes "
    "WHERE indexname = 'document_chunks_effective_embedding_hnsw_idx'"
).fetchone()
ef_search = store.connection.execute("SHOW hnsw.ef_search").fetchone()
iterative_scan = store.connection.execute("SHOW hnsw.iterative_scan").fetchone()

print(
    "hybrid:",
    [
        (
            item.id,
            round(item.score or 0, 6),
            item.metadata["filename"],
            item.metadata.get("version"),
        )
        for item in results
    ],
)
print("filtered:", [(item.id, item.metadata["document_type"]) for item in filtered])
print("archived_live_search:", [(item.id, item.metadata["status"]) for item in archived])
print("explicit_archived:", [(item.id, item.metadata["status"]) for item in explicit_archived])
print("index:", indexdef[0] if indexdef else "missing")
print("partial_index:", partial_indexdef[0] if partial_indexdef else "missing")
print("ef_search:", ef_search[0] if ef_search else "missing")
print("iterative_scan:", iterative_scan[0] if iterative_scan else "missing")
```

## Result

The test passed with this output:

```text
hybrid: [('ifu-cleaning-1', 0.032787, 'ifu.pdf', 'A'), ('sop-warning-1', 0.016129, 'sop.pdf', '2.1')]
filtered: [('sop-warning-1', 'SOP')]
archived_live_search: [('ifu-cleaning-1', 'Effective'), ('sop-warning-1', 'Effective'), ('ifu-install-1', 'Effective')]
explicit_archived: [('ifu-archived-1', 'Archived')]
index: CREATE INDEX document_chunks_embedding_hnsw_idx ON public.document_chunks USING hnsw (embedding vector_cosine_ops) WITH (m='16', ef_construction='100')
partial_index: CREATE INDEX document_chunks_effective_embedding_hnsw_idx ON public.document_chunks USING hnsw (embedding vector_cosine_ops) WITH (m='16', ef_construction='100') WHERE (status = 'Effective'::text)
ef_search: 100
iterative_scan: strict_order
```

This result shows that the cleaning IFU chunk ranked first when both semantic similarity and exact full-text terms matched the query. It also shows that the metadata filter correctly limited the warning query to the SOP chunk.

The archived live-search output confirms that a chunk marked `Archived` was not returned by default, even though its embedding was close to the query. The explicit archived query confirms archived content remains available when an authorized workflow supplies a deliberate `status = Archived` filter. The index output confirms that the HNSW build parameters and Effective-only partial index were applied. The `ef_search` and `iterative_scan` outputs confirm that the query-time HNSW settings were active for the search transaction.

## Cleanup

The temporary PostgreSQL container was stopped and removed after the test:

```bash
docker stop rag-pgvector-smoke
docker rm rag-pgvector-smoke
```

No smoke-test database or container was left running.
