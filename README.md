# Local Clinical RAG System

A local Retrieval Augmented Generation project for SOP and IFU PDFs, with structured ingestion, intelligent chunking, hybrid PostgreSQL retrieval, ChromaDB local retrieval, LLM answering, citations, Streamlit UI, retrieval evaluation, and query audit logging.

This repository is intended to act as the SOP/IFU RAG knowledge layer for the Lab Competence Portal:

```text
https://github.com/pccork/labcompetence-portal
```

The `labcompetence-portal` application should own user-facing workflows, authentication, authorization, role-based access control, security controls, trainee/staff context, and assignment delivery. This RAG system should remain focused on auditable document ingestion, chunking, retrieval, citation-grounded Q&A, and future question-generation support.

## Capabilities

- Ingests PDFs from `./data/raw`
- Requires document metadata from `./data/metadata` before ingest
- Extracts structured text while preserving headings, warnings, cautions, table-like rows, and page metadata
- Infers document type as `SOP`, `IFU`, or `unknown`
- Chunks SOP/IFU content by headings, numbered steps, warnings, cautions, and maintenance sections
- Uses pluggable embeddings with `all-mpnet-base-v2` by default
- Stores vectors locally in persistent ChromaDB by default
- Supports PostgreSQL with pgvector, PostgreSQL full-text search, GIN indexes, HNSW vector indexes, and Reciprocal Rank Fusion for hybrid retrieval
- Supports exact-match metadata filtering during retrieval
- Generates answers through a pluggable LLM backend
- Supports local Ollama or a remote vLLM server on an RTX 3090
- Returns source citations with filename, page, section, chunk id, and score
- Provides CLI and Streamlit interfaces
- Includes retrieval-only evaluation with manual scoring
- Writes hash-chained query audit logs for user/session traceability

## Portal Integration

The planned production split is:

```text
labcompetence-portal
  -> RBAC, authentication, security, user roles, lab scopes, assignments, UI
  -> calls this RAG service with user/lab context

my-rag-system
  -> SOP/IFU ingestion, metadata validation, hybrid retrieval, citations, audit logs
  -> returns grounded Q&A context and source-backed generated content
```

The portal can pass lab and role context as retrieval filters. For example, a Biochemistry or Immunology staff member can query documents tagged with those `related_labs`, while a POC trainee should be limited to documents tagged with `POC`.

Future AWS Bedrock integration can use this same retrieval layer to provide grounded context for:

- Q&A over Effective SOP/IFU versions
- draft MCQ generation from retrieved source chunks
- trainee assignment creation linked to instrument, lab, document version, and citations
- answer explanations that cite the exact SOP/IFU source, page, section, and version

Security-critical decisions should remain in `labcompetence-portal`; this service should receive already-authorized query context and enforce document metadata filters as a second layer of protection.

## Project Layout

```text
.
|-- app/
|   `-- streamlit_app.py
|-- data/
|   |-- audit/
|   |   `-- .gitkeep
|   |-- chroma/
|   |   `-- .gitkeep
|   |-- metadata/
|   |   |-- .gitkeep
|   |   `-- B89027AA.json
|   `-- raw/
|       `-- .gitkeep
|-- docs/
|   |-- .gitkeep
|   |-- POSTGRES_SMOKE_TEST.md
|   `-- VLLM_LLAMA_3090.md
|-- rag_system/
|   |-- audit.py
|   |-- chunking.py
|   |-- config.py
|   |-- embeddings.py
|   |-- evaluation.py
|   |-- ingest.py
|   |-- llm.py
|   |-- loaders.py
|   |-- models.py
|   |-- query.py
|   `-- vector_store.py
|-- scripts/
|   |-- evaluate.py
|   |-- ingest.py
|   |-- query.py
|   `-- verify_audit.py
|-- eval_questions.json
|-- .env.example
|-- .gitignore
`-- requirements.txt
```

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put PDF files in:

```text
data/raw/
```

Build the local index:

```bash
python scripts/ingest.py
```

Ask a question from the CLI:

```bash
python scripts/query.py "What are the setup steps?" --user-id alice
```

Run the Streamlit UI:

```bash
streamlit run app/streamlit_app.py
```

## Configuration

Configuration is loaded from `.env`.

```text
DOCS_DIR=data/raw
METADATA_DIR=data/metadata
CHROMA_DIR=data/chroma
CHROMA_COLLECTION=local_rag_docs
VECTOR_STORE_BACKEND=chroma
POSTGRES_DSN=postgresql://localhost:5432/rag_system
POSTGRES_EMBEDDING_DIMENSION=768
POSTGRES_RRF_K=60
POSTGRES_HNSW_M=16
POSTGRES_HNSW_EF_CONSTRUCTION=100
POSTGRES_HNSW_EF_SEARCH=100
POSTGRES_HNSW_ITERATIVE_SCAN=strict_order
POSTGRES_CANDIDATE_MULTIPLIER=5
POSTGRES_EFFECTIVE_ONLY=true
POSTGRES_EFFECTIVE_STATUS=Effective

EMBEDDING_BACKEND=sentence-transformers
EMBEDDING_MODEL=all-mpnet-base-v2

CHUNK_SIZE=650
CHUNK_OVERLAP=50
RETRIEVAL_TOP_K=5

LLM_BACKEND=ollama
LLM_MODEL=llama3.1
LLM_BASE_URL=http://localhost:11434
LLM_API_KEY=
LLM_TEMPERATURE=0.1

AUDIT_LOG_ENABLED=true
AUDIT_LOG_PATH=data/audit/query_audit.jsonl
AUDIT_INCLUDE_TEXT=true
```

## PDF Ingestion

PDFs are read from `DOCS_DIR`, which defaults to `data/raw`.

The loader extracts page-level text and emits structured text for chunking. It preserves:

- Headings
- Warning/caution blocks
- Table-like rows
- Page boundaries

Each page carries metadata:

```text
filename
page_number
section_title
document_type
total_pages
```

`document_type` is inferred from the filename when it contains `SOP` or `IFU`; otherwise it is set to `unknown`.

Document-level metadata must be added with a JSON file in `METADATA_DIR`.
Use either the PDF stem or full PDF filename, for example:

```text
data/raw/B89027AA-Remisol-Advance.pdf
data/metadata/B89027AA-Remisol-Advance.json
```

Example:

```json
{
  "manufacturer": "Beckman Coulter",
  "publisher": "Normand Info",
  "product": "Remisol Advance",
  "product_family": "Remisol",
  "document_code": "UG-ADV-SK-18",
  "part_number": "B89027AA",
  "status": "Effective",
  "version": "B89027AA",
  "effective_date": "2014-05",
  "related_labs": [
    "Biochemistry",
    "Immunology"
  ],
  "analysis_types": [
    "Middleware",
    "Result management"
  ],
  "document_type": "user_guide",
  "language": "sk",
  "created_date": "2014-05"
}
```

### Ingest Validation

`python scripts/ingest.py` validates every PDF before extraction, chunking, embedding, or vector-store writes. Ingest refuses PDFs without a matching metadata JSON file and refuses metadata without explicit:

```text
status
version
effective_date
related_labs
analysis_types
```

`related_labs` and `analysis_types` must be non-empty lists of strings. Use `related_labs` for access and retrieval scopes such as `Biochemistry`, `Immunology`, or `POC`. Current core analyser documents are tagged for both `Biochemistry` and `Immunology`. A staff member who works across laboratories can query multiple lab scopes, while a POC trainee should be limited to documents with `related_labs` containing `POC`.

Ingest warns if two documents share the same `document_code` while both are marked `Effective`. That warning allows historical versions to remain indexed but draws attention to possible source-of-truth conflicts before the live portal is used. Warnings are printed by the CLI; validation errors stop ingest.

You can run the same validation directly:

```bash
python - <<'PY'
from pathlib import Path
from rag_system.ingest_validation import validate_ingest_metadata

report = validate_ingest_metadata(Path("data/raw"), Path("data/metadata"))
print(f"documents: {report.document_count}")
print(f"warnings: {report.warnings}")
PY
```

## Chunking

SOP and IFU content is split on semantic boundaries:

- Headings
- Numbered procedural steps
- Warnings and cautions
- Maintenance, cleaning, calibration, service, inspection, and troubleshooting sections

Chunks target roughly `300-800` tokens. Overlap is only used when an oversized semantic block must be split.

Chunk metadata includes:

```text
filename
source_name
page_number
page_start
page_end
section_title
document_type
chunk_index
chunk_type
block_types
token_count
contains_warning
contains_steps
is_maintenance
```

## Embeddings And Vector Storage

The embedding interface is pluggable. The default provider is Sentence Transformers:

```text
EMBEDDING_BACKEND=sentence-transformers
EMBEDDING_MODEL=all-mpnet-base-v2
```

Vectors are stored in persistent ChromaDB by default:

```text
data/chroma
```

For portal-style deployment, use PostgreSQL with pgvector and full-text search:

```text
VECTOR_STORE_BACKEND=postgres
POSTGRES_DSN=postgresql://USER:PASSWORD@HOST:5432/rag_system
POSTGRES_EMBEDDING_DIMENSION=768
POSTGRES_RRF_K=60
POSTGRES_HNSW_M=16
POSTGRES_HNSW_EF_CONSTRUCTION=100
POSTGRES_HNSW_EF_SEARCH=100
POSTGRES_HNSW_ITERATIVE_SCAN=strict_order
POSTGRES_CANDIDATE_MULTIPLIER=5
POSTGRES_EFFECTIVE_ONLY=true
POSTGRES_EFFECTIVE_STATUS=Effective
```

The PostgreSQL backend creates this Phase 1 schema automatically:

```text
documents
document_chunks
document_chunks.embedding vector(POSTGRES_EMBEDDING_DIMENSION)
documents.status and document_chunks.status relational columns
documents.version and document_chunks.version relational columns
document_chunks.search_vector generated tsvector
GIN index on search_vector
GIN index on metadata
HNSW index on embedding using cosine distance, m, and ef_construction
partial HNSW and GIN indexes for status = POSTGRES_EFFECTIVE_STATUS
```

PostgreSQL retrieval runs both searches and combines them with Reciprocal Rank Fusion:

```text
question
  -> embedding
  -> pgvector HNSW semantic search
  -> PostgreSQL full-text search
  -> RRF score merge
  -> top-k cited chunks
```

PostgreSQL HNSW tuning defaults are chosen as a conservative Phase 1 starting point for 768- or 1536-dimension embeddings:

```text
POSTGRES_HNSW_M=16
POSTGRES_HNSW_EF_CONSTRUCTION=100
POSTGRES_HNSW_EF_SEARCH=100
POSTGRES_HNSW_ITERATIVE_SCAN=strict_order
POSTGRES_CANDIDATE_MULTIPLIER=5
```

`POSTGRES_HNSW_M` and `POSTGRES_HNSW_EF_CONSTRUCTION` are index-build settings. Changing them requires rebuilding `document_chunks_embedding_hnsw_idx` and `document_chunks_effective_embedding_hnsw_idx`. `POSTGRES_HNSW_EF_SEARCH` is applied at query time to trade speed for recall. `POSTGRES_HNSW_ITERATIVE_SCAN=strict_order` asks pgvector 0.8.0+ to scan deeper when filters are restrictive. `POSTGRES_CANDIDATE_MULTIPLIER` controls how many vector and full-text candidates are gathered before RRF; for example, `top_k=5` with multiplier `5` gathers up to `25` candidates from each branch.

For clinical portal retrieval, PostgreSQL searches only Effective document versions by default:

```text
POSTGRES_EFFECTIVE_ONLY=true
POSTGRES_EFFECTIVE_STATUS=Effective
```

The backend stores `status` and `version` as relational columns on both `documents` and `document_chunks`, not only inside JSON metadata. At ingest time, status is read from `status`, `document_status`, `lifecycle_status`, or `effective_status` metadata. Version is read from `version`, `document_version`, or `revision`. If no status is supplied, the document is treated as `Effective`; set explicit metadata for archived or under-review files before ingesting them. Live PostgreSQL retrieval automatically adds `status = POSTGRES_EFFECTIVE_STATUS` unless the caller supplies an explicit `status` or `document_status` filter.

The PostgreSQL backend smoke-test setup and result are recorded in:

```text
docs/POSTGRES_SMOKE_TEST.md
```

Retrieval supports exact-match metadata filters, including:

```text
document_type
manufacturer
product
product_family
document_code
part_number
filename
section_title
page_number
related_labs
analysis_types
contains_warning
contains_steps
is_maintenance
```

For PostgreSQL, `related_labs=Biochemistry` and `analysis_types=Immunoassay` match list metadata using JSONB containment. ChromaDB stores scalar metadata only, so lab-scope access control should use PostgreSQL in production.

Example:

```bash
python scripts/query.py "How do I clean the device?" \
  --user-id alice \
  --filter document_type=IFU \
  --filter is_maintenance=true
```

## Query Pipeline

The query flow is:

```text
question
  -> embedding
  -> configured retrieval backend
  -> top-k chunks
  -> citation-aware LLM prompt
  -> structured response
  -> audit log event
```

The structured response contains:

```text
question
answer
citations
results
audit_id
```

Each citation contains:

```text
filename
source
version
page
section
chunk_id
score
```

## LLM Backends

The LLM interface is pluggable.

### Ollama

Default `.env`:

```text
LLM_BACKEND=ollama
LLM_MODEL=llama3.1
LLM_BASE_URL=http://localhost:11434
LLM_API_KEY=
```

### vLLM On RTX 3090

For a separate RTX 3090 ML server running vLLM and Llama 3.1, use:

```text
LLM_BACKEND=vllm
LLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
LLM_BASE_URL=http://YOUR_3090_SERVER_IP:8000/v1
LLM_API_KEY=replace-with-a-long-random-token
```

Full setup guide:

```text
docs/VLLM_LLAMA_3090.md
```

## Streamlit UI

Start the UI:

```bash
streamlit run app/streamlit_app.py
```

The UI includes:

- Question input
- Top-k selector
- User id field for audit logging
- SOP/IFU and warning/maintenance filters
- Answer display
- Cited sources
- Optional retrieved chunk debug view

## Evaluation

The evaluation module runs retrieval-only checks and does not call the LLM.

Run the default evaluation questions:

```bash
python scripts/evaluate.py --questions eval_questions.json --top-k 5
```

Prompt for manual scores and notes:

```bash
python scripts/evaluate.py --manual-score
```

Evaluation prints:

- Test question
- Simple extractive answer from retrieved chunks
- Cited sources
- Retrieved chunks
- Optional manual score and notes

Edit `eval_questions.json` to add more questions. Each item may be a string or an object:

```json
{
  "question": "How should maintenance be performed?",
  "filters": {
    "is_maintenance": true
  }
}
```

## Audit Logging

Query audit logging is enabled by default:

```text
AUDIT_LOG_ENABLED=true
AUDIT_LOG_PATH=data/audit/query_audit.jsonl
AUDIT_INCLUDE_TEXT=true
```

Each query event records:

- Audit id
- UTC timestamp
- User id and session id
- Question and answer
- Retrieval filters and top-k
- Embedding backend and model
- LLM backend and model
- Chroma collection
- Retrieved chunk ids, metadata, scores, and text hashes
- Citations with filename, page, section, chunk id, and score
- Success or error outcome
- Latency in milliseconds
- Previous record hash and current record hash

Verify the audit hash chain:

```bash
python scripts/verify_audit.py
```

For privacy-sensitive environments:

```text
AUDIT_INCLUDE_TEXT=false
```

That omits raw question, answer, and retrieved chunk text while keeping hashes, metadata, citations, and model configuration. Use stable pseudonymous user ids if logs may contain patient or operator-identifying information.

This scaffold provides audit primitives. Clinical deployment still needs organization-approved retention, access control, encryption, backup, monitoring, and compliance review.

## Common Commands

```bash
# Install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Ingest PDFs
python scripts/ingest.py

# Query
python scripts/query.py "What warnings are listed?" --user-id alice

# Query with metadata filters
python scripts/query.py "How should cleaning be performed?" \
  --user-id alice \
  --filter document_type=IFU \
  --filter is_maintenance=true

# Run UI
streamlit run app/streamlit_app.py

# Evaluate retrieval without LLM
python scripts/evaluate.py --manual-score

# Verify audit log
python scripts/verify_audit.py
```

## Notes

This is a scaffold, not a complete regulated clinical product. Before production clinical use, add and validate authentication, authorization, encryption at rest, retention policy, backup and restore, audit export, monitoring, model evaluation, human review workflows, and deployment controls.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
