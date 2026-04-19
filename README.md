# Local Clinical RAG System

A local Retrieval Augmented Generation project for SOP and IFU PDFs, with structured ingestion, intelligent chunking, ChromaDB retrieval, LLM answering, citations, Streamlit UI, retrieval evaluation, and query audit logging.

## Capabilities

- Ingests PDFs from `./docs`
- Extracts structured text while preserving headings, warnings, cautions, table-like rows, and page metadata
- Infers document type as `SOP`, `IFU`, or `unknown`
- Chunks SOP/IFU content by headings, numbered steps, warnings, cautions, and maintenance sections
- Uses pluggable embeddings with `all-mpnet-base-v2` by default
- Stores vectors locally in persistent ChromaDB
- Supports exact-match metadata filtering during retrieval
- Generates answers through a pluggable LLM backend
- Supports local Ollama or a remote vLLM server on an RTX 3090
- Returns source citations with filename, page, section, chunk id, and score
- Provides CLI and Streamlit interfaces
- Includes retrieval-only evaluation with manual scoring
- Writes hash-chained query audit logs for user/session traceability

## Project Layout

```text
.
├── app/
│   └── streamlit_app.py
├── data/
│   ├── audit/
│   │   └── .gitkeep
│   └── chroma/
│       └── .gitkeep
├── docs/
│   ├── .gitkeep
│   └── VLLM_LLAMA_3090.md
├── rag_system/
│   ├── audit.py
│   ├── chunking.py
│   ├── config.py
│   ├── embeddings.py
│   ├── evaluation.py
│   ├── ingest.py
│   ├── llm.py
│   ├── loaders.py
│   ├── models.py
│   ├── query.py
│   └── vector_store.py
├── scripts/
│   ├── evaluate.py
│   ├── ingest.py
│   ├── query.py
│   └── verify_audit.py
├── eval_questions.json
├── .env.example
├── .gitignore
└── requirements.txt
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
docs/
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
DOCS_DIR=docs
CHROMA_DIR=data/chroma
CHROMA_COLLECTION=local_rag_docs

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

PDFs are read from `DOCS_DIR`, which defaults to `docs`.

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

Vectors are stored in persistent ChromaDB:

```text
data/chroma
```

Retrieval supports exact-match metadata filters, including:

```text
document_type
filename
section_title
page_number
contains_warning
contains_steps
is_maintenance
```

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
  -> ChromaDB semantic search
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
