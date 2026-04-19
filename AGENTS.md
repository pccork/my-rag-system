# AGENTS.md

## Project Overview
This project is a local Retrieval-Augmented Generation (RAG) system for querying technical documents such as IFUs (Instructions for Use) and SOPs (Standard Operating Procedures).

The system prioritizes:
- accuracy over creativity
- traceability (citations required)
- structured understanding of procedures

---

## Architecture Rules

- Always maintain modular structure:
  - ingestion/
  - chunking/
  - embeddings/
  - vectorstore/
  - retrieval/
  - ui/

- Do NOT mix responsibilities between modules
- Keep interfaces clean and replaceable

---

## Coding Standards

- Use Python
- Type hints required
- Functions must be small and focused
- Prefer clarity over cleverness
- Add docstrings to all public functions

---

## Retrieval Rules

- Always include metadata:
  - filename
  - page
  - section
- Never return answers without citations
- Prefer precise chunks over large context

---

## Chunking Rules

- Respect document structure:
  - warnings must not be split
  - procedural steps must stay intact
- Avoid arbitrary fixed-size chunking when structure exists

---

## LLM Usage Rules

- Do not hallucinate
- If answer is not found → say "Not found in documents"
- Use retrieved context only

---

## Development Workflow

- Plan before coding
- Show file changes clearly
- Do not modify unrelated files
- Keep dependencies minimal

---

## Future Extensions

- Knowledge graph for multi-hop reasoning
- Hybrid retrieval (vector + graph)
- Reranking models