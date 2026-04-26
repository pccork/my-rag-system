from __future__ import annotations

from rag_system.chunking import chunk_pages
from rag_system.config import Settings, get_settings
from rag_system.embeddings import get_embedding_provider
from rag_system.loaders import load_pdfs
from rag_system.vector_store import ChromaVectorStore


def ingest(settings: Settings | None = None, reset: bool = True) -> int:
    settings = settings or get_settings()
    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    settings.metadata_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    pages = load_pdfs(settings.docs_dir, metadata_dir=settings.metadata_dir)
    if not pages:
        return 0

    chunks = chunk_pages(
        pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    provider = get_embedding_provider(
        settings.embedding_backend,
        settings.embedding_model,
    )
    embeddings = provider.embed_documents([chunk.text for chunk in chunks])

    store = ChromaVectorStore(settings.chroma_dir, settings.chroma_collection)
    if reset:
        store.reset()
    store.add(chunks, embeddings)
    return len(chunks)
