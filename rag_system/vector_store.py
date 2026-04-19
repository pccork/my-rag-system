from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeAlias

from rag_system.models import DocumentChunk, SearchResult


MetadataValue: TypeAlias = str | int | float | bool
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


def is_chroma_metadata_value(value: object) -> bool:
    return isinstance(value, str | int | float | bool)
