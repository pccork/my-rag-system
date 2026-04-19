from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


DEFAULT_EMBEDDING_BACKEND = "sentence-transformers"
DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings = self.model.encode(list(texts), normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()


def get_embedding_provider(
    backend: str = DEFAULT_EMBEDDING_BACKEND,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> EmbeddingProvider:
    normalized = backend.strip().lower()
    if normalized in {"sentence-transformers", "sentence_transformers", "local"}:
        return SentenceTransformersEmbeddingProvider(model_name)
    raise ValueError(
        f"Unsupported embedding backend '{backend}'. "
        "Add a provider in rag_system/embeddings.py."
    )
