from __future__ import annotations

import re
from time import perf_counter

from rag_system.audit import new_audit_id, write_query_audit_event
from rag_system.config import Settings, get_settings
from rag_system.embeddings import get_embedding_provider
from rag_system.llm import get_llm_provider
from rag_system.models import Citation, QueryResponse, SearchResult
from rag_system.vector_store import MetadataFilter, PostgresHybridVectorStore, get_vector_store


def query(
    question: str,
    settings: Settings | None = None,
    top_k: int | None = None,
    filters: MetadataFilter | None = None,
    user_id: str = "unknown",
    session_id: str | None = None,
    request_metadata: dict[str, object] | None = None,
) -> QueryResponse:
    settings = settings or get_settings()
    audit_id = new_audit_id()
    started = perf_counter()
    results: list[SearchResult] = []
    citations: list[Citation] = []
    answer: str | None = None

    try:
        results = retrieve(question, settings=settings, top_k=top_k, filters=filters)
        citations = build_citations(results)
        llm = get_llm_provider(
            settings.llm_backend,
            settings.llm_model,
            settings.llm_base_url,
            settings.llm_api_key,
            settings.llm_temperature,
        )
        answer = llm.generate(build_prompt(question, results, citations))
    except Exception as exc:
        write_query_audit_event(
            settings=settings,
            audit_id=audit_id,
            user_id=user_id,
            session_id=session_id,
            question=question,
            top_k=top_k,
            filters=filters,
            results=results,
            citations=citations,
            answer=answer,
            status="error",
            latency_ms=elapsed_ms(started),
            error=f"{type(exc).__name__}: {exc}",
            request_metadata=request_metadata,
        )
        raise

    write_query_audit_event(
        settings=settings,
        audit_id=audit_id,
        user_id=user_id,
        session_id=session_id,
        question=question,
        top_k=top_k,
        filters=filters,
        results=results,
        citations=citations,
        answer=answer,
        status="success",
        latency_ms=elapsed_ms(started),
        request_metadata=request_metadata,
    )
    return QueryResponse(
        question=question,
        answer=answer,
        citations=citations,
        results=results,
        audit_id=audit_id,
    )


def retrieve(
    question: str,
    settings: Settings | None = None,
    top_k: int | None = None,
    filters: MetadataFilter | None = None,
) -> list[SearchResult]:
    settings = settings or get_settings()
    provider = get_embedding_provider(settings.embedding_backend, settings.embedding_model)
    store = get_vector_store(settings)
    query_embedding = provider.embed_query(question)
    if isinstance(store, PostgresHybridVectorStore):
        return store.hybrid_search(
            question,
            query_embedding,
            top_k or settings.retrieval_top_k,
            filters=filters,
        )
    return store.search(
        query_embedding,
        top_k or settings.retrieval_top_k,
        filters=filters,
    )


def elapsed_ms(started: float) -> int:
    return int((perf_counter() - started) * 1000)


def compose_extractive_answer(question: str, results: list[SearchResult]) -> str:
    if not results:
        return "No relevant chunks were retrieved."

    terms = meaningful_terms(question)
    sentences: list[tuple[int, str, int]] = []
    for index, result in enumerate(results, start=1):
        for sentence in split_sentences(result.text):
            score = sentence_score(sentence, terms)
            if score:
                sentences.append((score, sentence, index))

    if not sentences:
        return f"Top retrieved context: {results[0].text[:500]}"

    selected = sorted(sentences, key=lambda item: item[0], reverse=True)[:3]
    return " ".join(f"{sentence} [{index}]" for _, sentence, index in selected)


def meaningful_terms(question: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "for",
        "from",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
    return {
        term
        for term in re.findall(r"[a-z0-9]+", question.lower())
        if len(term) > 2 and term not in stopwords
    }


def split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]


def sentence_score(sentence: str, terms: set[str]) -> int:
    sentence_terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
    return len(sentence_terms & terms)


def build_prompt(
    question: str,
    results: list[SearchResult],
    citations: list[Citation],
) -> str:
    if not results:
        return (
            f"Question: {question}\n\n"
            "No context was retrieved. Say that the indexed documents do not contain "
            "enough information to answer."
        )

    context_blocks = []
    for citation, result in zip(citations, results, strict=False):
        context_blocks.append(
            "\n".join(
                [
                    f"[{citation.index}]",
                    f"Filename: {citation.filename}",
                    f"Source: {citation.source}",
                    f"Version: {citation.version}",
                    f"Page: {citation.page}",
                    f"Section: {citation.section or 'unknown'}",
                    "Text:",
                    result.text,
                ]
            )
        )

    return "\n\n".join(
        [
            "Use the context below to answer the question.",
            "Requirements:",
            "- Use only the retrieved context.",
            "- Include bracketed citations like [1] for each claim.",
            "- If the answer is not in the context, say the documents do not provide enough information.",
            "",
            f"Question: {question}",
            "",
            "Context:",
            "\n\n---\n\n".join(context_blocks),
        ]
    )


def build_citations(results: list[SearchResult]) -> list[Citation]:
    citations: list[Citation] = []
    for index, result in enumerate(results, start=1):
        citations.append(
            Citation(
                index=index,
                filename=str(
                    result.metadata.get("filename")
                    or result.metadata.get("source_name")
                    or "unknown"
                ),
                page=result.metadata.get("page_start")
                or result.metadata.get("page_number")
                or "unknown",
                section=str(result.metadata.get("section_title") or "unknown"),
                chunk_id=result.id,
                source=str(
                    result.metadata.get("source_name")
                    or result.metadata.get("source_path")
                    or result.metadata.get("filename")
                    or "unknown"
                ),
                version=str(
                    result.metadata.get("version")
                    or result.metadata.get("revision")
                    or result.metadata.get("document_version")
                    or "unknown"
                ),
                score=result.score,
            )
        )
    return citations


def format_citations(results: list[SearchResult]) -> str:
    return format_citation_list(build_citations(results))


def format_citation_list(citations: list[Citation]) -> str:
    lines: list[str] = []
    for citation in citations:
        score = f", score {citation.score:.3f}" if citation.score is not None else ""
        version = f", version: {citation.version}" if citation.version != "unknown" else ""
        lines.append(
            f"[{citation.index}] {citation.filename}, p. {citation.page}, "
            f"section: {citation.section}{version}{score}"
        )
    return "\n".join(lines)
