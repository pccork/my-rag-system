from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rag_system.config import Settings, get_settings
from rag_system.models import SearchResult
from rag_system.query import build_citations, compose_extractive_answer, format_citation_list, retrieve
from rag_system.vector_store import MetadataFilter


DEFAULT_QUESTIONS = [
    "What are the setup steps?",
    "What warnings or cautions are listed?",
    "How should maintenance or cleaning be performed?",
]


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    filters: MetadataFilter | None = None


@dataclass(frozen=True)
class EvaluationResult:
    question: str
    answer: str
    results: list[SearchResult]
    manual_score: str | None = None
    notes: str | None = None


def load_cases(path: Path | None = None) -> list[EvaluationCase]:
    if path is None:
        return [EvaluationCase(question=question) for question in DEFAULT_QUESTIONS]

    with path.open("r", encoding="utf-8") as file:
        raw_cases = json.load(file)

    cases: list[EvaluationCase] = []
    for item in raw_cases:
        if isinstance(item, str):
            cases.append(EvaluationCase(question=item))
        else:
            cases.append(
                EvaluationCase(
                    question=str(item["question"]),
                    filters=item.get("filters"),
                )
            )
    return cases


def run_case(
    case: EvaluationCase,
    settings: Settings | None = None,
    top_k: int | None = None,
) -> EvaluationResult:
    settings = settings or get_settings()
    results = retrieve(
        case.question,
        settings=settings,
        top_k=top_k,
        filters=case.filters,
    )
    answer = compose_extractive_answer(case.question, results)
    return EvaluationResult(question=case.question, answer=answer, results=results)


def print_result(result: EvaluationResult) -> None:
    print("\n" + "=" * 80)
    print(f"Question: {result.question}")
    print("\nAnswer")
    print("------")
    print(result.answer)
    print("\nSources")
    print("-------")
    print(format_citation_list(build_citations(result.results)) or "No sources retrieved.")
    print("\nRetrieved chunks")
    print("----------------")
    for index, item in enumerate(result.results, start=1):
        print(f"\n[{index}] score={item.score if item.score is not None else 'n/a'}")
        print(item.text[:1200])


def prompt_manual_score() -> tuple[str | None, str | None]:
    score = input("\nManual score (blank to skip): ").strip()
    notes = input("Notes (blank to skip): ").strip()
    return score or None, notes or None
