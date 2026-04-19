from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_system.query import format_citation_list, query
from rag_system.vector_store import MetadataFilter


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the local RAG index.")
    parser.add_argument("question", help="Question to ask the indexed PDFs.")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve.")
    parser.add_argument("--user-id", default="cli-user", help="User identifier for audit logging.")
    parser.add_argument("--session-id", default=None, help="Session identifier for audit logging.")
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Metadata filter. Can be repeated, e.g. --filter document_type=IFU.",
    )
    args = parser.parse_args()

    response = query(
        args.question,
        top_k=args.top_k,
        filters=parse_filters(args.filter),
        user_id=args.user_id,
        session_id=args.session_id,
        request_metadata={"channel": "cli"},
    )
    print("\nAnswer\n------")
    print(response.answer)
    print(f"\nAudit ID: {response.audit_id}")
    print("\nSources\n-------")
    print(format_citation_list(response.citations))
    print("\nRetrieved chunks\n----------------")
    for index, result in enumerate(response.results, start=1):
        print(f"\n[{index}] {result.text[:500]}")


def parse_filters(values: list[str]) -> MetadataFilter | None:
    filters: MetadataFilter = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"Invalid filter '{item}'. Use KEY=VALUE.")
        key, value = item.split("=", 1)
        filters[key.strip()] = parse_filter_value(value.strip())
    return filters or None


def parse_filter_value(value: str) -> str | int | float | bool:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


if __name__ == "__main__":
    main()
