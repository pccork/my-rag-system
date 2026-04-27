from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_system.config import get_settings
from rag_system.ingest import ingest
from rag_system.ingest_validation import validate_ingest_metadata


def main() -> None:
    settings = get_settings()
    report = validate_ingest_metadata(settings.docs_dir, settings.metadata_dir)
    for warning in report.warnings:
        print(f"Warning: {warning}")
    count = ingest(reset=True, validate_metadata=False)
    if count == 0:
        print("No PDFs found. Add files to ./data/raw and run ingestion again.")
        return
    print(f"Ingested {count} chunks into {settings.vector_store_backend}.")


if __name__ == "__main__":
    main()
