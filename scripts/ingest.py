from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_system.ingest import ingest


def main() -> None:
    count = ingest(reset=True)
    if count == 0:
        print("No PDFs found. Add files to ./data/raw and run ingestion again.")
        return
    print(f"Ingested {count} chunks into ChromaDB.")


if __name__ == "__main__":
    main()
