from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_system.audit import verify_hash_chain
from rag_system.config import get_settings


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Verify the query audit hash chain.")
    parser.add_argument(
        "--path",
        type=Path,
        default=settings.audit_log_path,
        help="Path to the JSONL audit log.",
    )
    args = parser.parse_args()

    ok, message = verify_hash_chain(args.path)
    print(message)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
