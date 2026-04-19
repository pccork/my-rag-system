from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_system.evaluation import load_cases, print_result, prompt_manual_score, run_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation questions.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="Optional JSON file containing questions or objects with question/filters.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--manual-score",
        action="store_true",
        help="Prompt for a manual score and notes after each question.",
    )
    args = parser.parse_args()

    cases = load_cases(args.questions)
    for case in cases:
        result = run_case(case, top_k=args.top_k)
        print_result(result)
        if args.manual_score:
            score, notes = prompt_manual_score()
            if score:
                print(f"Recorded manual score: {score}")
            if notes:
                print(f"Recorded notes: {notes}")


if __name__ == "__main__":
    main()
