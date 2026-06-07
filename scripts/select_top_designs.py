#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from handcdo.multifidelity import select_top_designs


def main() -> None:
    parser = argparse.ArgumentParser(description="Select top design ids from a collected results CSV.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--output-design-ids", required=True)
    parser.add_argument("--score-column", default="hand_score")
    parser.add_argument("--include-failed", action="store_true")
    args = parser.parse_args()
    design_ids = select_top_designs(
        args.input_csv,
        args.top_k,
        args.output_design_ids,
        score_column=args.score_column,
        include_failed=args.include_failed,
    )
    print(f"Wrote {len(design_ids)} design ids to {args.output_design_ids}")


if __name__ == "__main__":
    main()
