#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from handcdo.multifidelity import merge_multifidelity_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge collected results CSVs across fidelity levels.")
    parser.add_argument("--fast-csv")
    parser.add_argument("--medium-csv")
    parser.add_argument("--high-csv")
    parser.add_argument("--input", action="append", default=[], help="Additional input as fidelity=path/to/results.csv")
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    inputs = {}
    for fidelity, path in (("fast", args.fast_csv), ("medium", args.medium_csv), ("high", args.high_csv)):
        if path:
            inputs[fidelity] = path
    for item in args.input:
        if "=" not in item:
            raise SystemExit(f"--input must be in fidelity=path form, got {item!r}")
        fidelity, path = item.split("=", 1)
        fidelity = fidelity.strip()
        if not fidelity:
            raise SystemExit(f"--input has empty fidelity name: {item!r}")
        inputs[fidelity] = path

    merged = merge_multifidelity_results(inputs, args.output_csv)
    print(f"Wrote {len(merged)} merged rows to {args.output_csv}")


if __name__ == "__main__":
    main()
