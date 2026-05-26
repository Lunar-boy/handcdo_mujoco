from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def flatten_result(payload: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "design_id": payload.get("design_id"),
        "hand_score": payload.get("hand_score", 0.0),
        "failed": payload.get("failed", False),
        "error": payload.get("error"),
    }
    row.update(payload.get("parameters") or {})
    for tool_result in payload.get("tool_results", []):
        tool = tool_result.get("tool")
        if tool:
            row[f"{tool}_best_score"] = tool_result.get("best_score", 0.0)
    return row


def collect_results(results_dir: str | Path, output_csv: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            rows.append(flatten_result(json.loads(path.read_text(encoding="utf-8"))))
        except Exception as exc:
            rows.append({"design_id": path.stem, "hand_score": 0.0, "failed": True, "error": f"{type(exc).__name__}: {exc}"})
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    except Exception:
        import csv

        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for row in rows for k in row})
        with Path(output_csv).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="outputs/results")
    parser.add_argument("--output-csv", default="outputs/results.csv")
    args = parser.parse_args()
    rows = collect_results(args.results_dir, args.output_csv)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
