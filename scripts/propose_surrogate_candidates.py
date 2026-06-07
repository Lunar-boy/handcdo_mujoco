#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from handcdo.surrogate import propose_candidates, train_surrogate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight surrogate and propose candidate hand designs.")
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--search-space", default="configs/search_space.yaml")
    parser.add_argument("--target")
    parser.add_argument("--model-type", choices=("random_forest", "extra_trees"), default="random_forest")
    parser.add_argument("--n-random", type=int, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--existing-csv")
    parser.add_argument("--min-rows", type=int, default=5)
    parser.add_argument("--exclude-existing", dest="exclude_existing", action="store_true", default=True)
    parser.add_argument("--no-exclude-existing", dest="exclude_existing", action="store_false")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    model_path = train_surrogate(
        results_csv=args.results_csv,
        output_dir=output_dir / "model",
        search_space=args.search_space,
        target=args.target,
        model_type=args.model_type,
        seed=args.seed,
        min_rows=args.min_rows,
    )
    propose_candidates(
        model_path=model_path,
        search_space=args.search_space,
        n_random=args.n_random,
        top_k=args.top_k,
        output_dir=output_dir,
        seed=args.seed,
        existing_csv=args.existing_csv,
        exclude_existing=args.exclude_existing,
    )
    print(f"Wrote proposals to {output_dir / 'proposed_candidates.csv'}")
    print(f"Wrote proposed designs to {output_dir / 'proposed_designs'}")


if __name__ == "__main__":
    main()
