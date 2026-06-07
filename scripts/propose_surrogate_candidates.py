#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from handcdo.surrogate import propose_candidates, train_surrogate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight surrogate and propose candidate hand designs.")
    parser.add_argument("--mode", choices=("train-propose", "train-only", "propose-only"), default="train-propose")
    parser.add_argument("--results-csv")
    parser.add_argument("--model-path")
    parser.add_argument("--search-space")
    parser.add_argument("--target")
    parser.add_argument("--model-type", choices=("random_forest", "extra_trees"), default="random_forest")
    parser.add_argument("--n-random", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--existing-csv")
    parser.add_argument("--min-rows", type=int, default=5)
    parser.add_argument("--exclude-existing", dest="exclude_existing", action="store_true", default=True)
    parser.add_argument("--no-exclude-existing", dest="exclude_existing", action="store_false")
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", default=False)
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    args = parser.parse_args(argv)

    _validate_args(parser, args)

    output_dir = Path(args.output_dir)
    model_path: Path | str
    if args.mode in {"train-propose", "train-only"}:
        model_path = train_surrogate(
            results_csv=args.results_csv,
            output_dir=output_dir / "model",
            search_space=args.search_space or "configs/search_space.yaml",
            target=args.target,
            model_type=args.model_type,
            seed=args.seed,
            min_rows=args.min_rows,
        )
        diagnostics_path = output_dir / "model" / "surrogate_diagnostics.json"
        print(f"Wrote surrogate model to {model_path}")
        print(f"Wrote diagnostics to {diagnostics_path}")
        if args.mode == "train-only":
            return
    else:
        model_path = args.model_path

    propose_candidates(
        model_path=model_path,
        search_space=args.search_space,
        n_random=args.n_random,
        top_k=args.top_k,
        output_dir=output_dir,
        seed=args.seed,
        existing_csv=args.existing_csv,
        exclude_existing=args.exclude_existing,
        overwrite=args.overwrite,
    )
    print(f"Wrote proposals to {output_dir / 'proposed_candidates.csv'}")
    print(f"Wrote proposed designs to {output_dir / 'proposed_designs'}")


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.mode in {"train-propose", "train-only"} and not args.results_csv:
        parser.error(f"--mode {args.mode} requires --results-csv")
    if args.mode == "propose-only" and not args.model_path:
        parser.error("--mode propose-only requires --model-path")
    if args.mode in {"train-propose", "propose-only"}:
        if args.n_random is None:
            parser.error(f"--mode {args.mode} requires --n-random")
        if args.top_k is None:
            parser.error(f"--mode {args.mode} requires --top-k")
        if args.top_k > args.n_random:
            parser.error("--top-k must be <= --n-random")


if __name__ == "__main__":
    main()
