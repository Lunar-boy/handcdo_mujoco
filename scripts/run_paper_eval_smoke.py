#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from handcdo.paper_eval_protocol import run_paper_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic paper-like grasp stability protocol.")
    parser.add_argument("--config", default="configs/eval_paper_protocol.yaml")
    parser.add_argument("--num-designs", type=int, default=2)
    parser.add_argument("--output", default="outputs/paper_eval_smoke")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--backend", default="deterministic_smoke", choices=["deterministic_smoke", "mujoco", "mujoco_cpu"])
    parser.add_argument("--search-space", default="configs/search_space.yaml")
    args = parser.parse_args()
    run_paper_evaluation(
        args.config,
        num_designs=args.num_designs,
        output_dir=args.output,
        seed=args.seed,
        backend_name=args.backend,
        search_space_path=args.search_space,
    )


if __name__ == "__main__":
    main()
