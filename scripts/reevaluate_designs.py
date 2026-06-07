#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from handcdo.multifidelity import load_design_ids, reevaluate_designs


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate selected design ids at a named fidelity.")
    parser.add_argument("--design-dir", required=True)
    parser.add_argument("--design-ids", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--fidelity", required=True)
    parser.add_argument("--tools", required=True)
    parser.add_argument("--backend", default="mujoco_cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-grasp-trials", type=int)
    parser.add_argument("--sampler")
    parser.add_argument("--tool-assets-dir", default="assets/tools")
    args = parser.parse_args()
    tools = [tool.strip() for tool in args.tools.split(",") if tool.strip()]
    payloads = reevaluate_designs(
        design_dir=args.design_dir,
        design_ids=load_design_ids(args.design_ids),
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        fidelity=args.fidelity,
        tools=tools,
        backend=args.backend,
        seed=args.seed,
        n_grasp_trials=args.n_grasp_trials,
        sampler=args.sampler,
        tool_assets_dir=args.tool_assets_dir,
    )
    print(f"Wrote {len(payloads)} {args.fidelity} result payloads to {args.results_dir}")


if __name__ == "__main__":
    main()
