#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from handcdo.design_space import HandDesign
from handcdo.hand_model import build_hand_model
from handcdo.multifidelity import load_design_ids
from handcdo.palm_mesh_deformation import PalmSurfaceMeshConfig, export_palm_surface_mesh


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export static Gaussian-deformed palm visual meshes without changing MuJoCo collision."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--design-json")
    source.add_argument("--design-dir")
    parser.add_argument("--design-ids", help="Optional newline-delimited design IDs for --design-dir.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--margin-ratio", type=float, default=0.0)
    parser.add_argument("--include-skirt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skirt-depth", type=float, default=0.003)
    parser.add_argument("--max-height-cap", type=float)
    parser.add_argument("--formats", default="obj,stl")
    args = parser.parse_args()

    if args.design_json and args.design_ids:
        parser.error("--design-ids can only be used with --design-dir")

    formats = tuple(item.strip() for item in args.formats.split(",") if item.strip())
    config = PalmSurfaceMeshConfig(
        resolution=args.resolution,
        margin_ratio=args.margin_ratio,
        include_skirt=args.include_skirt,
        skirt_depth=args.skirt_depth,
        max_height_cap=args.max_height_cap,
    )
    output_root = Path(args.output_dir)

    if args.design_json:
        design_files = [Path(args.design_json)]
        batch = False
    else:
        design_root = Path(args.design_dir)
        if args.design_ids:
            design_files = [design_root / design_id / "design.json" for design_id in load_design_ids(args.design_ids)]
        else:
            design_files = sorted(design_root.glob("*/design.json"))
        batch = True

    if not design_files:
        parser.error("No design JSON files found")

    for design_file in design_files:
        if not design_file.exists():
            raise FileNotFoundError(f"Missing design JSON: {design_file}")
        design = HandDesign.from_json(design_file)
        if batch and design.design_id != design_file.parent.name:
            raise ValueError(
                f"Design id {design.design_id!r} does not match directory {design_file.parent.name!r}"
            )
        destination = output_root / design.design_id if batch else output_root
        export_palm_surface_mesh(build_hand_model(design), destination, config=config, formats=formats)
        print(destination)


if __name__ == "__main__":
    main()
