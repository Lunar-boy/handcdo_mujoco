#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from handcdo.design_space import HandDesign
from handcdo.geometry_config import PalmContactConfig
from handcdo.hand_model import build_hand_model
from handcdo.multifidelity import load_design_ids
from handcdo.palm_mesh_colliders import (
    PalmMeshCollider,
    build_palm_tiled_mesh_colliders,
    export_palm_tiled_mesh_colliders,
)
from handcdo.utils import write_json


def _collider_height_range(
    colliders: list[PalmMeshCollider],
    *,
    top_z: float,
    bottom_z: float,
) -> tuple[float, float]:
    top_heights = [
        float(vertex[2]) - top_z
        for collider in colliders
        for vertex in collider.mesh.vertices
        if float(vertex[2]) > bottom_z + 1e-12
    ]
    if not top_heights:
        return (0.0, 0.0)
    return (min(top_heights), max(top_heights))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export deterministic local mesh colliders for a deformed palm height field."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--design-json")
    source.add_argument("--design-dir")
    parser.add_argument("--design-ids", help="Optional newline-delimited design IDs for --design-dir.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resolution", type=int, default=6)
    parser.add_argument(
        "--collider-type",
        choices=("quad_frustum", "triangular_prism"),
        default="quad_frustum",
    )
    parser.add_argument(
        "--domain",
        choices=("bbox", "outline"),
        default="bbox",
        help="Use the legacy rectangular domain or clip tiles to the palm outline.",
    )
    parser.add_argument("--thickness", type=float, default=0.003)
    parser.add_argument("--margin-ratio", type=float, default=0.0)
    parser.add_argument("--format", choices=("obj", "stl"), default="stl")
    args = parser.parse_args()

    if args.design_json and args.design_ids:
        parser.error("--design-ids can only be used with --design-dir")

    output_root = Path(args.output_dir)
    if args.design_json:
        design_files = [Path(args.design_json)]
        batch = False
    else:
        design_root = Path(args.design_dir)
        if args.design_ids:
            design_files = [
                design_root / design_id / "design.json"
                for design_id in load_design_ids(args.design_ids)
            ]
        else:
            design_files = sorted(design_root.glob("*/design.json"))
        batch = True

    if not design_files:
        parser.error("No design JSON files found")

    multiplier = 2 if args.collider_type == "triangular_prism" else 1
    if args.domain == "outline" and args.collider_type == "triangular_prism":
        multiplier = 8
    config = PalmContactConfig(
        mode="tiled_mesh_colliders",
        mesh_collider_resolution=args.resolution,
        mesh_collider_type=args.collider_type,
        mesh_collider_domain=args.domain,
        mesh_collider_thickness=args.thickness,
        mesh_collider_margin_ratio=args.margin_ratio,
        max_num_mesh_colliders=multiplier * args.resolution**2,
    )

    for design_file in design_files:
        if not design_file.exists():
            raise FileNotFoundError(f"Missing design JSON: {design_file}")
        design = HandDesign.from_json(design_file)
        if batch and design.design_id != design_file.parent.name:
            raise ValueError(
                f"Design id {design.design_id!r} does not match directory "
                f"{design_file.parent.name!r}"
            )
        destination = output_root / design.design_id if batch else output_root
        hand = build_hand_model(design)
        colliders = build_palm_tiled_mesh_colliders(hand, config)
        export_palm_tiled_mesh_colliders(colliders, destination, file_format=args.format)
        height_min, height_max = _collider_height_range(
            colliders,
            top_z=float(hand.palm_size[2]),
            bottom_z=float(hand.palm_size[2]) - args.thickness,
        )
        write_json(
            destination / "palm_mesh_collider_metadata.json",
            {
                "type": "palm_tiled_mesh_colliders",
                "design_id": design.design_id,
                "mesh_collider_resolution": args.resolution,
                "mesh_collider_type": args.collider_type,
                "mesh_collider_domain": args.domain,
                "num_colliders": len(colliders),
                "resolution": args.resolution,
                "collider_type": args.collider_type,
                "domain": args.domain,
                "collider_count": len(colliders),
                "thickness": args.thickness,
                "margin_ratio": args.margin_ratio,
                "height_min": height_min,
                "height_max": height_max,
                "format": args.format,
                "note": "static small collider decomposition of the selected palm domain",
            },
        )
        print(destination)


if __name__ == "__main__":
    main()
