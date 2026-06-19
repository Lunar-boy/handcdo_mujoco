from __future__ import annotations

import json

import pytest

from handcdo.design_space import (
    DEFAULT_PALM_OUTLINE_PARAMETERS,
    DesignSpace,
    HandDesign,
)
from handcdo.hand_model import build_hand_model
from handcdo.mjcf_generator import build_mjcf_xml


PALM_PARAMETER_NAMES = set(DEFAULT_PALM_OUTLINE_PARAMETERS)


def _design_with(**overrides: float | int) -> HandDesign:
    params = DesignSpace().sample(seed=501).to_dict()
    params.update(DEFAULT_PALM_OUTLINE_PARAMETERS)
    params.update(overrides)
    return HandDesign(params)


def test_sampled_design_contains_bounded_palm_outline_parameters():
    space = DesignSpace()
    design = space.sample(seed=500)

    assert PALM_PARAMETER_NAMES <= design.params.keys()
    assert 0.070 <= design.params["palm_half_x"] <= 0.105
    assert 0.095 <= design.params["palm_half_y"] <= 0.135
    assert 0.024 <= design.params["palm_half_z"] <= 0.042
    assert 0.75 <= design.params["palm_aspect_ratio"] <= 1.30
    assert design.params["palm_polygon_sides"] in {6, 8, 10, 12}


def test_palm_outline_parameters_control_rigid_body_size():
    hand = build_hand_model(
        _design_with(
            palm_half_x=0.101,
            palm_half_y=0.123,
            palm_half_z=0.039,
            palm_aspect_ratio=1.2,
            palm_polygon_sides=10,
        )
    )

    assert hand.palm_body.half_extents == pytest.approx((0.101, 0.123 * 1.2, 0.039))
    assert hand.palm_body.outline_spec.polygon_sides == 10


def test_palm_kernel_height_does_not_control_rigid_body_size():
    low = build_hand_model(_design_with(palm_kernel_max_height=0.0))
    high = build_hand_model(_design_with(palm_kernel_max_height=0.035))

    assert low.palm_body.half_extents == pytest.approx(high.palm_body.half_extents)


def test_old_design_json_loads_with_legacy_palm_defaults(tmp_path):
    old_parameters = DesignSpace().sample(seed=502).to_dict()
    for name in PALM_PARAMETER_NAMES:
        old_parameters.pop(name)
    path = tmp_path / "old_design.json"
    path.write_text(json.dumps({"parameters": old_parameters}), encoding="utf-8")

    restored = HandDesign.from_json(path)

    for name, value in DEFAULT_PALM_OUTLINE_PARAMETERS.items():
        assert restored.params[name] == value
    assert build_hand_model(restored).palm_body.half_extents == pytest.approx(
        (0.085, 0.115, 0.032)
    )


def test_design_id_and_json_roundtrip_include_palm_outline_parameters(tmp_path):
    baseline = _design_with(palm_half_x=0.085)
    changed = _design_with(palm_half_x=0.086)

    assert baseline.design_id != changed.design_id

    path = tmp_path / "design.json"
    changed.to_json(path)
    restored = HandDesign.from_json(path)
    assert restored.design_id == changed.design_id
    assert restored.to_dict() == changed.to_dict()


@pytest.mark.parametrize(
    ("half_x", "half_y", "half_z", "aspect_ratio", "polygon_sides"),
    [
        (0.070, 0.095, 0.024, 0.75, 6),
        (0.085, 0.115, 0.032, 1.0, 8),
        (0.105, 0.135, 0.042, 1.30, 12),
    ],
)
def test_mjcf_generation_supports_varied_palm_outlines(
    half_x: float,
    half_y: float,
    half_z: float,
    aspect_ratio: float,
    polygon_sides: int,
):
    design = _design_with(
        palm_half_x=half_x,
        palm_half_y=half_y,
        palm_half_z=half_z,
        palm_aspect_ratio=aspect_ratio,
        palm_polygon_sides=polygon_sides,
    )

    xml = build_mjcf_xml(build_hand_model(design))

    assert 'name="palm_body_mesh"' in xml
    assert 'name="palm_geom" type="mesh"' in xml
