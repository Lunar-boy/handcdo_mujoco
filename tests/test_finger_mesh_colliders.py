from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from handcdo.design_space import DesignSpace
from handcdo.finger_mesh_colliders import (
    build_fingertip_local_mesh_colliders,
    export_fingertip_local_mesh_colliders,
)
from handcdo.finger_mesh_deformation import fingertip_contact_half_extents
from handcdo.geometry_config import FingerContactConfig
from handcdo.hand_model import build_hand_model


def _digit_and_link():
    hand = build_hand_model(DesignSpace().sample(seed=302))
    digit = hand.digits[0]
    return digit, digit.links[-1]


def _config(collider_type: str = "quad_frustum", resolution: int = 4, **overrides):
    values = {
        "mode": "local_convex_patches",
        "local_patch_resolution": resolution,
        "local_patch_collider_type": collider_type,
        "local_patch_thickness": 0.0025,
        "local_patch_margin_ratio": 0.05,
        "local_patch_max_height": 0.003,
        "local_patch_min_height": 0.0004,
        "max_num_local_patch_colliders": (
            2 * resolution**2
            if collider_type == "triangular_prism"
            else resolution**2
        ),
    }
    values.update(overrides)
    return FingerContactConfig(**values)


@pytest.mark.parametrize(
    ("collider_type", "expected_count"),
    [("quad_frustum", 16), ("triangular_prism", 32)],
)
def test_local_patch_collider_count_and_mesh_validity(collider_type, expected_count):
    digit, link = _digit_and_link()
    config = _config(collider_type)
    colliders = build_fingertip_local_mesh_colliders(digit, link, config)

    assert len(colliders) == expected_count
    assert len({collider.name for collider in colliders}) == expected_count
    assert all(len(collider.mesh.vertices) > 0 for collider in colliders)
    assert all(len(collider.mesh.faces) > 0 for collider in colliders)
    assert all(collider.mesh.is_watertight for collider in colliders)
    assert all(collider.mesh.is_convex for collider in colliders)
    assert all(np.all(np.isfinite(collider.mesh.vertices)) for collider in colliders)

    contact_half_x, contact_half_y, contact_half_z = fingertip_contact_half_extents(
        link,
        config,
    )
    vertices = np.vstack([collider.mesh.vertices for collider in colliders])
    assert vertices[:, 0].min() == pytest.approx(link.length - 2 * contact_half_x)
    assert vertices[:, 0].max() == pytest.approx(link.length)
    assert np.abs(vertices[:, 1]).max() <= contact_half_y + 1e-12
    assert vertices[:, 2].min() < -contact_half_z
    assert vertices[:, 2].max() == pytest.approx(
        -contact_half_z + config.local_patch_thickness
    )


def test_nonterminal_link_produces_no_local_patch_colliders():
    digit, _ = _digit_and_link()

    assert build_fingertip_local_mesh_colliders(
        digit,
        digit.links[0],
        _config(),
    ) == []


def test_collider_names_are_deterministic():
    digit, link = _digit_and_link()
    quad = build_fingertip_local_mesh_colliders(digit, link, _config())
    triangles = build_fingertip_local_mesh_colliders(
        digit,
        link,
        _config("triangular_prism"),
    )

    assert quad[0].name == f"{link.name}_local_patch_r00_c00"
    assert quad[-1].name == f"{link.name}_local_patch_r03_c03"
    assert triangles[0].name == f"{link.name}_local_patch_r00_c00_tri0"
    assert triangles[-1].name == f"{link.name}_local_patch_r03_c03_tri1"


def test_export_local_patch_colliders(tmp_path):
    digit, link = _digit_and_link()
    colliders = build_fingertip_local_mesh_colliders(
        digit,
        link,
        _config(resolution=2),
    )

    paths = export_fingertip_local_mesh_colliders(colliders, tmp_path)

    assert len(paths) == 4
    assert all(path.exists() for path in paths)
