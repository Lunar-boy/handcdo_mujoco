from __future__ import annotations

import numpy as np
import pytest

from handcdo.design_space import DesignSpace
from handcdo.finger_mesh_deformation import compute_fingertip_height_field
from handcdo.geometry_config import FingerContactConfig
from handcdo.hand_model import build_hand_model


def _terminal_link():
    hand = build_hand_model(DesignSpace().sample(seed=301))
    return hand.digits[0].links[-1]


def test_fingertip_height_field_shape_bounds_and_center_peak():
    link = _terminal_link()
    max_height = 0.003
    X, Y, H = compute_fingertip_height_field(
        link,
        resolution=4,
        margin_ratio=0.05,
        max_height=max_height,
        min_height=0.0004,
        finger_config=FingerContactConfig(fingertip_body_shape="ellipsoid"),
    )

    assert X.shape == (5, 5)
    assert Y.shape == (5, 5)
    assert H.shape == (5, 5)
    assert np.all(np.isfinite(H))
    assert np.all(H >= 0)
    assert np.all(H <= max_height)
    assert np.unravel_index(np.argmax(H), H.shape) == (2, 2)


def test_nonpositive_max_height_returns_zero_height_field():
    _, _, H = compute_fingertip_height_field(
        _terminal_link(),
        resolution=3,
        margin_ratio=0.0,
        max_height=0.0,
        min_height=0.0004,
    )

    assert np.count_nonzero(H) == 0


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"resolution": 1}, "resolution"),
        ({"margin_ratio": 0.5}, "margin_ratio"),
        ({"min_height": -0.001}, "min_height"),
    ],
)
def test_invalid_height_field_inputs_raise_value_error(overrides, field):
    values = {
        "resolution": 4,
        "margin_ratio": 0.05,
        "max_height": 0.003,
        "min_height": 0.0004,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=field):
        compute_fingertip_height_field(_terminal_link(), **values)
