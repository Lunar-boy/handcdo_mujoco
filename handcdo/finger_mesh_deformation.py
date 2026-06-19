from __future__ import annotations

import numpy as np

from .geometry_config import FingerContactConfig
from .hand_model import LinkSpec


def fingertip_contact_half_extents(
    link: LinkSpec,
    finger_config: FingerContactConfig,
) -> tuple[float, float, float]:
    """Return the contact half-extents of the configured fingertip body."""
    if (
        finger_config.fingertip_body_shape == "ellipsoid"
        and link.fingertip_geometry is not None
    ):
        geometry = link.fingertip_geometry
        return (geometry.half_x, geometry.half_y, geometry.half_z)
    return (
        min(0.008, max(0.003, 0.28 * link.length)),
        link.radius,
        link.radius,
    )


def compute_fingertip_height_field(
    link: LinkSpec,
    *,
    resolution: int,
    margin_ratio: float,
    max_height: float,
    min_height: float,
    finger_config: FingerContactConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a deterministic Gaussian height field on the palmar fingertip surface."""
    if resolution < 2:
        raise ValueError(f"resolution must be >= 2; got {resolution!r}")
    if not 0 <= margin_ratio < 0.5:
        raise ValueError(f"margin_ratio must be >= 0 and < 0.5; got {margin_ratio!r}")
    if min_height < 0:
        raise ValueError(f"min_height must be >= 0; got {min_height!r}")

    config = finger_config or FingerContactConfig()
    contact_half_x, contact_half_y, _ = fingertip_contact_half_extents(link, config)
    x_min = link.length - 2.0 * contact_half_x
    x_max = link.length
    usable_y = contact_half_y * (1.0 - margin_ratio)
    x = np.linspace(x_min, x_max, resolution + 1)
    y = np.linspace(-usable_y, usable_y, resolution + 1)
    X, Y = np.meshgrid(x, y)
    H = np.zeros_like(X, dtype=float)
    if max_height <= 0:
        return X, Y, H

    center_x = link.length - contact_half_x
    sigma_x = max(1e-6, 0.60 * contact_half_x)
    sigma_y = max(1e-6, 0.60 * usable_y)
    H = max_height * np.exp(
        -0.5
        * (
            ((X - center_x) / sigma_x) ** 2
            + (Y / sigma_y) ** 2
        )
    )
    return X, Y, np.clip(H, min_height, max_height)
