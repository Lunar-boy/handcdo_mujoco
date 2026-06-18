from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, TypeVar


@dataclass(frozen=True)
class FingerContactConfig:
    mode: str = "capsule"
    fingertip_pad_enabled: bool = False
    fingertip_pad_shape: str = "box"
    fingertip_pad_thickness: float = 0.004
    fingertip_pad_friction: tuple[float, float, float] = (1.4, 0.03, 0.003)


@dataclass(frozen=True)
class PalmContactConfig:
    mode: str = "box_pads"
    pad_resolution: int = 2
    pad_friction: tuple[float, float, float] = (1.4, 0.02, 0.002)
    max_num_pad_geoms: int = 16
    convex_patch_resolution: int = 4
    convex_patch_max_height: float | None = None
    convex_patch_base_thickness: float = 0.0025
    convex_patch_min_height: float = 0.0005
    convex_patch_margin_ratio: float = 0.15
    mesh_collider_resolution: int = 6
    mesh_collider_type: str = "quad_frustum"
    mesh_collider_thickness: float = 0.003
    mesh_collider_margin_ratio: float = 0.0
    max_num_mesh_colliders: int = 64
    mesh_collider_export: bool = False
    mesh_collider_export_dir: str | None = None


@dataclass(frozen=True)
class ToolContactConfig:
    mode: str = "primitive"
    friction: tuple[float, float, float] | None = None
    collision_margin: float = 0.001


@dataclass(frozen=True)
class GeometryConfig:
    finger: FingerContactConfig = field(default_factory=FingerContactConfig)
    palm: PalmContactConfig = field(default_factory=PalmContactConfig)
    tool: ToolContactConfig = field(default_factory=ToolContactConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GeometryConfig":
        data = data or {}
        if not isinstance(data, dict):
            raise ValueError(f"geometry config data={data!r} must be a mapping")

        if "geometry" in data:
            source = data["geometry"] or {}
            if not isinstance(source, dict):
                raise ValueError(f"geometry={source!r} must be a mapping")
            return cls(
                finger=_parse_section(source.get("finger"), FingerContactConfig, "geometry.finger"),
                palm=_parse_section(source.get("palm"), PalmContactConfig, "geometry.palm"),
                tool=_parse_section(source.get("tool"), ToolContactConfig, "geometry.tool"),
            )

        return cls(
            finger=_parse_section(_first_section(data, "finger_contact", "finger"), FingerContactConfig, "finger_contact"),
            palm=_parse_section(_first_section(data, "palm_contact", "palm"), PalmContactConfig, "palm_contact"),
            tool=_parse_section(_first_section(data, "tool_contact", "tool"), ToolContactConfig, "tool_contact"),
        )


ConfigT = TypeVar("ConfigT", FingerContactConfig, PalmContactConfig, ToolContactConfig)


def _first_section(data: dict[str, Any], legacy_key: str, alias_key: str) -> Any:
    if legacy_key in data:
        return data[legacy_key]
    return data.get(alias_key)


def _parse_section(data: Any, cls: type[ConfigT], section_name: str) -> ConfigT:
    if data is None:
        values: dict[str, Any] = {}
    elif isinstance(data, dict):
        values = dict(data)
    else:
        raise ValueError(f"{section_name}={data!r} must be a mapping")

    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        key = unknown[0]
        raise ValueError(f"Unknown geometry config field {section_name}.{key}={values[key]!r}")

    if "fingertip_pad_friction" in values:
        values["fingertip_pad_friction"] = _friction_tuple(
            values["fingertip_pad_friction"], f"{section_name}.fingertip_pad_friction"
        )
    if "pad_friction" in values:
        values["pad_friction"] = _friction_tuple(values["pad_friction"], f"{section_name}.pad_friction")
    if "friction" in values and values["friction"] is not None:
        values["friction"] = _friction_tuple(values["friction"], f"{section_name}.friction")
    if "fingertip_pad_thickness" in values:
        values["fingertip_pad_thickness"] = _float_value(
            values["fingertip_pad_thickness"], f"{section_name}.fingertip_pad_thickness"
        )
    if "pad_resolution" in values:
        values["pad_resolution"] = _int_value(values["pad_resolution"], f"{section_name}.pad_resolution")
    if "convex_patch_resolution" in values:
        values["convex_patch_resolution"] = _int_value(
            values["convex_patch_resolution"], f"{section_name}.convex_patch_resolution"
        )
    if "mesh_collider_resolution" in values:
        values["mesh_collider_resolution"] = _int_value(
            values["mesh_collider_resolution"], f"{section_name}.mesh_collider_resolution"
        )
    if "max_num_pad_geoms" in values:
        values["max_num_pad_geoms"] = _int_value(values["max_num_pad_geoms"], f"{section_name}.max_num_pad_geoms")
    if "max_num_mesh_colliders" in values:
        values["max_num_mesh_colliders"] = _int_value(
            values["max_num_mesh_colliders"], f"{section_name}.max_num_mesh_colliders"
        )
    for field_name in (
        "convex_patch_base_thickness",
        "convex_patch_min_height",
        "convex_patch_margin_ratio",
        "mesh_collider_thickness",
        "mesh_collider_margin_ratio",
    ):
        if field_name in values:
            values[field_name] = _float_value(values[field_name], f"{section_name}.{field_name}")
    if "convex_patch_max_height" in values and values["convex_patch_max_height"] is not None:
        values["convex_patch_max_height"] = _float_value(
            values["convex_patch_max_height"], f"{section_name}.convex_patch_max_height"
        )
    if "collision_margin" in values:
        values["collision_margin"] = _float_value(values["collision_margin"], f"{section_name}.collision_margin")

    config = cls(**values)
    _validate_config(config, section_name)
    return config


def _friction_tuple(value: Any, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name}={value!r} must contain exactly three numbers")
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name}={value!r} must contain exactly three numbers") from exc


def _float_value(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name}={value!r} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name}={value!r} must be numeric") from exc


def _int_value(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name}={value!r} must be an integer")
    try:
        ivalue = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name}={value!r} must be an integer") from exc
    if ivalue != value and not (isinstance(value, float) and value.is_integer()):
        raise ValueError(f"{field_name}={value!r} must be an integer")
    return ivalue


def _validate_config(config: FingerContactConfig | PalmContactConfig | ToolContactConfig, section_name: str) -> None:
    if isinstance(config, FingerContactConfig):
        _validate_choice(
            f"{section_name}.mode",
            config.mode,
            {"capsule", "capsule_tip_pad", "local_convex_patches"},
        )
        _validate_choice(
            f"{section_name}.fingertip_pad_shape",
            config.fingertip_pad_shape,
            {"box", "ellipsoid", "capsule", "convex_mesh"},
        )
        if config.fingertip_pad_thickness <= 0:
            raise ValueError(
                f"{section_name}.fingertip_pad_thickness={config.fingertip_pad_thickness!r} must be > 0"
            )
    elif isinstance(config, PalmContactConfig):
        _validate_choice(
            f"{section_name}.mode",
            config.mode,
            {"box_pads", "pad_grid", "convex_patches", "tiled_mesh_colliders"},
        )
        if config.pad_resolution < 1:
            raise ValueError(f"{section_name}.pad_resolution={config.pad_resolution!r} must be >= 1")
        if config.max_num_pad_geoms < 1:
            raise ValueError(f"{section_name}.max_num_pad_geoms={config.max_num_pad_geoms!r} must be >= 1")
        if config.convex_patch_resolution < 2:
            raise ValueError(
                f"{section_name}.convex_patch_resolution={config.convex_patch_resolution!r} must be >= 2"
            )
        if (
            config.mode == "convex_patches"
            and config.convex_patch_resolution * config.convex_patch_resolution > config.max_num_pad_geoms
        ):
            raise ValueError(
                "palm convex_patches requires convex_patch_resolution^2 <= max_num_pad_geoms; "
                f"got convex_patch_resolution={config.convex_patch_resolution!r}, "
                f"max_num_pad_geoms={config.max_num_pad_geoms!r}"
            )
        if config.convex_patch_base_thickness <= 0:
            raise ValueError(
                f"{section_name}.convex_patch_base_thickness="
                f"{config.convex_patch_base_thickness!r} must be > 0"
            )
        if config.convex_patch_min_height < 0:
            raise ValueError(
                f"{section_name}.convex_patch_min_height={config.convex_patch_min_height!r} must be >= 0"
            )
        if not 0 <= config.convex_patch_margin_ratio < 0.5:
            raise ValueError(
                f"{section_name}.convex_patch_margin_ratio="
                f"{config.convex_patch_margin_ratio!r} must be >= 0 and < 0.5"
            )
        if config.convex_patch_max_height is not None and config.convex_patch_max_height <= 0:
            raise ValueError(
                f"{section_name}.convex_patch_max_height={config.convex_patch_max_height!r} must be > 0"
            )
        _validate_choice(
            f"{section_name}.mesh_collider_type",
            config.mesh_collider_type,
            {"quad_frustum", "triangular_prism"},
        )
        if config.mesh_collider_resolution < 2:
            raise ValueError(
                f"{section_name}.mesh_collider_resolution="
                f"{config.mesh_collider_resolution!r} must be >= 2"
            )
        if config.mesh_collider_thickness <= 0:
            raise ValueError(
                f"{section_name}.mesh_collider_thickness="
                f"{config.mesh_collider_thickness!r} must be > 0"
            )
        if not 0 <= config.mesh_collider_margin_ratio < 0.5:
            raise ValueError(
                f"{section_name}.mesh_collider_margin_ratio="
                f"{config.mesh_collider_margin_ratio!r} must be >= 0 and < 0.5"
            )
        if config.max_num_mesh_colliders <= 0:
            raise ValueError(
                f"{section_name}.max_num_mesh_colliders="
                f"{config.max_num_mesh_colliders!r} must be > 0"
            )
        collider_count = config.mesh_collider_resolution**2
        if config.mesh_collider_type == "triangular_prism":
            collider_count *= 2
        if config.mode == "tiled_mesh_colliders" and collider_count > config.max_num_mesh_colliders:
            multiplier = "2 * " if config.mesh_collider_type == "triangular_prism" else ""
            raise ValueError(
                "palm tiled_mesh_colliders requires "
                f"{multiplier}mesh_collider_resolution^2 <= max_num_mesh_colliders; "
                f"got mesh_collider_type={config.mesh_collider_type!r}, "
                f"mesh_collider_resolution={config.mesh_collider_resolution!r}, "
                f"max_num_mesh_colliders={config.max_num_mesh_colliders!r}"
            )
        if not isinstance(config.mesh_collider_export, bool):
            raise ValueError(
                f"{section_name}.mesh_collider_export="
                f"{config.mesh_collider_export!r} must be a boolean"
            )
    elif isinstance(config, ToolContactConfig):
        _validate_choice(f"{section_name}.mode", config.mode, {"primitive", "hybrid", "convex_mesh"})
        if config.collision_margin < 0:
            raise ValueError(f"{section_name}.collision_margin={config.collision_margin!r} must be >= 0")


def _validate_choice(field_name: str, value: str, allowed: set[str]) -> None:
    if value not in allowed:
        expected = ", ".join(sorted(allowed))
        raise ValueError(f"{field_name}={value!r} must be one of: {expected}")
