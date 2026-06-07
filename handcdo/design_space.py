from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from .utils import read_yaml, stable_design_id, write_json


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    kind: str
    bounds: tuple[float, float] | tuple[int, int] | None = None
    choices: tuple[Any, ...] | None = None

    def sample(self, rng: np.random.Generator) -> Any:
        if self.kind == "categorical":
            assert self.choices is not None
            return self.choices[int(rng.integers(0, len(self.choices)))]
        if self.kind == "int":
            assert self.bounds is not None
            lo, hi = int(self.bounds[0]), int(self.bounds[1])
            return int(rng.integers(lo, hi + 1))
        if self.kind == "float":
            assert self.bounds is not None
            lo, hi = float(self.bounds[0]), float(self.bounds[1])
            return float(rng.uniform(lo, hi))
        raise ValueError(f"Unknown parameter kind {self.kind!r}")

    def validate(self, value: Any) -> Any:
        if self.kind == "categorical":
            if value not in (self.choices or ()):
                raise ValueError(f"{self.name}={value!r} not in {self.choices}")
            return value
        if self.kind == "int":
            ivalue = int(value)
            lo, hi = self.bounds or (0, 0)
            if ivalue < int(lo) or ivalue > int(hi):
                raise ValueError(f"{self.name}={ivalue} outside [{lo}, {hi}]")
            return ivalue
        if self.kind == "float":
            fvalue = float(value)
            lo, hi = self.bounds or (0.0, 0.0)
            if fvalue < float(lo) or fvalue > float(hi):
                raise ValueError(f"{self.name}={fvalue} outside [{lo}, {hi}]")
            return fvalue
        raise ValueError(f"Unknown parameter kind {self.kind!r}")


DEFAULT_SPECS: tuple[ParameterSpec, ...] = (
    ParameterSpec("finger_number", "categorical", choices=(2, 3)),
    ParameterSpec("finger_code", "categorical", choices=("1-1-1", "0-121")),
    ParameterSpec("thumb_code", "categorical", choices=("1-22", "0-22")),
    ParameterSpec("finger_angle_1", "float", bounds=(-0.55, 0.55)),
    ParameterSpec("finger_angle_2", "float", bounds=(-0.55, 0.55)),
    ParameterSpec("finger_angle_3", "float", bounds=(-0.55, 0.55)),
    ParameterSpec("finger_normal_offset_1", "float", bounds=(-0.025, 0.04)),
    ParameterSpec("finger_normal_offset_2", "float", bounds=(-0.025, 0.04)),
    ParameterSpec("finger_normal_offset_3", "float", bounds=(-0.025, 0.04)),
    ParameterSpec("finger_side_offset_1", "float", bounds=(-0.045, 0.045)),
    ParameterSpec("finger_side_offset_2", "float", bounds=(-0.045, 0.045)),
    ParameterSpec("finger_side_offset_3", "float", bounds=(-0.045, 0.045)),
    ParameterSpec("thumb_angle", "float", bounds=(-1.1, 1.1)),
    ParameterSpec("thumb_normal_offset", "float", bounds=(-0.025, 0.04)),
    ParameterSpec("thumb_side_offset", "float", bounds=(-0.06, 0.06)),
    ParameterSpec("palm_kernel_max_height", "float", bounds=(0.0, 0.035)),
    ParameterSpec("palm_kernel_spread_1", "float", bounds=(0.015, 0.08)),
    ParameterSpec("palm_kernel_spread_2", "float", bounds=(0.015, 0.08)),
    ParameterSpec("palm_kernel_center_angle_1", "float", bounds=(-1.57, 1.57)),
    ParameterSpec("palm_kernel_center_angle_2", "float", bounds=(-1.57, 1.57)),
    ParameterSpec("palm_kernel_center_offset_1", "float", bounds=(-0.04, 0.04)),
    ParameterSpec("palm_kernel_center_offset_2", "float", bounds=(-0.04, 0.04)),
    ParameterSpec("palm_kernel_intensity_ratio_1", "float", bounds=(0.2, 1.0)),
    ParameterSpec("palm_kernel_intensity_ratio_2", "float", bounds=(0.2, 1.0)),
    ParameterSpec("fingertip_scale_y", "float", bounds=(0.7, 1.6)),
    ParameterSpec("fingertip_scale_z", "float", bounds=(0.7, 1.6)),
    ParameterSpec("added_link_length_1", "float", bounds=(-0.01, 0.035)),
    ParameterSpec("added_link_length_2", "float", bounds=(-0.01, 0.035)),
    ParameterSpec("added_link_length_3", "float", bounds=(-0.01, 0.035)),
    ParameterSpec("added_link_length_4", "float", bounds=(-0.01, 0.035)),
)


class DesignSpace:
    def __init__(self, specs: tuple[ParameterSpec, ...] = DEFAULT_SPECS):
        self.specs = specs
        self.by_name = {s.name: s for s in specs}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DesignSpace":
        data = read_yaml(path)
        specs = []
        for name, item in data.get("parameters", data).items():
            choices = tuple(item["choices"]) if "choices" in item else None
            bounds = tuple(item["bounds"]) if "bounds" in item else None
            specs.append(ParameterSpec(name=name, kind=item["type"], choices=choices, bounds=bounds))
        return cls(tuple(specs))

    def sample(self, seed: int | None = None, rng: np.random.Generator | None = None) -> "HandDesign":
        rng = rng or np.random.default_rng(seed)
        return HandDesign({spec.name: spec.sample(rng) for spec in self.specs}, space=self)

    def validate(self, params: dict[str, Any]) -> dict[str, Any]:
        missing = [s.name for s in self.specs if s.name not in params]
        if missing:
            raise ValueError(f"Missing design parameters: {missing}")
        return {s.name: s.validate(params[s.name]) for s in self.specs}

    def optuna_suggest(self, trial: Any) -> "HandDesign":
        values: dict[str, Any] = {}
        for spec in self.specs:
            if spec.kind == "categorical":
                values[spec.name] = trial.suggest_categorical(spec.name, list(spec.choices or ()))
            elif spec.kind == "int":
                lo, hi = spec.bounds or (0, 0)
                values[spec.name] = trial.suggest_int(spec.name, int(lo), int(hi))
            else:
                lo, hi = spec.bounds or (0.0, 0.0)
                values[spec.name] = trial.suggest_float(spec.name, float(lo), float(hi))
        return HandDesign(values, space=self)


@dataclass(frozen=True)
class HandDesign:
    params: dict[str, Any]
    space: DesignSpace | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        space = self.space or DesignSpace()
        object.__setattr__(self, "params", space.validate(dict(self.params)))

    @property
    def design_id(self) -> str:
        return stable_design_id(self.params)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.params)

    def to_json(self, path: str | Path) -> None:
        write_json(path, {"design_id": self.design_id, "parameters": self.to_dict()})

    @classmethod
    def from_dict(cls, payload: dict[str, Any], space: DesignSpace | None = None) -> "HandDesign":
        return cls(payload.get("parameters", payload), space=space)

    @classmethod
    def from_json(cls, path: str | Path, space: DesignSpace | None = None) -> "HandDesign":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")), space=space)
