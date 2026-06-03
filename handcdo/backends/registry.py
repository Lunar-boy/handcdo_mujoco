from __future__ import annotations

from .base import SimulatorBackend
from .mujoco_cpu import MujocoCpuBackend

_BACKEND_ALIASES = {
    "mujoco": "mujoco_cpu",
    "mujoco_cpu": "mujoco_cpu",
}


def get_backend(name: str) -> SimulatorBackend:
    key = name.strip().lower()
    normalized = _BACKEND_ALIASES.get(key)
    if normalized == "mujoco_cpu":
        return MujocoCpuBackend()
    valid = ", ".join(sorted(_BACKEND_ALIASES))
    raise ValueError(f"Unknown simulator backend {name!r}. Expected one of: {valid}")
