from __future__ import annotations

import importlib
import sys

import pytest

from handcdo.backends.batched import supports_batched_grasps
from handcdo.warp_utils import WarpAvailability


def test_importing_backends_does_not_import_mujoco_warp_package():
    for module_name in (
        "handcdo.backends",
        "handcdo.backends.registry",
        "handcdo.backends.mujoco_warp",
        "mujoco_warp",
    ):
        sys.modules.pop(module_name, None)

    backends = importlib.import_module("handcdo.backends")

    assert backends.get_backend("mujoco").name == "mujoco_cpu"
    assert "mujoco_warp" not in sys.modules
    assert "handcdo.backends.mujoco_warp" not in sys.modules


def test_get_backend_cpu_aliases_do_not_require_mujoco_warp():
    from handcdo.backends import get_backend

    assert get_backend("mujoco").name == "mujoco_cpu"
    assert get_backend("mujoco_cpu").name == "mujoco_cpu"


def test_get_backend_mujoco_warp_returns_backend_when_dependency_available(monkeypatch):
    from handcdo import warp_utils
    from handcdo.backends import get_backend

    monkeypatch.setattr(
        warp_utils,
        "check_warp_available",
        lambda: WarpAvailability(True, None, "mujoco_warp", "test"),
    )

    backend = get_backend("mujoco_warp")

    assert backend.name == "mujoco_warp"


def test_get_backend_mujoco_warp_fails_clearly_when_dependency_absent(monkeypatch):
    from handcdo import warp_utils
    from handcdo.backends import get_backend
    from handcdo.backends.mujoco_warp import MujocoWarpUnavailableError

    monkeypatch.setattr(
        warp_utils,
        "check_warp_available",
        lambda: WarpAvailability(False, "ModuleNotFoundError: No module named 'mujoco_warp'", "mujoco_warp", None),
    )

    with pytest.raises(MujocoWarpUnavailableError) as exc_info:
        get_backend("mujoco_warp")

    message = str(exc_info.value)
    assert "MuJoCo Warp backend requires the optional warp extra" in message
    assert 'python3 -m pip install -e ".[warp]"' in message


def test_missing_mujoco_warp_does_not_break_cpu_backend_construction(monkeypatch):
    from handcdo import warp_utils
    from handcdo.backends import get_backend

    monkeypatch.setattr(
        warp_utils,
        "check_warp_available",
        lambda: WarpAvailability(False, "missing for test", "mujoco_warp", None),
    )

    assert get_backend("mujoco").name == "mujoco_cpu"
    assert get_backend("mujoco_cpu").name == "mujoco_cpu"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"nworld": 0}, "nworld"),
        ({"nconmax": 0}, "nconmax"),
        ({"naconmax": 0}, "naconmax"),
        ({"njmax": 0}, "njmax"),
        ({"warmup_steps": -1}, "warmup_steps"),
        ({"capture_graph": "yes"}, "capture_graph"),
        ({"allow_sequential_fallback": "no"}, "allow_sequential_fallback"),
    ],
)
def test_invalid_constructor_values_are_rejected_before_optional_import(monkeypatch, kwargs, match):
    from handcdo import warp_utils
    from handcdo.backends.mujoco_warp import MujocoWarpBackend

    def fail_if_called():
        raise AssertionError("optional Warp availability should not be checked for invalid constructor values")

    monkeypatch.setattr(warp_utils, "check_warp_available", fail_if_called)

    with pytest.raises((TypeError, ValueError), match=match):
        MujocoWarpBackend(**kwargs)


def test_supports_batched_grasps_true_for_mujoco_warp_skeleton_when_available(monkeypatch):
    from handcdo import warp_utils
    from handcdo.backends.mujoco_warp import MujocoWarpBackend

    monkeypatch.setattr(
        warp_utils,
        "check_warp_available",
        lambda: WarpAvailability(True, None, "mujoco_warp", "test"),
    )

    backend = MujocoWarpBackend()

    assert callable(getattr(backend, "evaluate_grasps_batch", None))
    assert supports_batched_grasps(backend) is True


def test_mujoco_warp_skeleton_evaluation_methods_are_not_implemented(monkeypatch):
    from handcdo import warp_utils
    from handcdo.backends.mujoco_warp import MujocoWarpBackend

    monkeypatch.setattr(
        warp_utils,
        "check_warp_available",
        lambda: WarpAvailability(True, None, "mujoco_warp", "test"),
    )

    backend = MujocoWarpBackend()

    with pytest.raises(NotImplementedError, match="not implemented in PR11-b"):
        backend.evaluate_grasp(None, "hammer", None, None)
    with pytest.raises(NotImplementedError, match="not implemented in PR11-b"):
        backend.evaluate_grasps_batch(None, "hammer", [], None)
