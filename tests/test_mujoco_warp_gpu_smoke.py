from __future__ import annotations

import os

import pytest


pytestmark = pytest.mark.gpu


def test_mujoco_warp_gpu_import_and_device() -> None:
    if os.environ.get("RUN_GPU_TESTS") != "1":
        pytest.skip("Set RUN_GPU_TESTS=1 to enable GPU tests.")

    pytest.importorskip("mujoco")
    pytest.importorskip("mujoco_warp")
    warp = pytest.importorskip("warp")

    try:
        device = warp.get_device("cuda")
    except TypeError:
        device = warp.get_device()
    except Exception as exc:  # pragma: no cover - depends on GPU runtime
        pytest.fail(f"Could not get CUDA device through warp: {type(exc).__name__}: {exc}")

    assert device is not None
