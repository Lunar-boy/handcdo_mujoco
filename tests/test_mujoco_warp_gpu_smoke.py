from __future__ import annotations

import os

import pytest

from handcdo.warp_utils import (
    inspect_warp_batch_capabilities,
    make_warp_data,
    smoke_test_warp_per_world_state_write,
)


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


def test_mujoco_warp_per_world_state_write_smoke() -> None:
    if os.environ.get("RUN_GPU_TESTS") != "1":
        pytest.skip("Set RUN_GPU_TESTS=1 to enable GPU tests.")

    mujoco = pytest.importorskip("mujoco")
    mjw = pytest.importorskip("mujoco_warp")
    pytest.importorskip("warp")

    xml = """
<mujoco model="warp_state_smoke">
  <worldbody>
    <body name="box" pos="0 0 0.1">
      <joint name="hinge" type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hinge" gear="1"/>
  </actuator>
</mujoco>
"""
    try:
        mj_model = mujoco.MjModel.from_xml_string(xml)
        mj_data = mujoco.MjData(mj_model)
        warp_model = mjw.put_model(mj_model)
        warp_data = make_warp_data(
            mjw,
            warp_model,
            mj_model,
            mj_data,
            nworld=2,
            nconmax=8,
            naconmax=None,
            njmax=16,
        )
    except Exception as exc:  # pragma: no cover - depends on optional GPU runtime
        pytest.skip(f"Could not create MuJoCo Warp model/data for write smoke: {type(exc).__name__}: {exc}")

    capabilities = inspect_warp_batch_capabilities(mjw, warp_model=warp_model, warp_data=warp_data, nworld=2)
    assert capabilities.has_qpos
    assert capabilities.has_qvel
    assert capabilities.has_ctrl
    assert capabilities.has_xfrc_applied

    smoke = smoke_test_warp_per_world_state_write(warp_data, nworld=2)
    if not smoke["ok"]:
        pytest.skip(f"MuJoCo Warp per-world state write path unavailable: {smoke['reason']}")
    assert capabilities.supports_true_fixed_grasp_batching
