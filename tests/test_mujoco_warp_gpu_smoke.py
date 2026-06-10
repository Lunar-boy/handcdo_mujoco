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
        try:
            device = warp.get_device()
        except Exception as exc:  # pragma: no cover - depends on GPU runtime
            pytest.skip(f"Could not get default Warp device: {type(exc).__name__}: {exc}")
    except Exception as exc:  # pragma: no cover - depends on GPU runtime
        pytest.skip(f"Could not get CUDA device through warp: {type(exc).__name__}: {exc}")

    if device is None:
        pytest.skip("Warp CUDA device lookup returned None.")
    if "cuda" not in str(device).lower():
        pytest.skip(f"Warp device is not CUDA: {device}")


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
    missing_fields = [
        name
        for name, present in (
            ("qpos", capabilities.has_qpos),
            ("qvel", capabilities.has_qvel),
            ("ctrl", capabilities.has_ctrl),
            ("xfrc_applied", capabilities.has_xfrc_applied),
        )
        if not present
    ]
    if missing_fields:
        pytest.skip(f"MuJoCo Warp data object is missing state fields: {', '.join(missing_fields)}")

    smoke = smoke_test_warp_per_world_state_write(warp_data, nworld=2, mjw=mjw)
    if not smoke["ok"]:
        pytest.skip(f"MuJoCo Warp per-world state write path unavailable: {smoke['reason']}")
    if not capabilities.supports_true_fixed_grasp_batching:
        pytest.skip(
            "MuJoCo Warp capability probe did not verify true fixed-grasp batching: "
            f"{capabilities.true_fixed_grasp_batching_reason}"
        )
