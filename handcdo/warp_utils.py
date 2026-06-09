from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any
import xml.etree.ElementTree as ET

from handcdo.utils import ensure_dir


@dataclass(frozen=True)
class WarpAvailability:
    available: bool
    reason: str | None
    package: str | None
    version: str | None
    device_count: int | None = None
    device_names: list[str] | None = None
    cuda_available: bool | None = None


@dataclass(frozen=True)
class WarpBatchCapabilities:
    can_put_model: bool
    can_put_data: bool
    can_make_data: bool
    can_step: bool

    accepted_data_allocation_kwargs: list[str]
    data_allocation_probe_error: str | None

    can_set_per_world_qpos: bool
    can_set_per_world_qvel: bool
    can_set_per_world_ctrl: bool
    can_set_per_world_xfrc: bool

    true_fixed_grasp_batching_reason: str

    @property
    def supports_true_fixed_grasp_batching(self) -> bool:
        return (
            self.can_put_model
            and (self.can_put_data or self.can_make_data)
            and self.can_step
            and self.can_set_per_world_qpos
            and self.can_set_per_world_qvel
            and self.can_set_per_world_ctrl
            and self.can_set_per_world_xfrc
        )


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_warp_available() -> WarpAvailability:
    try:
        mjw = importlib.import_module("mujoco_warp")
    except Exception as exc:
        return WarpAvailability(
            available=False,
            reason=f"{type(exc).__name__}: {exc}",
            package="mujoco_warp",
            version=None,
        )

    version = getattr(mjw, "__version__", None)
    if version is None:
        try:
            version = importlib.metadata.version("mujoco-warp")
        except importlib.metadata.PackageNotFoundError:
            version = None

    device_count: int | None = None
    device_names: list[str] | None = None
    cuda_available: bool | None = None
    reason: str | None = None
    try:
        warp = importlib.import_module("warp")
        cuda_device = None
        if hasattr(warp, "get_cuda_device_count"):
            device_count = int(warp.get_cuda_device_count())
        if hasattr(warp, "get_device"):
            cuda_device = warp.get_device("cuda")
            device_names = [str(cuda_device)]
        cuda_available = bool(device_count) if device_count is not None else cuda_device is not None
    except Exception as exc:
        reason = f"device probe failed: {type(exc).__name__}: {exc}"

    return WarpAvailability(
        available=True,
        reason=reason,
        package="mujoco_warp",
        version=version,
        device_count=device_count,
        device_names=device_names,
        cuda_available=cuda_available,
    )


def availability_payload(availability: WarpAvailability) -> dict[str, Any]:
    slurm_keys = [
        "SLURM_JOB_ID",
        "SLURM_JOB_PARTITION",
        "SLURM_CPUS_PER_TASK",
        "SLURM_GPUS",
        "CUDA_VISIBLE_DEVICES",
    ]
    return {
        "warp_available": availability.available,
        "reason": availability.reason,
        "package": availability.package,
        "version": availability.version,
        "device_count": availability.device_count,
        "device_names": availability.device_names,
        "cuda_available": availability.cuda_available,
        "python_version": sys.version,
        "platform": platform.platform(),
        "timestamp": utc_timestamp(),
        "slurm": {key: value for key in slurm_keys if (value := os.environ.get(key)) is not None},
    }


def inspect_warp_batch_capabilities(mjw: Any) -> WarpBatchCapabilities:
    """Conservative runtime capability probe for true per-world grasp batches.

    PR11-d intentionally refuses to infer unsupported MuJoCo Warp state mutation
    APIs. Batched stepping alone is insufficient for fixed-grasp scoring: each
    world must receive its own tool free-joint pose and actuator controls.
    """

    data_allocation_probe_error = (
        "not probed: inspect_warp_batch_capabilities received no concrete "
        "mj_model, mj_data, or warp_model objects for guarded allocation calls"
    )
    true_fixed_grasp_batching_reason = (
        "Refusing true fixed-grasp batching: this probe verified MuJoCo Warp "
        "module-level model/data/step symbols only, but did not verify safe "
        "per-world qpos/qvel/ctrl/xfrc mutation on a batched data object. "
        "Batched stepping alone is insufficient for fixed-grasp scoring."
    )

    return WarpBatchCapabilities(
        can_put_model=hasattr(mjw, "put_model"),
        can_put_data=hasattr(mjw, "put_data"),
        can_make_data=hasattr(mjw, "make_data"),
        can_step=hasattr(mjw, "step"),
        accepted_data_allocation_kwargs=[],
        data_allocation_probe_error=data_allocation_probe_error,
        can_set_per_world_qpos=False,
        can_set_per_world_qvel=False,
        can_set_per_world_ctrl=False,
        can_set_per_world_xfrc=False,
        true_fixed_grasp_batching_reason=true_fixed_grasp_batching_reason,
    )


def warp_batch_metadata(
    *,
    nworld: int,
    nconmax: int | None,
    naconmax: int | None,
    njmax: int,
    num_grasps: int,
    num_chunks: int,
    failure_count: int,
    seconds_total: float,
    score_semantics: str = "experimental_non_equivalent",
    sequential_fallback: bool = False,
    mjcf_rewrites: list[dict[str, Any]] | None = None,
    grasps_per_second: float | None = None,
    world_steps_per_second: float | None = None,
    capabilities: WarpBatchCapabilities | None = None,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "backend": "mujoco_warp",
        "experimental": True,
        "score_semantics": score_semantics,
        "nworld": nworld,
        "nconmax": nconmax,
        "naconmax": naconmax,
        "njmax": njmax,
        "num_grasps": num_grasps,
        "num_chunks": num_chunks,
        "failure_count": failure_count,
        "sequential_fallback": sequential_fallback,
        "seconds_total": float(seconds_total),
        "grasps_per_second": grasps_per_second,
        "world_steps_per_second": world_steps_per_second,
        "mjcf_rewrites": mjcf_rewrites or [],
    }
    if capabilities is not None:
        payload["warp_capabilities"] = {
            "can_put_model": capabilities.can_put_model,
            "can_put_data": capabilities.can_put_data,
            "can_make_data": capabilities.can_make_data,
            "can_step": capabilities.can_step,
            "accepted_data_allocation_kwargs": capabilities.accepted_data_allocation_kwargs,
            "data_allocation_probe_error": capabilities.data_allocation_probe_error,
            "can_set_per_world_qpos": capabilities.can_set_per_world_qpos,
            "can_set_per_world_qvel": capabilities.can_set_per_world_qvel,
            "can_set_per_world_ctrl": capabilities.can_set_per_world_ctrl,
            "can_set_per_world_xfrc": capabilities.can_set_per_world_xfrc,
            "supports_true_fixed_grasp_batching": capabilities.supports_true_fixed_grasp_batching,
            "true_fixed_grasp_batching_reason": capabilities.true_fixed_grasp_batching_reason,
        }
    if failure_reason is not None:
        payload["failure_reason"] = failure_reason
    return payload


def _first_float(text: str) -> float | None:
    parts = text.split()
    if not parts:
        return None
    try:
        return float(parts[0])
    except ValueError:
        return None


def _rewrite_nonzero_margins_for_warp(root: ET.Element) -> list[dict[str, Any]]:
    rewrites: list[dict[str, Any]] = []
    reason = "benchmark-local MuJoCo Warp MULTICCD compatibility"

    for geom in root.findall(".//geom"):
        old = geom.get("margin")
        if old is None:
            continue
        value = _first_float(old)
        if value is None or value == 0.0:
            continue
        geom.set("margin", "0")
        rewrites.append(
            {
                "field": "geom.margin",
                "element": "geom",
                "name": geom.get("name"),
                "old": old,
                "new": "0",
                "reason": reason,
            }
        )

    for pair in root.findall(".//pair"):
        old = pair.get("margin")
        if old is None:
            continue
        value = _first_float(old)
        if value is None or value == 0.0:
            continue
        pair.set("margin", "0")
        rewrites.append(
            {
                "field": "pair.margin",
                "element": "pair",
                "name": pair.get("name"),
                "geom1": pair.get("geom1"),
                "geom2": pair.get("geom2"),
                "old": old,
                "new": "0",
                "reason": reason,
            }
        )

    return rewrites


def prepare_warp_compatible_mjcf(
    original_mjcf_path: str | Path,
    output_path: str | Path,
    allow_rewrite: bool = True,
) -> dict[str, Any]:
    original_mjcf_path = Path(original_mjcf_path)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    tree = ET.parse(original_mjcf_path)
    root = tree.getroot()
    rewrites: list[dict[str, Any]] = []
    if allow_rewrite:
        option = root.find("option")
        if option is not None and option.get("integrator") == "implicitfast":
            option.set("integrator", "Euler")
            rewrites.append(
                {
                    "field": "option.integrator",
                    "old": "implicitfast",
                    "new": "Euler",
                    "reason": "benchmark-local MuJoCo Warp compatibility",
                }
            )
        rewrites.extend(_rewrite_nonzero_margins_for_warp(root))
    if rewrites:
        tree.write(output_path, encoding="unicode", xml_declaration=False)
    else:
        shutil.copyfile(original_mjcf_path, output_path)
    original_text = original_mjcf_path.read_text(encoding="utf-8")
    warp_text = output_path.read_text(encoding="utf-8")
    return {
        "warp_mjcf_path": str(output_path),
        "mjcf_rewrites": rewrites,
        "mjcf_files_differ": original_text != warp_text,
    }


def make_warp_data(
    mjw: Any,
    warp_model: Any,
    mj_model: Any,
    mj_data: Any,
    nworld: int,
    nconmax: int | None,
    naconmax: int | None,
    njmax: int,
) -> Any:
    kwargs = {"nworld": nworld, "njmax": njmax}
    if nconmax is not None:
        kwargs["nconmax"] = nconmax
    if naconmax is not None:
        kwargs["naconmax"] = naconmax
    if hasattr(mjw, "put_data"):
        for args in ((mj_model, mj_data), (warp_model, mj_data), (warp_model,)):
            try:
                return mjw.put_data(*args, **kwargs)
            except TypeError:
                continue
        return mjw.put_data(warp_model)
    if hasattr(mjw, "make_data"):
        try:
            return mjw.make_data(warp_model, **kwargs)
        except TypeError:
            return mjw.make_data(warp_model, nworld=nworld)
    raise AttributeError("mujoco_warp has neither make_data nor put_data")


def synchronize_warp() -> tuple[bool, str | None]:
    try:
        warp = importlib.import_module("warp")
        if hasattr(warp, "synchronize"):
            warp.synchronize()
            return True, None
        return False, "warp.synchronize is unavailable"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
