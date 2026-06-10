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
    import_available: bool = True
    has_qpos: bool = False
    has_qvel: bool = False
    has_ctrl: bool = False
    has_xfrc_applied: bool = False
    qpos_is_batched: bool = False
    qvel_is_batched: bool = False
    ctrl_is_batched: bool = False
    xfrc_is_batched: bool = False
    qpos_write_tested: bool = False
    qvel_write_tested: bool = False
    ctrl_write_tested: bool = False
    xfrc_write_tested: bool = False

    @property
    def supports_true_fixed_grasp_batching(self) -> bool:
        return (
            self.import_available
            and self.can_put_model
            and (self.can_put_data or self.can_make_data)
            and self.can_step
            and self.has_qpos
            and self.has_qvel
            and self.has_ctrl
            and self.has_xfrc_applied
            and self.qpos_is_batched
            and self.qvel_is_batched
            and self.ctrl_is_batched
            and self.xfrc_is_batched
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


def _import_optional_mujoco_warp() -> tuple[Any | None, str | None]:
    try:
        return importlib.import_module("mujoco_warp"), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(part) for part in shape)
    except Exception:
        return None


def _field_is_batched(value: Any, nworld: int | None) -> bool:
    shape = _shape_of(value)
    if shape is None or len(shape) < 2:
        return False
    if nworld is None:
        return shape[0] > 1
    return shape[0] == nworld


def _field_host_array(value: Any) -> Any | None:
    try:
        import numpy as np
    except Exception:
        return None

    for method_name in ("numpy", "to_numpy"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return np.array(method(), copy=True)
            except Exception:
                pass
    try:
        return np.array(value, copy=True)
    except Exception:
        return None


def _write_world_slice(field: Any, world_index: int, world_value: Any) -> tuple[bool, str | None]:
    try:
        field[world_index, ...] = world_value
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _restore_field(field: Any, snapshot: Any) -> tuple[bool, str | None]:
    try:
        field[...] = snapshot
        return True, None
    except Exception as exc:
        whole_error = f"{type(exc).__name__}: {exc}"
    try:
        for world_index in range(int(snapshot.shape[0])):
            field[world_index, ...] = snapshot[world_index]
        return True, None
    except Exception as exc:
        return False, f"whole-field restore failed: {whole_error}; per-world restore failed: {type(exc).__name__}: {exc}"


def _mutated_snapshot_for_world(snapshot: Any, world_index: int) -> Any | None:
    try:
        import numpy as np
    except Exception:
        return None

    mutated = np.array(snapshot, copy=True)
    try:
        world_view = mutated[world_index]
    except Exception:
        return None
    if world_view.size == 0:
        return None
    flat = world_view.reshape(-1)
    baseline = float(flat[0])
    flat[0] = baseline + 0.125 if baseline != 0.125 else baseline + 0.25
    return mutated


def _field_report(
    warp_data: Any,
    field_name: str,
    *,
    nworld: int,
    write_test: bool,
) -> dict[str, Any]:
    field = getattr(warp_data, field_name, None)
    present = field is not None
    shape = _shape_of(field) if present else None
    batched = bool(present and _field_is_batched(field, nworld))
    report: dict[str, Any] = {
        "present": present,
        "shape": list(shape) if shape is not None else None,
        "batched": batched,
        "write_tested": False,
        "reason": "",
    }
    if not present:
        report["reason"] = f"{field_name} is absent"
        return report
    if not batched:
        report["reason"] = f"{field_name} does not have leading nworld={nworld}"
        return report
    if not write_test:
        report["reason"] = f"{field_name} batched shape detected; write path not tested"
        return report

    snapshot = _field_host_array(field)
    if snapshot is None:
        report["reason"] = f"{field_name} could not be copied to host for round-trip verification"
        return report
    mutated = _mutated_snapshot_for_world(snapshot, 0)
    if mutated is None:
        report["reason"] = f"{field_name} has no writable scalar entries to verify"
        return report

    synchronize_warp()
    wrote, write_error = _write_world_slice(field, 0, mutated[0])
    synchronize_warp()
    if not wrote:
        report["reason"] = f"{field_name} per-world assignment failed: {write_error}"
        return report

    observed = _field_host_array(field)
    restored, restore_error = _restore_field(field, snapshot)
    synchronize_warp()
    if not restored:
        report["reason"] = f"{field_name} write verified status unknown; restore failed: {restore_error}"
        return report
    if observed is None:
        report["reason"] = f"{field_name} write could not be verified by host round trip"
        return report

    try:
        import numpy as np

        restored_snapshot = _field_host_array(field)
        write_ok = bool(
            np.allclose(observed[0], mutated[0])
            and restored_snapshot is not None
            and np.allclose(restored_snapshot, snapshot)
        )
    except Exception as exc:
        report["reason"] = f"{field_name} write verification failed: {type(exc).__name__}: {exc}"
        return report
    if not write_ok:
        report["reason"] = f"{field_name} per-world write did not round-trip cleanly"
        return report

    report["write_tested"] = True
    report["reason"] = "per-world write round-trip verified and original values restored"
    return report


def smoke_test_warp_per_world_state_write(
    warp_data: Any,
    *,
    nworld: int,
    require_fields: tuple[str, ...] = ("qpos", "qvel", "ctrl", "xfrc_applied"),
) -> dict[str, Any]:
    fields = {
        field_name: _field_report(warp_data, field_name, nworld=nworld, write_test=True)
        for field_name in require_fields
    }
    ok = all(field["present"] and field["batched"] and field["write_tested"] for field in fields.values())
    missing = [name for name, field in fields.items() if not field["present"]]
    unbatched = [name for name, field in fields.items() if field["present"] and not field["batched"]]
    unverified = [
        name
        for name, field in fields.items()
        if field["present"] and field["batched"] and not field["write_tested"]
    ]
    if ok:
        reason = "verified per-world writes for qpos, qvel, ctrl, and xfrc_applied"
    elif missing:
        reason = f"missing fields: {', '.join(missing)}"
    elif unbatched:
        reason = f"fields are not batched with leading nworld={nworld}: {', '.join(unbatched)}"
    else:
        reason = f"fields lack verified per-world write support: {', '.join(unverified)}"
    return {"ok": ok, "fields": fields, "reason": reason}


def inspect_warp_batch_capabilities(
    mjw: Any | None = None,
    *,
    warp_model: Any | None = None,
    warp_data: Any | None = None,
    nworld: int | None = None,
) -> WarpBatchCapabilities:
    """Probe true per-world grasp-batch readiness without inventing APIs."""
    if mjw is None:
        mjw, import_error = _import_optional_mujoco_warp()
        if mjw is None:
            reason = f"MuJoCo Warp import unavailable: {import_error}"
            return WarpBatchCapabilities(
                can_put_model=False,
                can_put_data=False,
                can_make_data=False,
                can_step=False,
                accepted_data_allocation_kwargs=[],
                data_allocation_probe_error=reason,
                can_set_per_world_qpos=False,
                can_set_per_world_qvel=False,
                can_set_per_world_ctrl=False,
                can_set_per_world_xfrc=False,
                true_fixed_grasp_batching_reason=reason,
                import_available=False,
            )

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
    field_reports: dict[str, dict[str, Any]] = {}
    if warp_data is not None:
        inferred_nworld = nworld
        if inferred_nworld is None:
            for field_name in ("qpos", "qvel", "ctrl", "xfrc_applied"):
                shape = _shape_of(getattr(warp_data, field_name, None))
                if shape is not None and len(shape) >= 2:
                    inferred_nworld = shape[0]
                    break
        if inferred_nworld is None:
            inferred_nworld = 0
        smoke = smoke_test_warp_per_world_state_write(warp_data, nworld=inferred_nworld)
        field_reports = smoke["fields"]
        true_fixed_grasp_batching_reason = smoke["reason"]
    elif warp_model is not None:
        data_allocation_probe_error = (
            "not probed: warp_model was provided without mj_model/mj_data; "
            "safe data allocation signatures remain runtime-dependent"
        )

    return WarpBatchCapabilities(
        can_put_model=hasattr(mjw, "put_model"),
        can_put_data=hasattr(mjw, "put_data"),
        can_make_data=hasattr(mjw, "make_data"),
        can_step=hasattr(mjw, "step"),
        accepted_data_allocation_kwargs=[],
        data_allocation_probe_error=data_allocation_probe_error,
        true_fixed_grasp_batching_reason=true_fixed_grasp_batching_reason,
        has_qpos=field_reports.get("qpos", {}).get("present", False),
        has_qvel=field_reports.get("qvel", {}).get("present", False),
        has_ctrl=field_reports.get("ctrl", {}).get("present", False),
        has_xfrc_applied=field_reports.get("xfrc_applied", {}).get("present", False),
        qpos_is_batched=field_reports.get("qpos", {}).get("batched", False),
        qvel_is_batched=field_reports.get("qvel", {}).get("batched", False),
        ctrl_is_batched=field_reports.get("ctrl", {}).get("batched", False),
        xfrc_is_batched=field_reports.get("xfrc_applied", {}).get("batched", False),
        qpos_write_tested=field_reports.get("qpos", {}).get("write_tested", False),
        qvel_write_tested=field_reports.get("qvel", {}).get("write_tested", False),
        ctrl_write_tested=field_reports.get("ctrl", {}).get("write_tested", False),
        xfrc_write_tested=field_reports.get("xfrc_applied", {}).get("write_tested", False),
        can_set_per_world_qpos=field_reports.get("qpos", {}).get("write_tested", False),
        can_set_per_world_qvel=field_reports.get("qvel", {}).get("write_tested", False),
        can_set_per_world_ctrl=field_reports.get("ctrl", {}).get("write_tested", False),
        can_set_per_world_xfrc=field_reports.get("xfrc_applied", {}).get("write_tested", False),
    )


def warp_capabilities_payload(capabilities: WarpBatchCapabilities) -> dict[str, Any]:
    return {
        "can_put_model": capabilities.can_put_model,
        "can_put_data": capabilities.can_put_data,
        "can_make_data": capabilities.can_make_data,
        "can_step": capabilities.can_step,
        "accepted_data_allocation_kwargs": capabilities.accepted_data_allocation_kwargs,
        "data_allocation_probe_error": capabilities.data_allocation_probe_error,
        "import_available": capabilities.import_available,
        "has_qpos": capabilities.has_qpos,
        "has_qvel": capabilities.has_qvel,
        "has_ctrl": capabilities.has_ctrl,
        "has_xfrc_applied": capabilities.has_xfrc_applied,
        "qpos_is_batched": capabilities.qpos_is_batched,
        "qvel_is_batched": capabilities.qvel_is_batched,
        "ctrl_is_batched": capabilities.ctrl_is_batched,
        "xfrc_is_batched": capabilities.xfrc_is_batched,
        "qpos_write_tested": capabilities.qpos_write_tested,
        "qvel_write_tested": capabilities.qvel_write_tested,
        "ctrl_write_tested": capabilities.ctrl_write_tested,
        "xfrc_write_tested": capabilities.xfrc_write_tested,
        "can_set_per_world_qpos": capabilities.can_set_per_world_qpos,
        "can_set_per_world_qvel": capabilities.can_set_per_world_qvel,
        "can_set_per_world_ctrl": capabilities.can_set_per_world_ctrl,
        "can_set_per_world_xfrc": capabilities.can_set_per_world_xfrc,
        "supports_true_fixed_grasp_batching": capabilities.supports_true_fixed_grasp_batching,
        "true_fixed_grasp_batching_reason": capabilities.true_fixed_grasp_batching_reason,
    }


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
        payload["warp_capabilities"] = warp_capabilities_payload(capabilities)
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
