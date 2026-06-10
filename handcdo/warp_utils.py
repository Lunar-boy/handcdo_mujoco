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
    can_forward: bool

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
    has_xpos: bool = False
    has_xmat: bool = False
    qpos_is_batched: bool = False
    qvel_is_batched: bool = False
    ctrl_is_batched: bool = False
    xfrc_is_batched: bool = False
    xpos_is_batched: bool = False
    xmat_is_batched: bool = False
    qpos_write_tested: bool = False
    qvel_write_tested: bool = False
    ctrl_write_tested: bool = False
    xfrc_write_tested: bool = False
    qpos_write_method: str | None = None
    qvel_write_method: str | None = None
    ctrl_write_method: str | None = None
    xfrc_write_method: str | None = None
    kinematics_update_method: str | None = None

    @property
    def supports_true_fixed_grasp_batching(self) -> bool:
        return (
            self.import_available
            and self.can_put_model
            and (self.can_put_data or self.can_make_data)
            and self.can_step
            and (self.can_forward or self.kinematics_update_method is not None)
            and self.has_qpos
            and self.has_qvel
            and self.has_ctrl
            and self.has_xfrc_applied
            and self.has_xpos
            and self.has_xmat
            and self.qpos_is_batched
            and self.qvel_is_batched
            and self.ctrl_is_batched
            and self.xfrc_is_batched
            and self.xpos_is_batched
            and self.xmat_is_batched
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


def _slice_world(field: Any, world_index: int) -> tuple[Any | None, str | None]:
    try:
        return field[world_index, ...], None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _try_field_native_assignment(target: Any, value: Any) -> tuple[bool, str | None, str | None]:
    errors: list[str] = []
    for method_name in ("assign", "copy_", "copy"):
        method = getattr(target, method_name, None)
        if not callable(method):
            continue
        try:
            method(value)
            return True, method_name, None
        except Exception as exc:
            errors.append(f"{method_name}: {type(exc).__name__}: {exc}")
    return False, None, "; ".join(errors) if errors else "no field-native assignment method exposed"


def _try_warp_copy_assignment(target: Any, value: Any, warp_module: Any | None) -> tuple[bool, str | None]:
    if warp_module is None:
        try:
            warp_module = importlib.import_module("warp")
        except Exception as exc:
            return False, f"warp import failed: {type(exc).__name__}: {exc}"
    if not hasattr(warp_module, "copy") or not hasattr(warp_module, "from_numpy"):
        return False, "warp.copy/from_numpy unavailable"
    try:
        import numpy as np

        array_value = np.asarray(value)
    except Exception as exc:
        return False, f"could not convert write value to numpy: {type(exc).__name__}: {exc}"
    try:
        kwargs: dict[str, Any] = {}
        dtype = getattr(target, "dtype", None)
        device = getattr(target, "device", None)
        if dtype is not None:
            kwargs["dtype"] = dtype
        if device is not None:
            kwargs["device"] = device
        source = warp_module.from_numpy(array_value, **kwargs)
    except Exception as exc:
        return False, f"warp.from_numpy failed: {type(exc).__name__}: {exc}"
    for call in (
        lambda: warp_module.copy(target, source),
        lambda: warp_module.copy(dest=target, src=source),
    ):
        try:
            call()
            return True, "warp.copy"
        except Exception:
            continue
    return False, "warp.copy failed for positional and keyword call forms"


def _try_write_field_per_world(
    field: Any,
    *,
    field_name: str,
    world_index: int,
    value: Any,
    mjw: Any | None = None,
    warp_module: Any | None = None,
) -> tuple[bool, str, str]:
    errors: list[str] = []
    try:
        field[world_index, ...] = value
        return True, "direct_setitem", "direct Python per-world assignment succeeded"
    except Exception as exc:
        errors.append(f"direct_setitem: {type(exc).__name__}: {exc}")

    target, slice_error = _slice_world(field, world_index)
    if target is None:
        errors.append(f"world_slice: {slice_error}")
    else:
        assigned, method_name, native_error = _try_field_native_assignment(target, value)
        if assigned and method_name is not None:
            return True, f"field.{method_name}", f"field-native per-world {method_name} succeeded"
        errors.append(f"field_native: {native_error}")

        copied, copy_reason = _try_warp_copy_assignment(target, value, warp_module)
        if copied:
            return True, "warp.copy", "Warp-native copy into per-world field slice succeeded"
        errors.append(f"warp_copy: {copy_reason}")

    available_state_apis = [
        name
        for name in ("get_state", "set_state", "reset_data")
        if mjw is not None and callable(getattr(mjw, name, None))
    ]
    if available_state_apis:
        errors.append(
            "mujoco_warp_state_api: available but not used by field-level probe "
            f"for {field_name}: {', '.join(available_state_apis)}"
        )
    else:
        errors.append("mujoco_warp_state_api: no guarded state API write path available")

    return (
        False,
        "none",
        "no supported direct assignment, field-native assignment, Warp copy, "
        "or MuJoCo Warp state API write path was available"
        + (f" ({'; '.join(errors)})" if errors else ""),
    )


def _restore_field(
    field: Any,
    snapshot: Any,
    *,
    field_name: str,
    mjw: Any | None = None,
    warp_module: Any | None = None,
) -> tuple[bool, str | None]:
    try:
        field[...] = snapshot
        return True, None
    except Exception as exc:
        whole_error = f"{type(exc).__name__}: {exc}"
    errors: list[str] = []
    for world_index in range(int(snapshot.shape[0])):
        restored, method, reason = _try_write_field_per_world(
            field,
            field_name=field_name,
            world_index=world_index,
            value=snapshot[world_index],
            mjw=mjw,
            warp_module=warp_module,
        )
        if not restored:
            errors.append(f"world {world_index}: {method}: {reason}")
    if not errors:
        return True, None
    return False, f"whole-field restore failed: {whole_error}; per-world restore failed: {'; '.join(errors)}"


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
    mjw: Any | None = None,
    warp_module: Any | None = None,
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
        "write_method": None,
        "write_roundtrip_verified": False,
        "restore_ok": None,
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
    wrote, write_method, write_reason = _try_write_field_per_world(
        field,
        field_name=field_name,
        world_index=0,
        value=mutated[0],
        mjw=mjw,
        warp_module=warp_module,
    )
    synchronize_warp()
    if not wrote:
        report["write_method"] = write_method
        report["restore_ok"] = None
        report["reason"] = f"{field_name} per-world assignment failed: {write_reason}"
        return report

    observed = _field_host_array(field)
    restored, restore_error = _restore_field(
        field,
        snapshot,
        field_name=field_name,
        mjw=mjw,
        warp_module=warp_module,
    )
    synchronize_warp()
    report["write_method"] = write_method
    report["restore_ok"] = restored
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
    report["write_roundtrip_verified"] = True
    report["reason"] = f"per-world write round-trip verified via {write_method} and original values restored"
    return report


def smoke_test_warp_per_world_state_write(
    warp_data: Any,
    *,
    nworld: int,
    require_fields: tuple[str, ...] = ("qpos", "qvel", "ctrl", "xfrc_applied"),
    mjw: Any | None = None,
    warp_module: Any | None = None,
) -> dict[str, Any]:
    fields = {
        field_name: _field_report(
            warp_data,
            field_name,
            nworld=nworld,
            write_test=True,
            mjw=mjw,
            warp_module=warp_module,
        )
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


def _required_batching_reasons(
    *,
    import_available: bool,
    can_put_model: bool,
    can_put_data: bool,
    can_make_data: bool,
    can_step: bool,
    can_forward: bool,
    kinematics_update_method: str | None,
    field_reports: dict[str, dict[str, Any]],
    nworld: int | None,
) -> list[str]:
    reasons: list[str] = []
    if not import_available:
        reasons.append("mujoco_warp import unavailable")
    if not can_put_model:
        reasons.append("missing required MuJoCo Warp put_model")
    if not (can_put_data or can_make_data):
        reasons.append("missing required MuJoCo Warp put_data/make_data")
    if not can_step:
        reasons.append("missing required MuJoCo Warp step")
    if not (can_forward or kinematics_update_method is not None):
        reasons.append("missing required MuJoCo Warp forward/kinematics update")
    for field_name in ("qpos", "qvel", "ctrl", "xfrc_applied"):
        report = field_reports.get(field_name, {})
        if not report.get("present", False):
            reasons.append(f"warp_data.{field_name} is absent")
        elif not report.get("batched", False):
            reasons.append(f"warp_data.{field_name} does not have leading nworld={nworld}")
        elif not report.get("write_tested", False):
            reasons.append(f"warp_data.{field_name} lacks verified per-world write support")
    for field_name in ("xpos", "xmat"):
        report = field_reports.get(field_name, {})
        if not report.get("present", False):
            reasons.append(f"warp_data.{field_name} is absent")
        elif not report.get("batched", False):
            reasons.append(f"warp_data.{field_name} does not have leading nworld={nworld}")
    return reasons


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
                can_forward=False,
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
    can_put_model = hasattr(mjw, "put_model")
    can_put_data = hasattr(mjw, "put_data")
    can_make_data = hasattr(mjw, "make_data")
    can_step = hasattr(mjw, "step")
    can_forward = callable(getattr(mjw, "forward", None)) or callable(getattr(mjw, "mj_forward", None))
    kinematics_update_method = "forward" if callable(getattr(mjw, "forward", None)) else None
    if kinematics_update_method is None and callable(getattr(mjw, "mj_forward", None)):
        kinematics_update_method = "mj_forward"
    field_reports: dict[str, dict[str, Any]] = {}
    inferred_nworld = nworld
    if warp_data is not None:
        if inferred_nworld is None:
            for field_name in ("qpos", "qvel", "ctrl", "xfrc_applied", "xpos", "xmat"):
                shape = _shape_of(getattr(warp_data, field_name, None))
                if shape is not None and len(shape) >= 2:
                    inferred_nworld = shape[0]
                    break
        if inferred_nworld is None:
            inferred_nworld = 0
        smoke = smoke_test_warp_per_world_state_write(warp_data, nworld=inferred_nworld, mjw=mjw)
        field_reports = smoke["fields"]
        for readback_field in ("xpos", "xmat"):
            field_reports[readback_field] = _field_report(
                warp_data,
                readback_field,
                nworld=inferred_nworld,
                write_test=False,
                mjw=mjw,
            )
        reasons = _required_batching_reasons(
            import_available=True,
            can_put_model=can_put_model,
            can_put_data=can_put_data,
            can_make_data=can_make_data,
            can_step=can_step,
            can_forward=can_forward,
            kinematics_update_method=kinematics_update_method,
            field_reports=field_reports,
            nworld=inferred_nworld,
        )
        true_fixed_grasp_batching_reason = "all required true fixed-grasp batching capabilities verified" if not reasons else "; ".join(reasons)
    elif warp_model is not None:
        data_allocation_probe_error = (
            "not probed: warp_model was provided without mj_model/mj_data; "
            "safe data allocation signatures remain runtime-dependent"
        )

    return WarpBatchCapabilities(
        can_put_model=can_put_model,
        can_put_data=can_put_data,
        can_make_data=can_make_data,
        can_step=can_step,
        can_forward=can_forward,
        accepted_data_allocation_kwargs=[],
        data_allocation_probe_error=data_allocation_probe_error,
        true_fixed_grasp_batching_reason=true_fixed_grasp_batching_reason,
        has_qpos=field_reports.get("qpos", {}).get("present", False),
        has_qvel=field_reports.get("qvel", {}).get("present", False),
        has_ctrl=field_reports.get("ctrl", {}).get("present", False),
        has_xfrc_applied=field_reports.get("xfrc_applied", {}).get("present", False),
        has_xpos=field_reports.get("xpos", {}).get("present", False),
        has_xmat=field_reports.get("xmat", {}).get("present", False),
        qpos_is_batched=field_reports.get("qpos", {}).get("batched", False),
        qvel_is_batched=field_reports.get("qvel", {}).get("batched", False),
        ctrl_is_batched=field_reports.get("ctrl", {}).get("batched", False),
        xfrc_is_batched=field_reports.get("xfrc_applied", {}).get("batched", False),
        xpos_is_batched=field_reports.get("xpos", {}).get("batched", False),
        xmat_is_batched=field_reports.get("xmat", {}).get("batched", False),
        qpos_write_tested=field_reports.get("qpos", {}).get("write_tested", False),
        qvel_write_tested=field_reports.get("qvel", {}).get("write_tested", False),
        ctrl_write_tested=field_reports.get("ctrl", {}).get("write_tested", False),
        xfrc_write_tested=field_reports.get("xfrc_applied", {}).get("write_tested", False),
        qpos_write_method=field_reports.get("qpos", {}).get("write_method"),
        qvel_write_method=field_reports.get("qvel", {}).get("write_method"),
        ctrl_write_method=field_reports.get("ctrl", {}).get("write_method"),
        xfrc_write_method=field_reports.get("xfrc_applied", {}).get("write_method"),
        can_set_per_world_qpos=field_reports.get("qpos", {}).get("write_tested", False),
        can_set_per_world_qvel=field_reports.get("qvel", {}).get("write_tested", False),
        can_set_per_world_ctrl=field_reports.get("ctrl", {}).get("write_tested", False),
        can_set_per_world_xfrc=field_reports.get("xfrc_applied", {}).get("write_tested", False),
        kinematics_update_method=kinematics_update_method,
    )


def warp_capabilities_payload(capabilities: WarpBatchCapabilities) -> dict[str, Any]:
    return {
        "can_put_model": capabilities.can_put_model,
        "can_put_data": capabilities.can_put_data,
        "can_make_data": capabilities.can_make_data,
        "can_step": capabilities.can_step,
        "can_forward": capabilities.can_forward,
        "kinematics_update_method": capabilities.kinematics_update_method,
        "accepted_data_allocation_kwargs": capabilities.accepted_data_allocation_kwargs,
        "data_allocation_probe_error": capabilities.data_allocation_probe_error,
        "import_available": capabilities.import_available,
        "has_qpos": capabilities.has_qpos,
        "has_qvel": capabilities.has_qvel,
        "has_ctrl": capabilities.has_ctrl,
        "has_xfrc_applied": capabilities.has_xfrc_applied,
        "has_xpos": capabilities.has_xpos,
        "has_xmat": capabilities.has_xmat,
        "qpos_is_batched": capabilities.qpos_is_batched,
        "qvel_is_batched": capabilities.qvel_is_batched,
        "ctrl_is_batched": capabilities.ctrl_is_batched,
        "xfrc_is_batched": capabilities.xfrc_is_batched,
        "xpos_is_batched": capabilities.xpos_is_batched,
        "xmat_is_batched": capabilities.xmat_is_batched,
        "qpos_write_tested": capabilities.qpos_write_tested,
        "qvel_write_tested": capabilities.qvel_write_tested,
        "ctrl_write_tested": capabilities.ctrl_write_tested,
        "xfrc_write_tested": capabilities.xfrc_write_tested,
        "qpos_write_method": capabilities.qpos_write_method,
        "qvel_write_method": capabilities.qvel_write_method,
        "ctrl_write_method": capabilities.ctrl_write_method,
        "xfrc_write_method": capabilities.xfrc_write_method,
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
    true_batched_scoring: bool | None = None,
    per_world_state_init: bool | None = None,
    scene_build_ok: bool = False,
    capability_probe_ok: bool = False,
    warmup_completed: bool = False,
    warmup_requested_steps: int = 0,
    warmup_executed_steps: int = 0,
    warmup_seconds: float = 0.0,
    warmup_reason: str | None = None,
    capture_graph_requested: bool = False,
    capture_graph_enabled: bool = False,
    capture_graph_reason: str | None = "disabled",
    capture_graph_sections: list[str] | None = None,
    capture_graph_replay_count: int = 0,
    completed_chunks: int = 0,
    failed_chunks: int = 0,
    chunk_reset_strategy: str = "unknown",
    chunk_reset_count: int = 0,
    inactive_worlds_zeroed: bool = False,
    sync_strategy: str = "phase_boundary_and_readback_interval",
    readback_interval: int = 1,
    sync_count: int | None = None,
    host_readback_count: int | None = None,
    readback_semantics: str = "per-step threshold detection",
) -> dict[str, Any]:
    inferred_true_batched = num_grasps > 0 and not sequential_fallback and failure_reason is None
    if true_batched_scoring is None:
        true_batched_scoring = inferred_true_batched
    if per_world_state_init is None:
        per_world_state_init = inferred_true_batched
    payload: dict[str, Any] = {
        "backend": "mujoco_warp",
        "experimental": True,
        "score_semantics": score_semantics,
        "true_batched_scoring": bool(true_batched_scoring),
        "per_world_state_init": bool(per_world_state_init),
        "wrench_directions": 12,
        "include_in_multifidelity": False,
        "scene_build_ok": scene_build_ok,
        "capability_probe_ok": capability_probe_ok,
        "warmup_completed": warmup_completed,
        "warmup_requested_steps": int(warmup_requested_steps),
        "warmup_executed_steps": int(warmup_executed_steps),
        "warmup_seconds": float(warmup_seconds),
        "warmup_reason": warmup_reason,
        "capture_graph_requested": capture_graph_requested,
        "capture_graph_enabled": capture_graph_enabled,
        "capture_graph_reason": capture_graph_reason,
        "capture_graph_sections": capture_graph_sections or [],
        "capture_graph_replay_count": int(capture_graph_replay_count),
        "completed_chunks": int(completed_chunks),
        "failed_chunks": int(failed_chunks),
        "failure_reason": failure_reason,
        "chunk_reset_strategy": chunk_reset_strategy,
        "chunk_reset_count": int(chunk_reset_count),
        "inactive_worlds_zeroed": bool(inactive_worlds_zeroed),
        "sync_strategy": sync_strategy,
        "readback_interval": int(readback_interval),
        "sync_count": sync_count,
        "host_readback_count": host_readback_count,
        "readback_semantics": readback_semantics,
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
        for args in ((mj_model, mj_data), (warp_model, mj_data), (mj_model,), (warp_model,)):
            try:
                return mjw.put_data(*args, **kwargs)
            except TypeError:
                continue
        return mjw.put_data(mj_model, mj_data)
    if hasattr(mjw, "make_data"):
        for args in ((mj_model,), (mj_model, mj_data), (warp_model,)):
            try:
                return mjw.make_data(*args, **kwargs)
            except TypeError:
                continue
        return mjw.make_data(mj_model, nworld=nworld)
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
