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
