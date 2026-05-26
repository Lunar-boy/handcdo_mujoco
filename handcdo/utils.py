from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def stable_design_id(params: dict[str, Any]) -> str:
    import hashlib

    blob = json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]
