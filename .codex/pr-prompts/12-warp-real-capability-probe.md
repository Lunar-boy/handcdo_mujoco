# Codex Prompt: PR12 — Real MuJoCo Warp Capability Probe and Per-World State Write Smoke Test

You are working in the `Lunar-boy/handcdo_mujoco` repository.

Start from latest `main` after PR11 has merged:

```bash
git checkout main
git pull origin main
git checkout -b pr12-warp-real-capability-probe
```

## Goal

Replace the current static/conservative MuJoCo Warp capability detection with a runtime probe that can inspect actual MuJoCo Warp `Model`/`Data` objects when available.

This PR must answer one concrete question:

> Can this installed MuJoCo Warp environment represent and write per-world `qpos`, `qvel`, `ctrl`, and `xfrc_applied` arrays for true fixed-grasp batch evaluation?

This PR is not yet full grasp scoring. It is a capability and state-write foundation PR.

## Files to inspect first

Inspect the current repository before editing. In particular, inspect files matching these roles:

```text
handcdo/backends/registry.py
handcdo/backends/mujoco_warp.py
handcdo/backends/*warp*.py
handcdo/backends/*batched*.py
handcdo/warp_utils.py
handcdo/benchmarks/mujoco_warp.py
scripts/benchmark_mujoco_warp.py
scripts/evaluate_design_batch_warp.py
tests/test_*warp*.py
tests/test_warp_backend_metadata.py
tests/test_mujoco_warp_backend_optional.py
tests/test_*backend*.py
pyproject.toml
README.md
.codex/pr-prompts/README.md
```

Use the actual existing filenames. Do not create duplicate utilities if equivalent utilities already exist.


## Implementation guardrails

`handcdo/warp_utils.py` is the canonical home for `WarpBatchCapabilities`,
`inspect_warp_batch_capabilities`, `make_warp_data`, and `warp_batch_metadata`.
Do not create a second competing capability-probe module unless the existing
structure clearly requires it.

Preserve existing public field names and JSON metadata keys where possible,
especially:

```text
can_put_model
can_put_data
can_make_data
can_step
can_set_per_world_qpos
can_set_per_world_qvel
can_set_per_world_ctrl
can_set_per_world_xfrc
supports_true_fixed_grasp_batching
true_fixed_grasp_batching_reason
```

Prefer additive fields over renaming existing fields. If a field is renamed,
update all call sites, metadata serialization, benchmark outputs, and existing
tests deliberately in the same PR.

For real `warp_data` probing, distinguish three levels:

1. field is present;
2. field appears batched with leading `nworld`;
3. field has an actually tested per-world write path.

Do not set `can_set_per_world_* = True` from shape alone. Batched shape is
necessary but not sufficient for true fixed-grasp batching.

For write smoke tests:

- Use a fresh smoke-test `warp_data` whenever possible.
- If mutating an existing object, snapshot original values first and restore
  them before returning.
- Prefer official/runtime-discovered state APIs such as `set_state`,
  `get_state`, or `reset_data` if exposed by the installed `mujoco_warp`.
- Only use direct field writes when the field object clearly supports safe
  assignment/copy semantics.
- Synchronize before and after GPU writes when a `warp.synchronize()` path
  exists.
- Verify writes by round-tripping to host when possible.
- If verification is impossible, return `write_tested=False` with a clear
  reason instead of reporting success.
- Never leave mutated `qpos`, `qvel`, `ctrl`, or `xfrc_applied` values in a
  caller-owned data object.

`supports_true_fixed_grasp_batching` may be true only when:

- MuJoCo Warp import/module capability is available;
- model/data creation and stepping APIs are present;
- `qpos`, `qvel`, `ctrl`, and `xfrc_applied` are present;
- all four fields are batched with leading `nworld`;
- all four fields have verified per-world write support.

After `handcdo/benchmarks/mujoco_warp.py` creates `warp_model` and `warp_data`,
record the real capability probe result in benchmark JSON/CSV diagnostics
without changing benchmark scoring semantics.

The optional GPU smoke test is diagnostic only. It must skip clearly when the
installed MuJoCo Warp runtime, CUDA context, or data object does not expose the
required write path. A skip is acceptable; a fake pass is not.

Keep PR12 narrow. It must not implement fixed-grasp batch scoring, wrench
evaluation, TPE batching, graph capture, CPU-vs-Warp score validation, or CLI
default changes.

## Required changes

### 1. Add a real capability dataclass if it does not already exist

Create or extend a capability structure similar to:

```python
@dataclass(frozen=True)
class WarpBatchCapabilities:
    import_available: bool
    has_put_model: bool
    has_make_data: bool
    has_step: bool
    has_qpos: bool
    has_qvel: bool
    has_ctrl: bool
    has_xfrc_applied: bool
    qpos_is_batched: bool
    qvel_is_batched: bool
    ctrl_is_batched: bool
    xfrc_is_batched: bool
    can_set_per_world_qpos: bool
    can_set_per_world_qvel: bool
    can_set_per_world_ctrl: bool
    can_set_per_world_xfrc: bool
    supports_true_fixed_grasp_batching: bool
    reason: str
```

If an existing capability dataclass already exists, extend it instead of replacing it.
Do not rename existing public fields merely to match this sketch; the sketch is
illustrative. Prefer preserving the current `can_*` metadata schema and adding
new fields only when needed.

### 2. Replace static probing with runtime probing

Current probing may be conservative and hard-code per-world state support to `False`. Replace or extend it with a function that can accept actual objects:

```python
def inspect_warp_batch_capabilities(
    mjw: Any | None = None,
    *,
    warp_model: Any | None = None,
    warp_data: Any | None = None,
    nworld: int | None = None,
) -> WarpBatchCapabilities:
    ...
```

Requirements:

- Must work when `mujoco_warp` is not installed.
- Must work when called with only the module object.
- Must work when called with real `warp_model` and `warp_data`.
- Must not require GPU in CPU-only tests.
- Must not import `mujoco_warp` at package import time.
- Must not raise raw `ModuleNotFoundError` for normal backend import paths.

When `warp_data` is provided, inspect fields such as:

```text
qpos
qvel
ctrl
xfrc_applied
```

Check whether they are shaped with a leading world dimension, for example:

```text
qpos: (nworld, nq)
qvel: (nworld, nv)
ctrl: (nworld, nu)
xfrc_applied: (nworld, nbody, 6)
```

Do not assume exact concrete array classes. Use guarded introspection.
Do not infer per-world write support from shape inspection alone.

### 3. Add a guarded per-world state write smoke helper

Add a helper used only by optional GPU tests or diagnostic scripts:

```python
def smoke_test_warp_per_world_state_write(
    warp_data: Any,
    *,
    nworld: int,
    require_fields: tuple[str, ...] = ("qpos", "qvel", "ctrl", "xfrc_applied"),
) -> dict[str, Any]:
    ...
```

The helper should attempt to verify that each field can be assigned per world.

Implementation policy:

- Prefer runtime-safe APIs discovered from the existing PR11 utility code.
- Prefer official/runtime-discovered MuJoCo Warp state APIs when available,
  such as `set_state`, `get_state`, or `reset_data`.
- If a direct assignment or copy path is not available, return a clear diagnostic instead of guessing.
- Snapshot and restore original values when mutating an existing `warp_data`.
- Synchronize around GPU writes when `warp.synchronize()` is available.
- Verify writes by round-tripping values back to host when possible.
- Do not fake success.
- Do not call CPU backend and label it as Warp.
- Do not require this helper in CPU-only tests.
- Do not leave caller-owned `warp_data` in a mutated state.

Suggested return schema:

```python
{
    "ok": bool,
    "fields": {
        "qpos": {"present": bool, "batched": bool, "write_tested": bool, "reason": str},
        "qvel": {...},
        "ctrl": {...},
        "xfrc_applied": {...},
    },
    "reason": str,
}
```

### 4. Update `MujocoWarpBackend` to expose capabilities

Add a method such as:

```python
def capabilities(self) -> WarpBatchCapabilities:
    ...
```

or expose a property if the codebase already uses that style.

The method must:

- lazily import optional Warp only when the experimental backend is explicitly constructed or queried;
- avoid constructing heavy models unless explicitly requested;
- return conservative `supports_true_fixed_grasp_batching=False` when no `warp_data` is available;
- return precise diagnostics when actual `warp_data` is available.

### 5. Add CPU-only tests

Add tests that pass without `mujoco_warp` installed:

1. Importing `handcdo.backends` does not import `mujoco_warp`.
2. `get_backend("mujoco")` and `get_backend("mujoco_cpu")` still return the CPU backend.
3. Requesting `mujoco_warp` either constructs the backend when the optional dependency is available or raises a clear optional-dependency error.
4. Capability probing without `warp_data` remains conservative and does not claim true batching.
5. Fake/minimal mock objects with `qpos/qvel/ctrl/xfrc_applied` shaped as batched arrays are detected as batched.
6. Fake/minimal mock objects without batched leading world dimensions are rejected.
7. No default test requires CUDA.

### 6. Add optional GPU smoke test

If the repository already has GPU test markers, reuse them. Otherwise add a marker such as:

```python
pytestmark = pytest.mark.gpu
```

Gate execution behind:

```text
RUN_GPU_TESTS=1
```

The optional GPU smoke test should:

- create the smallest feasible MJCF scene using existing helpers;
- create `warp_model` and `warp_data`;
- run capability inspection;
- verify that per-world state fields are present and batched;
- verify write support only through an actual round-trip smoke test;
- optionally call the per-world state write smoke helper.

Skip clearly if:

- `mujoco_warp` is unavailable;
- no CUDA/GPU context is available;
- the runtime API does not support the required write path.

## Validation

Run on CPU-only environment:

```bash
pytest -q
python3 scripts/benchmark_mujoco_warp.py --help
python3 scripts/evaluate_design_batch_warp.py --help
```

Optional GPU validation:

```bash
python3 -m pip install -e ".[warp]"
RUN_GPU_TESTS=1 pytest -q -m gpu
```

## Out of scope

Do not implement:

- full grasp scoring;
- full wrench testing;
- TPE batching;
- CPU-vs-Warp score validation;
- multi-tool screening;
- graph capture optimization;
- default CLI changes;
- multifidelity integration;
- JAX/MJX/autodiff.

## Acceptance criteria

This PR is acceptable if:

1. CPU-only tests pass without `mujoco_warp`.
2. Default backend behavior is unchanged.
3. Runtime capability probing can distinguish conservative/no-data mode from real `warp_data` mode.
4. The code no longer hard-codes all per-world state support to `False` when actual `warp_data` is available.
5. Shape-only detection never reports verified per-world write support.
6. No fake GPU success path exists.
7. Optional GPU tests, when run in a valid environment, report whether per-world `qpos/qvel/ctrl/xfrc_applied` write support is actually available.
