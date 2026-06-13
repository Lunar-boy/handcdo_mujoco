# PR13b Prompt: Implement MuJoCo Warp Batched Field Write Adapter

You are working in the `handcdo_mujoco` repository.

## Goal

Make the experimental MuJoCo Warp backend pass strict GPU validation on a real H100 runtime by implementing a safe, verified batched field write adapter.

This PR must **not** bypass the existing safety gate. It must make the gate pass only when the backend can really write different grasp initial states into different MuJoCo Warp worlds and verify the result by host readback.

## Current failure observed on real GPU

PR13a strict Slurm validation reaches the real GPU runtime successfully:

```text
GPU: NVIDIA H100
CUDA_VISIBLE_DEVICES=0
mujoco==3.8.1
mujoco-warp==3.9.0.1
warp-lang==1.14.0
```

But strict validation fails with:

```text
MujocoWarpCapabilityError:
True per-world fixed-grasp initialization is not available for this MuJoCo Warp API;
refusing to report fake batched scores.
```

The JSON report shows that MuJoCo Warp data fields exist and are recognized as batched:

```json
{
  "has_qpos": true,
  "has_qvel": true,
  "has_ctrl": true,
  "has_xfrc_applied": true,
  "has_xpos": true,
  "has_xmat": true,
  "qpos_is_batched": true,
  "qvel_is_batched": true,
  "ctrl_is_batched": true,
  "xfrc_is_batched": true,
  "xpos_is_batched": true,
  "xmat_is_batched": true
}
```

But write verification fails:

```json
{
  "can_set_per_world_qpos": false,
  "can_set_per_world_qvel": false,
  "can_set_per_world_ctrl": false,
  "can_set_per_world_xfrc": false,
  "qpos_write_tested": false,
  "qvel_write_tested": false,
  "ctrl_write_tested": false,
  "xfrc_write_tested": false,
  "qpos_write_method": "none",
  "qvel_write_method": "none",
  "ctrl_write_method": "none",
  "xfrc_write_method": "none",
  "supports_true_fixed_grasp_batching": false,
  "true_fixed_grasp_batching_reason": "warp_data.qpos lacks verified per-world write support; warp_data.qvel lacks verified per-world write support; warp_data.ctrl lacks verified per-world write support; warp_data.xfrc_applied lacks verified per-world write support"
}
```

Interpretation:

- This is **not** a Slurm, CUDA, H100, package installation, or import failure.
- The data fields are present and batched.
- The current capability probe fails because it cannot verify a safe write path for `qpos`, `qvel`, `ctrl`, and `xfrc_applied`.
- The current implementation likely tests direct per-world slice assignment too narrowly.
- MuJoCo Warp may support writing the whole batched field even if `field[world_index, ...] = value` does not work.

## Required scope

Modify only what is needed for MuJoCo Warp batched field writing and capability reporting.

Expected files:

```text
handcdo/warp_utils.py
handcdo/backends/mujoco_warp.py
tests/test_warp_utils.py
tests/test_mujoco_warp_backend.py
docs/mujoco_warp_gpu_validation.md        # only if documentation needs a small update
```

Do **not** modify unrelated optimizer logic, design-space logic, CPU MuJoCo scoring, or multifidelity behavior.

## Non-negotiable safety constraints

1. Do **not** remove, weaken, or bypass the `supports_true_fixed_grasp_batching` safety gate.
2. Do **not** silently fall back to CPU.
3. Do **not** report fake batched scores.
4. Do **not** set `supports_true_fixed_grasp_batching=True` unless write + readback verification really passes.
5. Do **not** change CPU backend semantics.
6. Keep the MuJoCo Warp backend explicitly experimental.
7. Keep `include_in_multifidelity=False`.
8. Keep default CPU tests CUDA-free.

## Main implementation requirement

The backend currently requires verified per-world state initialization. That requirement is correct, but the write adapter should support both:

1. **Per-world slice writes**, when supported by the runtime.
2. **Whole-batch writes**, when the runtime supports writing an entire batched array but not individual world slices.

The actual backend only needs to initialize and update batched arrays such as:

```text
qpos:         (nworld, nq)
qvel:         (nworld, nv)
ctrl:         (nworld, nu)
xfrc_applied: (nworld, nbody, 6)
```

Therefore, whole-batch write is valid if it can write a full array with different values per world and then verify those values by host readback.

## Implementation details

### 1. Add a reusable batched field write helper in `handcdo/warp_utils.py`

Create an internal helper with behavior similar to:

```python
def try_write_batched_field(
    field: object,
    value: np.ndarray,
    *,
    field_name: str,
    mjw: object | None = None,
) -> tuple[bool, str, str]:
    ...
```

Return:

```text
(success, method, reason)
```

The helper should try guarded write methods in a deterministic order and never crash on unsupported APIs.

Recommended method order:

1. Direct whole-field assignment:

```python
field[...] = value
```

2. Object methods if available:

```python
field.assign(value)
field.copy_(value)
field.copy(value)
```

3. Warp copy from a `wp.array` source if available:

```python
import warp as wp
source = wp.from_numpy(value, dtype=..., device=...)
wp.copy(field, source)
```

Be defensive about dtype and device discovery. Try to infer from `field.dtype` and `field.device` when available. If dtype/device inference fails, try `wp.from_numpy(value)` as a fallback. Capture exceptions in the returned reason.

The helper must not assume a single Warp API signature. Use guarded attempts and detailed error messages.

### 2. Keep or adapt the existing per-world helper

If `_try_write_field_per_world(...)` already exists, keep it for runtimes that support slice writes.

However, capability verification should not fail merely because slice writes fail if whole-batch writes work.

### 3. Update capability probe to verify whole-batch write

Update the capability probe in `handcdo/warp_utils.py`, especially the logic that reports:

```text
qpos_write_tested
qvel_write_tested
ctrl_write_tested
xfrc_write_tested
qpos_write_method
qvel_write_method
ctrl_write_method
xfrc_write_method
can_set_per_world_qpos
can_set_per_world_qvel
can_set_per_world_ctrl
can_set_per_world_xfrc
supports_true_fixed_grasp_batching
```

The probe should verify this property:

> A batched MuJoCo Warp field can be written with different values for different worlds, and the written values can be read back correctly.

Suggested verification algorithm for each writable field:

1. Read the original full host snapshot using the existing host-readback helper.
2. Require leading dimension `>= nworld` and recognized batched shape.
3. Construct a mutated full-batch copy:
   - Change at least world 0.
   - If `nworld >= 2`, also change world 1 differently from world 0.
   - Keep all other worlds unchanged.
4. Try per-world write if already supported.
5. If per-world write fails, try whole-batch write of the mutated full array.
6. Read back from the field.
7. Verify:
   - world 0 equals its mutated value;
   - if `nworld >= 2`, world 1 equals its distinct mutated value;
   - untouched worlds remain unchanged when applicable.
8. Restore the original full snapshot.
9. Read back again and verify restoration.
10. Only then mark the write as tested/supported.

If the successful method is whole-batch write, set the method string to something explicit, for example:

```text
whole_batch_setitem
whole_batch_assign
whole_batch_copy_
whole_batch_copy
whole_batch_wp_copy
```

If a per-world method works, keep the existing method label.

### 4. Update backend runtime write path

Update `_write_required_field(...)` in `handcdo/backends/mujoco_warp.py` so the actual backend uses the same verified write strategy.

Current behavior likely tries:

```python
field[: value.shape[0], ...] = value
```

and then falls back to per-world writes. Extend it so that it can use whole-batch writes.

Required behavior:

- If `value.shape[0] == field_nworld`, try writing the full batch directly.
- If `value.shape[0] < field_nworld`, handle partial chunks safely:
  1. Read the full current field to host.
  2. Replace only `full[: value.shape[0]]` with `value`.
  3. Write the whole full-batch array back.
- If host readback is unavailable for partial chunks, raise `MujocoWarpCapabilityError`; do not silently continue.
- Keep inactive worlds zeroed where existing backend logic expects them to be zeroed.
- Preserve existing direct assignment/per-world fallback behavior where it already works.

The backend path and the capability probe should share the same helper as much as possible. Avoid duplicating low-level Warp write logic.

### 5. Improve capability diagnostics

Expose enough detail in metadata to debug future runtime/API differences.

If existing `WarpBatchCapabilities` already stores field reports, extend `warp_capabilities_payload(...)` to include a compact per-field report, for example:

```json
"field_reports": {
  "qpos": {
    "present": true,
    "batched": true,
    "shape": [2, 31],
    "write_tested": true,
    "write_method": "whole_batch_wp_copy",
    "write_reason": null
  }
}
```

Do not break existing metadata keys. Add new keys only.

If adding full `field_reports` is too invasive, at minimum improve the existing reason strings so the JSON report shows why each write method failed.

## Tests

Add or update CPU-safe unit tests using fake field objects. These tests must not require CUDA, MuJoCo Warp, or a real GPU.

### Required test cases

1. **Whole-batch write fallback succeeds**
   - Fake field rejects per-world slice assignment.
   - Fake field accepts full-batch assignment.
   - Capability probe reports:
     - `*_write_tested=True`
     - `can_set_per_world_*=True`
     - write method contains `whole_batch`
     - `supports_true_fixed_grasp_batching=True` when all required fields support it.

2. **Whole-batch write verifies distinct worlds**
   - Use `nworld=2`.
   - Ensure the probe writes different values to world 0 and world 1.
   - Verify the fake field receives and stores different values for each world.

3. **Restore original field after probe**
   - After probing, the fake field must contain the original values, not the mutated probe values.

4. **Backend `_write_required_field(...)` uses whole-batch fallback**
   - Fake field rejects direct partial slice assignment.
   - Fake field accepts whole-batch assignment.
   - Calling `_write_required_field(...)` successfully updates the active worlds.

5. **Partial chunk preserves or clears inactive worlds according to backend expectations**
   - For `value.shape[0] < nworld`, verify inactive worlds are not polluted by old active-world data.
   - If existing backend semantics zero inactive worlds, assert zeros.
   - If existing semantics preserve inactive worlds until explicitly zeroed elsewhere, assert the documented current behavior.
   - Do not introduce hidden stale-world contamination.

6. **Unsupported write still fails truthfully**
   - Fake field rejects all write methods.
   - Capability probe must report `supports_true_fixed_grasp_batching=False`.
   - Backend write path must raise `MujocoWarpCapabilityError` rather than silently continuing.

7. **CPU test suite stays CUDA-free**
   - No test should import `mujoco_warp` or initialize Warp CUDA unless explicitly marked/gated as GPU.

## Validation commands

Run CPU-safe checks:

```bash
python -m py_compile \
  handcdo/warp_utils.py \
  handcdo/backends/mujoco_warp.py

python -m pytest -q
python -m pytest -q -rs
```

Run targeted tests:

```bash
python -m pytest -q tests/test_warp_utils.py tests/test_mujoco_warp_backend.py
```

If a real Slurm GPU environment is available, run strict validation:

```bash
scripts/submit_mujoco_warp_gpu_validation.sh
```

Expected strict validation success after this PR:

```text
status: passed
```

The generated report should show:

```json
{
  "backend_metadata": {
    "failure_count": 0,
    "failure_reason": null,
    "true_batched_scoring": true,
    "per_world_state_init": true,
    "sequential_fallback": false,
    "warp_capabilities": {
      "supports_true_fixed_grasp_batching": true,
      "can_set_per_world_qpos": true,
      "can_set_per_world_qvel": true,
      "can_set_per_world_ctrl": true,
      "can_set_per_world_xfrc": true
    }
  }
}
```

If strict validation still fails, do not hide the failure. Improve the field-level diagnostics so the JSON report makes the remaining runtime/API incompatibility clear.

## Acceptance criteria

This PR is complete only if:

1. CPU tests still pass without CUDA.
2. The MuJoCo Warp capability probe can verify whole-batch write support.
3. The real backend write path can use the same whole-batch write adapter.
4. `supports_true_fixed_grasp_batching=True` is reported only after write + readback verification.
5. The safety gate remains intact.
6. No CPU fallback is introduced.
7. No fake batched scoring is introduced.
8. Partial chunks do not pollute inactive worlds.
9. Metadata diagnostics are more informative than before.
10. Strict Slurm GPU validation is expected to pass on the observed H100 environment if MuJoCo Warp supports whole-batch writes.

## Final PR summary format

After implementation, write the PR summary as:

```text
Summary:
- Added a guarded MuJoCo Warp batched field write adapter.
- Extended capability probing to verify per-world-different whole-batch writes.
- Updated the MuJoCo Warp backend write path to use the shared adapter.
- Improved write capability diagnostics in validation metadata.
- Added CPU-safe tests for whole-batch write fallback, restoration, partial chunks, and unsupported writes.

Validation:
- python -m py_compile handcdo/warp_utils.py handcdo/backends/mujoco_warp.py
- python -m pytest -q
- python -m pytest -q -rs
- If available: scripts/submit_mujoco_warp_gpu_validation.sh

Notes:
- No backend safety gate was bypassed.
- No CPU fallback was introduced.
- MuJoCo Warp remains experimental.
```
