# Codex Prompt: PR11-d — Minimal Experimental MuJoCo Warp Batch Backend

You are working in the `Lunar-boy/handcdo_mujoco` repository.

Start from latest `main` after PR11-a, PR11-b, and PR11-c have merged:

```bash
git checkout main
git pull origin main
git checkout -b pr11d-minimal-experimental-warp-backend
```

## Goal

Implement the first minimal, explicit, experimental MuJoCo Warp backend path for batched fixed-grasp evaluation.

This PR should focus on backend implementation only.

Do not add the public CLI yet.

Do not change existing CPU evaluation scripts.

Do not change default CPU behavior.

## Scientific and semantic guardrails

CPU MuJoCo remains the reference implementation.

MuJoCo Warp output must default to:

```json
"score_semantics": "experimental_non_equivalent"
```

Do not output:

```json
"score_semantics": "intended_cpu_equivalent"
```

in this PR.

That label is reserved for a later PR only after CPU-vs-Warp regression tests demonstrate score agreement and ranking stability.

Do not use Warp scores in multifidelity `best_available_score`.

Do not modify default `evaluate_design_batch.py`, `run_optuna_round.py`, or existing CPU ranking logic to consume Warp results.

## Required changes

Update:

```text
handcdo/backends/mujoco_warp.py
handcdo/warp_utils.py
```

Add or update tests:

```text
tests/test_mujoco_warp_backend_optional.py
tests/test_warp_backend_metadata.py
```

Optional GPU tests, skipped by default:

```text
tests/test_mujoco_warp_backend_gpu_smoke.py
```

## Core implementation requirement

`MujocoWarpBackend.evaluate_grasps_batch()` must not implement fake batching.

Unacceptable:

```python
for grasp in grasps:
    self.evaluate_grasp(...)
```

Acceptable high-level behavior:

1. Build or load the same MJCF as CPU MuJoCo for the same `HandDesign`, `tool_name`, and `GeometryConfig`.
2. Use CPU MuJoCo only for XML/model loading and naming utilities.
3. Apply the existing PR10 MJCF compatibility rewrite.
4. Transfer the model to MuJoCo Warp.
5. Allocate one batched Warp data object with `nworld`.
6. Process input grasps in chunks of size `<= nworld`.
7. Map each grasp candidate to one world index inside the chunk.
8. Initialize distinct grasp candidate states per world if and only if the installed MuJoCo Warp API safely supports this.
9. Step worlds together with MuJoCo Warp.
10. Return exactly one `GraspEvaluation` per input grasp in the same order.
11. Record structured failures where possible.
12. Include conservative metadata.

If true per-world grasp initialization is not safely implementable with the available MuJoCo Warp API, fail clearly:

```python
raise NotImplementedError(
    "True per-world fixed-grasp initialization is not available for this MuJoCo Warp API; refusing to report fake batched scores."
)
```

Do not silently return fake scores.

Do not silently call the CPU backend.

## Per-world state initialization requirements

Document the implementation in code comments.

For each chunk:

1. create or reset one batched MuJoCo Warp data object;
2. map each input grasp to exactly one world index;
3. set the tool free-joint pose per world;
4. set hand actuator controls per world for closure when supported;
5. run close steps and settle steps together across worlds;
6. if wrench scoring is implemented, snapshot the settled state per world;
7. for each wrench direction, restore each world to its own settled state before applying disturbance;
8. aggregate per-world scores back to input order.

If any of these operations are not supported by the installed MuJoCo Warp API, the backend must either:

- raise a clear `NotImplementedError`, or
- return structured failed evaluations with `score_semantics="experimental_non_equivalent"`.

It must not pretend that incomplete physics is CPU-equivalent.

## Minimal scoring policy

This PR may implement one of two options.

### Option A: Clear refusal path

If correct per-world initialization or stepping cannot be implemented safely, keep `evaluate_grasps_batch()` as a clear refusal path with detailed metadata and tests.

This is acceptable.

### Option B: Experimental non-equivalent score

If a safe minimal Warp rollout can be implemented, return experimental scores with:

```json
{
  "backend": "mujoco_warp",
  "experimental": true,
  "score_semantics": "experimental_non_equivalent"
}
```

The score may be a conservative proxy only if documented as such.

Do not claim the same semantics as CPU `evaluate_grasp()`.

## Sequential fallback policy

Sequential fallback is disabled by default.

If the constructor has `allow_sequential_fallback=True`, then and only then a sequential path may be used for debugging.

If used, metadata must say:

```json
"score_semantics": "experimental_sequential_fallback"
```

and:

```json
"sequential_fallback": true
```

Such results must not claim GPU batch throughput.

## Metadata requirements

Every batch evaluation or batch-level helper should expose metadata where practical:

```json
{
  "backend": "mujoco_warp",
  "experimental": true,
  "score_semantics": "experimental_non_equivalent",
  "nworld": 64,
  "nconmax": 64,
  "naconmax": null,
  "njmax": 128,
  "num_grasps": 128,
  "num_chunks": 2,
  "failure_count": 0,
  "sequential_fallback": false,
  "seconds_total": 0.0,
  "grasps_per_second": null,
  "world_steps_per_second": null,
  "mjcf_rewrites": []
}
```

Device info is optional. If included, keep it safe and non-fragile.

## API safety

Do not invent MuJoCo Warp APIs.

Use only:

- helper code already validated by PR10;
- runtime introspection with explicit guards;
- imports verified in tests or optional GPU smoke.

If an API is unavailable, fail clearly.

## Tests

CPU-only tests must pass without `mujoco_warp`.

Cover:

1. Missing `mujoco_warp` gives a helpful error only when backend is explicitly requested or used.
2. Invalid constructor values fail before optional import.
3. Metadata helper includes required keys.
4. Sequential fallback is disabled by default.
5. The backend does not call CPU `evaluate_grasp()` silently.
6. Empty batch returns empty list or clearly documented behavior.
7. Existing CPU workflows are unchanged.

Optional GPU smoke tests may be added but must be skipped unless both conditions hold:

```text
RUN_GPU_TESTS=1
mujoco_warp is importable and a CUDA device is available
```

Use a marker:

```python
@pytest.mark.gpu
```

## Validation

CPU-only:

```bash
pytest -q
python3 scripts/benchmark_mujoco_warp.py --help
```

Optional GPU smoke:

```bash
python3 -m pip install -e ".[warp]"
RUN_GPU_TESTS=1 pytest -q -m gpu
```

## Out of scope

Do not implement:

- public Warp batch CLI;
- CPU-vs-Warp comparison helper;
- TPE batching;
- integration into multifidelity ranking;
- claims of CPU-equivalent scoring;
- JAX/MJX/autodiff;
- Isaac Sim;
- ROS;
- RL or policy learning;
- Slurm production templates.

## Success criteria

This stage is successful if:

1. The backend either implements a real non-fake batch path or clearly refuses when the installed API cannot support it.
2. No silent CPU fallback exists.
3. All Warp results are marked experimental and non-equivalent.
4. CPU-only tests pass without GPU dependencies.
5. Existing CPU workflows remain unchanged.


## Mandatory capability probe:
Even if the backend cannot safely implement real per-world grasp initialization, this PR must add a small import-safe capability helper, for example:

`probe_mujoco_warp_capabilities() -> dict`

When `mujoco_warp` is available, it should report:
- whether `put_model` exists;
- whether `put_data` or `make_data` exists;
- accepted data allocation kwargs discovered by guarded calls;
- whether `step` exists;
- whether per-world qpos/qvel/ctrl/xfrc assignment appears accessible;
- the exact reason why true per-world fixed-grasp initialization is or is not supported.

Do not invent APIs. Use guarded runtime introspection and catch exceptions.
The refusal path must include this capability report in the error metadata or logs.