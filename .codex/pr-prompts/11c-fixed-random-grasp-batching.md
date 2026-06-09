# Codex Prompt: PR11c — Experimental Fixed Random-Grasp MuJoCo Warp Batching

You are working in the `Lunar-boy/handcdo_mujoco` repository.

Start from latest `main`, after PR11a and PR11b have been merged:

```bash
git checkout main
git pull origin main
git checkout -b pr11c-warp-fixed-grasp-batching
```

## Goal

Implement the first experimental true batched path for fixed random-grasp evaluation with MuJoCo Warp.

This stage should focus on the core backend and batch orchestration only.
Do not change the existing CPU Optuna/TPE path.
Do not change default CPU behavior.

## Repository context

Existing CPU scoring is the reference implementation.

The purpose of this stage is H100-class throughput exploration, not differentiable physics and not final scientific scoring.

Default output score semantics must remain conservative:

```json
"score_semantics": "experimental_non_equivalent"
```

Do not claim CPU-equivalent scoring unless the full CPU wrench semantics are implemented and regression-tested, which is not required in this stage.

## Required changes

Update:

```text
handcdo/backends/mujoco_warp.py
```

Add, if useful:

```text
handcdo/warp_batch_eval.py
```

Add tests:

```text
tests/test_warp_batch_eval_schema.py
tests/test_mujoco_warp_backend_optional.py
```

Do not add a CLI yet unless it is trivial. CLI belongs to PR11d.

## Core implementation requirement

`MujocoWarpBackend.evaluate_grasps_batch()` must not implement fake batching.

Unacceptable implementation:

```python
for grasp in grasps:
    self.evaluate_grasp(...)
```

Acceptable behavior:

1. Build/load the same MJCF as CPU MuJoCo for the same `HandDesign`, `tool_name`, and `GeometryConfig`.
2. Use CPU MuJoCo only for model construction/loading and reference naming utilities.
3. Transfer the model to MuJoCo Warp.
4. Allocate Warp data with `nworld`.
5. Process input grasps in chunks of at most `nworld`.
6. Initialize distinct grasp candidate states per world where the MuJoCo Warp API safely allows it.
7. Step worlds together with `mujoco_warp.step`.
8. Return exactly one `GraspEvaluation` per input grasp.
9. Record per-grasp structured failures when possible.

If true per-world grasp initialization is not safely implementable with the available MuJoCo Warp API, fail clearly with:

```python
NotImplementedError(
    "True per-world fixed-grasp initialization is not available for this MuJoCo Warp API; refusing to report fake batched scores."
)
```

A sequential fallback is only allowed when an explicit parameter such as `allow_sequential_fallback=True` is provided. If used, output metadata must clearly say:

```json
"score_semantics": "experimental_sequential_fallback"
```

and must not claim H100 batch throughput.

## Score semantics

Default for this stage:

```json
"score_semantics": "experimental_non_equivalent"
```

Only emit:

```json
"score_semantics": "intended_cpu_equivalent"
```

if the implementation actually matches CPU reference semantics:

- close hand;
- settle;
- save post-settle state;
- reset state for each wrench direction;
- apply the same 12 ramped Cartesian wrench directions;
- measure translation failure using the same threshold;
- measure rotation failure using the same threshold;
- aggregate normalized stable durations the same way.

This is not required in PR11c. Prefer conservative semantics.

## Batch orchestration

Add a helper function if useful, for example:

```python
def evaluate_fixed_grasps_batched(
    backend: BatchedSimulatorBackend,
    design: HandDesign,
    tool_name: str,
    grasps: list[GraspParams],
    config: EvaluationConfig | None,
    geometry_config: GeometryConfig | None = None,
    tool_assets_dir: str | Path = "assets/tools",
) -> list[GraspEvaluation]:
    ...
```

Rules:

- Preserve input ordering.
- Preserve list length.
- Do not drop failed trials.
- Failed trials should produce structured failure results when the existing data model supports it; otherwise raise clear exceptions at batch level.
- Keep random grasp generation outside the backend. The backend receives fixed `grasps`.

## Performance guardrails

Avoid obviously slow behavior:

- Do not transfer the model for every single grasp if it can be reused per design/tool/chunk.
- Do not allocate Warp data per single grasp inside a chunk.
- Do not synchronize after every single step unless necessary for correctness or timing.
- Put timing metadata in result payloads, but do not claim speedup unless measured.

## Tests

CPU-only tests must pass without `mujoco_warp`.

Add tests covering:

1. Batch helper preserves input length and order using a dummy batched backend.
2. Batch helper handles empty grasp lists deterministically.
3. `MujocoWarpBackend` still fails clearly when optional dependency is absent.
4. No CPU backend behavior changes.
5. Output metadata helper, if added, includes:
   - `backend`
   - `experimental`
   - `score_semantics`
   - `nworld`
   - `num_grasps`
   - `num_chunks`
   - `failure_count`

GPU tests may be added but must be skipped by default unless both `mujoco_warp` and a CUDA device are available.

Use markers such as:

```python
@pytest.mark.gpu
```

and require:

```bash
RUN_GPU_TESTS=1
```

for real GPU tests.

## Validation

CPU-only:

```bash
pytest -q
python3 scripts/benchmark_mujoco_warp.py --help
```

Optional GPU smoke, only when available:

```bash
python3 -m pip install -e ".[warp]"
RUN_GPU_TESTS=1 pytest -q -m gpu
```

## Out of scope

Do not implement:

- new public CLI;
- TPE batching;
- CPU-equivalent scoring unless already straightforward;
- CPU-vs-Warp comparison helper;
- JAX/MJX/autodiff;
- Isaac Sim;
- Slurm production templates.

## Success criteria

This stage is successful if:

1. There is a non-fake batched backend path or a clear refusal to report fake scores.
2. CPU tests remain independent of MuJoCo Warp and GPU.
3. Output semantics remain conservative.
4. CPU MuJoCo remains the reference backend.
