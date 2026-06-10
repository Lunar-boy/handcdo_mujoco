# Codex Prompt: PR13 — True MuJoCo Warp Fixed-Random Batch Scoring for One Design/Tool Path

You are working in the `Lunar-boy/handcdo_mujoco` repository.

Start from latest `main` after PR12 has merged:

```bash
git checkout main
git pull origin main
git checkout -b pr13-true-warp-fixed-random-scoring
```

## Goal

Implement a minimal true MuJoCo Warp fixed-random grasp batch evaluator for one design and one tool.

This PR should move beyond skeleton backend behavior. It should perform actual batched GPU simulation when `mujoco_warp` is available and capability probing confirms true per-world state support.

The target is not final performance. The target is physical pipeline correctness:

```text
generate design MJCF
load MuJoCo model
create Warp model/data with nworld
probe the concrete Warp data object for per-world write support
map fixed random grasps into per-world qpos/qvel/ctrl
close/settle hand
run 12-direction wrench tests
return GraspEvaluation-compatible results and expose batch-level experimental Warp metadata
```

Do not claim CPU equivalence in this PR. The Warp score semantics must remain explicitly marked experimental and non-equivalent.

## Files to inspect first

Inspect the current implementation before editing:

```text
handcdo/backends/mujoco_warp.py
handcdo/backends/mujoco_cpu.py
handcdo/mujoco_eval.py
handcdo/grasp_sampling.py
handcdo/geometry_config.py
handcdo/design_space.py
handcdo/mjcf_generator.py
handcdo/tools.py
handcdo/wrench_score.py
handcdo/warp_utils.py
handcdo/backends/*warp*.py
scripts/evaluate_design_batch_warp.py
tests/test_*warp*.py
tests/test_*mujoco*.py
```

Important repository-path constraint:

- Do not create `handcdo/mjcf_builder.py`.
- The current MJCF generator is `handcdo/mjcf_generator.py`.

Find the CPU reference logic for:

- setting tool free-joint pose;
- applying hand actuator controls;
- closing/settling the hand;
- saving/restoring settled state;
- applying 12 wrench directions;
- computing final normalized score.

Do not duplicate semantics blindly. Extract shared helper logic where safe.

CPU semantics that must be matched where possible:

- Tool pose must follow the current CPU helper semantics: find the `tool_free` free joint, write `tool.reference_pos + [dx, dy, dz]` to the free-joint position, and convert `roll/pitch/yaw` with the same MuJoCo `XYZ` Euler convention.
- Hand closure must follow the current actuator-control semantics: actuator names beginning with `thumb` use `thumb_closure`; other actuators use `closure`; preserve the existing `spread_bias` parity rule and clipping range.
- Wrench testing must follow the current CPU semantics: restore settled `qpos/qvel/ctrl` before every direction, clear `xfrc_applied`, run the MuJoCo forward/kinematics-equivalent update before measuring the start pose, ramp wrench magnitudes linearly, and use the existing translation/rotation thresholds.
- Use existing `WRENCH_DIRECTIONS` and `aggregate_wrench_results` from `handcdo/wrench_score.py`; do not redefine another list of wrench directions.

## Required changes

### 1. Add a small Warp scene bundle

Create a small internal dataclass such as:

```python
@dataclass
class WarpSceneBundle:
    mj_model: mujoco.MjModel
    mj_data: mujoco.MjData
    warp_model: Any
    warp_data: Any
    tool_body_id: int
    tool_qpos_addr: int
    actuator_names: list[str]
    nworld: int
```

Responsibilities:

- build or load the same MJCF used by CPU evaluation;
- apply existing Warp-compatible MJCF preparation utilities;
- create standard MuJoCo `MjModel` and `MjData`;
- call verified `mujoco_warp` model/data creation APIs;
- identify tool body id and tool free-joint qpos address;
- expose actuator names and counts.

MuJoCo Warp API requirements:

- Prefer the current public API shape:
  - `warp_model = mjw.put_model(mj_model)`
  - `warp_data = mjw.put_data(mj_model, mj_data, nworld=nworld, nconmax=..., naconmax=..., njmax=...)`
- Do not assume `make_data` accepts `warp_model`. If `make_data` is used as a fallback, audit and test whether it requires `mujoco.MjModel`.
- If `handcdo.warp_utils.make_warp_data` is used, audit and fix its fallback signatures instead of hiding API mismatches in the backend.
- Preserve guarded fallbacks only when they are tested or clearly justified.

Do not expose this as a public stable API yet unless existing code style requires it.

### 2. Implement per-world fixed-grasp state initialization

Add a function or method that maps a list of `GraspParams` into batched state arrays:

```python
def build_batched_initial_state(
    bundle: WarpSceneBundle,
    grasps: list[GraspParams],
    config: EvaluationConfig,
) -> BatchedInitialState:
    ...
```

It should produce host-side arrays such as:

```text
qpos_init: (batch, nq)
qvel_init: (batch, nv)
ctrl_init: (batch, nu)
xfrc_zero: (batch, nbody, 6)
```

Requirements:

- one world corresponds to one fixed grasp;
- write tool position and orientation into the tool free joint using the CPU reference convention;
- write actuator controls according to the same closure semantics as the CPU evaluator;
- zero velocities;
- zero external forces;
- preserve CPU reference semantics as much as possible;
- validate batch size does not exceed `nworld`;
- return clear errors if required tool/free-joint/actuator metadata cannot be found.

### 3. Implement concrete capability probing and close/settle in Warp

Capability probing order is mandatory:

1. generate/prepare MJCF;
2. load `mujoco.MjModel` and create `mujoco.MjData`;
3. create real `warp_model` and real `warp_data`;
4. call capability probing on the concrete `warp_data` object, not only on the module:
   ```python
   inspect_warp_batch_capabilities(
       mjw,
       warp_model=warp_model,
       warp_data=warp_data,
       nworld=nworld,
   )
   ```
5. proceed only if `supports_true_fixed_grasp_batching` is true.

A module-only call such as `inspect_warp_batch_capabilities(mjw)` is intentionally conservative and is not sufficient for PR13.

Add minimal batched close/settle support:

1. copy `qpos_init/qvel_init/ctrl_init/xfrc_zero` to `warp_data`;
2. run a configurable number of settle steps;
3. capture settled `qpos/qvel/ctrl` for restoration before wrench tests.

Use existing `EvaluationConfig` fields where possible.

Do not silently ignore missing fields. If `qpos`, `qvel`, `ctrl`, or `xfrc_applied` cannot be set per world, raise a clear `NotImplementedError` or backend capability error.

### 4. Implement 12-direction batched wrench testing

Implement a minimal batched version of the CPU wrench test.

Requirements:

- use the same 12 Cartesian force/torque directions as CPU reference via `WRENCH_DIRECTIONS`;
- restore settled state before each direction;
- clear `xfrc_applied` before each direction;
- apply external wrench to the tool body through per-world `xfrc_applied`;
- step all worlds together;
- detect failure using the same translation and rotation thresholds as CPU reference;
- compute stable steps and normalized score per world;
- aggregate wrench results with the existing scoring helper where possible;
- return one result per input grasp.

Performance note:

- It is acceptable in this PR to read back tool pose to host every step or every small interval if that is the safest implementation.
- Do not prematurely optimize with custom Warp kernels unless already straightforward.
- Correctness and transparency are more important than speed in PR13.

### 5. Wire into `MujocoWarpBackend.evaluate_grasps_batch`

Update:

```python
MujocoWarpBackend.evaluate_grasps_batch(...)
```

so that it:

- returns `[]` for empty grasp lists before constructing GPU state;
- builds the Warp scene bundle for each one-design/one-tool evaluation path;
- probes runtime capabilities on the concrete `warp_data`;
- rejects unsupported environments clearly;
- chunks input grasps by `nworld`;
- evaluates each chunk with true Warp batched simulation;
- returns a list of `GraspEvaluation` objects compatible with the existing dataclass/schema;
- never calls CPU backend and labels the result as Warp;
- never silently falls back to CPU;
- retains experimental metadata through existing batch-level metadata paths.

Metadata constraints:

- Do not add fields to `GraspEvaluation` in this PR unless absolutely necessary.
- Prefer exposing Warp metadata through `MujocoWarpBackend.last_batch_metadata` and any existing CLI/tool summary metadata path.
- Keep per-grasp results compatible with the current `GraspEvaluation` dataclass/schema.

Suggested batch-level metadata:

```json
{
  "backend": "mujoco_warp",
  "experimental": true,
  "score_semantics": "experimental_non_equivalent",
  "true_batched_scoring": true,
  "per_world_state_init": true,
  "wrench_directions": 12,
  "include_in_multifidelity": false,
  "sequential_fallback": false
}
```

### 6. Keep single-grasp behavior conservative

Do not make `evaluate_grasp()` pretend to be CPU-equivalent.

Acceptable options:

- keep it `NotImplementedError`;
- or implement it by calling `evaluate_grasps_batch(..., [grasp], ...)` only if it is explicitly marked experimental and does not silently use CPU.

Prefer the option that best matches existing code style and tests.

### 7. Add tests

CPU-only tests:

1. `evaluate_grasps_batch` rejects unavailable `mujoco_warp` clearly at the backend level.
2. `evaluate_grasps_batch` rejects environments without true per-world state support.
3. Capability probing is called after a concrete `warp_data` object exists.
4. Batched initial-state construction works with small fake/mock model metadata.
5. Empty grasp lists return an empty list without constructing GPU state.
6. Chunking respects `nworld`.
7. Existing CLI/orchestration tests remain compatible with structured failed `GraspEvaluation` results when backend exceptions are intentionally caught at a higher level.

Optional GPU tests gated by `RUN_GPU_TESTS=1` and `pytest.mark.gpu`:

1. one design;
2. one tool, preferably the simplest existing tool asset;
3. small number of grasps, for example 2–4;
4. verify result count equals input grasp count;
5. verify finite scores;
6. verify metadata marks results as experimental non-equivalent;
7. verify no CPU fallback happened.

Test policy:

- CPU-only tests must not require `mujoco_warp`, CUDA, or a GPU.
- Backend-level unsupported-environment tests may assert `NotImplementedError` or a project-specific backend capability error.
- CLI/orchestration-level tests may expect structured failed `GraspEvaluation` objects if the existing orchestration layer catches backend exceptions.
- Do not mark the Warp implementation as CPU-equivalent or include it in multifidelity scoring.

## Validation

CPU-only:

```bash
pytest -q
python3 scripts/evaluate_design_batch_warp.py --help
```

Optional GPU:

```bash
python3 -m pip install -e ".[warp]"
RUN_GPU_TESTS=1 pytest -q -m gpu
python3 scripts/evaluate_design_batch_warp.py \
  --design-dir outputs/designs \
  --results-dir outputs/warp_results_pr13 \
  --tools hammer \
  --n-grasp-trials 4 \
  --nworld 4 \
  --require-warp \
  --overwrite
```

If the script interface has changed, adjust the example to the current `scripts/evaluate_design_batch_warp.py --help` output before committing.

## Out of scope

Do not implement:

- CPU-equivalence claims;
- multi-tool orchestration beyond what already exists;
- TPE/Optuna batching;
- graph capture performance tuning;
- multifidelity inclusion;
- surrogate training;
- ROS2;
- Isaac Sim;
- MJX/JAX/autodiff.

## Acceptance criteria

This PR is acceptable if:

1. A real MuJoCo Warp batch path exists behind `evaluate_grasps_batch`.
2. The path initializes one fixed grasp per world.
3. The path probes concrete `warp_data` for per-world write support before scoring.
4. The path runs close/settle and 12-direction wrench testing.
5. Results are explicitly marked experimental and non-equivalent through batch-level metadata.
6. `GraspEvaluation` remains schema-compatible with existing code unless a justified schema migration is included.
7. CPU-only CI still passes without `mujoco_warp`.
8. Optional GPU smoke tests can execute a tiny batch without CPU fallback.
