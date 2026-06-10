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
map fixed random grasps into per-world qpos/qvel/ctrl
close/settle hand
run 12-direction wrench tests
return GraspEvaluation-like results with experimental Warp metadata
```

## Files to inspect first

Inspect the current implementation before editing:

```text
handcdo/backends/mujoco_warp.py
handcdo/backends/mujoco_cpu.py
handcdo/mujoco_eval.py
handcdo/grasp_sampling.py
handcdo/geometry_config.py
handcdo/design_space.py
handcdo/mjcf_builder.py
handcdo/tools.py
handcdo/backends/*warp*.py
scripts/evaluate_design_batch_warp.py
tests/test_*warp*.py
tests/test_*mujoco*.py
```

Find the CPU reference logic for:

- setting tool free-joint pose;
- applying hand actuator controls;
- closing/settling the hand;
- saving/restoring settled state;
- applying 12 wrench directions;
- computing final normalized score.

Do not duplicate semantics blindly. Extract shared helper logic where safe.

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
- write tool position and orientation into the tool free joint;
- write actuator controls according to the same closure semantics as the CPU evaluator;
- zero velocities;
- zero external forces;
- preserve CPU reference semantics as much as possible;
- validate batch size does not exceed `nworld`;
- return clear errors if required tool/free-joint/actuator metadata cannot be found.

### 3. Implement close/settle in Warp

Add minimal batched close/settle support:

1. copy `qpos_init/qvel_init/ctrl_init/xfrc_zero` to `warp_data`;
2. run a configurable number of settle steps;
3. capture settled `qpos/qvel/ctrl` for restoration before wrench tests.

Use existing `EvaluationConfig` fields where possible.

Do not silently ignore missing fields. If `qpos`, `qvel`, `ctrl`, or `xfrc_applied` cannot be set per world, raise a clear `NotImplementedError` or backend capability error.

### 4. Implement 12-direction batched wrench testing

Implement a minimal batched version of the CPU wrench test.

Requirements:

- use the same 12 Cartesian force/torque directions as CPU reference;
- restore settled state before each direction;
- apply external wrench to the tool body through per-world `xfrc_applied`;
- step all worlds together;
- detect failure using the same translation and rotation thresholds as CPU reference;
- compute stable steps and normalized score per world;
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

- checks runtime capabilities;
- rejects unsupported environments clearly;
- chunks input grasps by `nworld`;
- evaluates each chunk with true Warp batched simulation;
- returns a list of evaluation objects matching the existing expected type or schema;
- never calls CPU backend and labels the result as Warp;
- never silently falls back to CPU;
- retains experimental metadata.

Suggested metadata:

```json
{
  "backend": "mujoco_warp",
  "experimental": true,
  "score_semantics": "experimental_non_equivalent",
  "true_batched_scoring": true,
  "per_world_state_init": true,
  "wrench_directions": 12,
  "include_in_multifidelity": false
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

1. `evaluate_grasps_batch` rejects unavailable `mujoco_warp` clearly.
2. `evaluate_grasps_batch` rejects environments without true per-world state support.
3. Batched initial-state construction works with small fake/mock model metadata.
4. Empty grasp lists return an empty list without constructing GPU state.
5. Chunking respects `nworld`.

Optional GPU tests gated by `RUN_GPU_TESTS=1`:

1. one design;
2. one tool, preferably the simplest existing tool asset;
3. small number of grasps, for example 2–4;
4. verify result count equals input grasp count;
5. verify finite scores;
6. verify metadata marks results as experimental non-equivalent;
7. verify no CPU fallback happened.

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
python3 scripts/evaluate_design_batch_warp.py --design preset --tool hammer --num-grasps 4 --backend mujoco_warp --nworld 4
```

Adjust CLI arguments to the actual script interface.

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
3. The path runs close/settle and 12-direction wrench testing.
4. Results are explicitly marked experimental and non-equivalent.
5. CPU-only CI still passes without `mujoco_warp`.
6. Optional GPU smoke tests can execute a tiny batch without CPU fallback.
