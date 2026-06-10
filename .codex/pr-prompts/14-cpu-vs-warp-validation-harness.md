# Codex Prompt: PR14 — CPU-vs-Warp Validation Harness

You are working in the `Lunar-boy/handcdo_mujoco` repository.

Start from latest `main` after PR13 has merged:

```bash
git checkout main
git pull origin main
git checkout -b pr14-cpu-warp-validation-harness
```

## Goal

Add a validation harness that compares CPU MuJoCo reference scoring against experimental MuJoCo Warp batch scoring on fixed deterministic samples.

This PR must not claim equivalence. It should quantify differences.

The output should help answer:

1. Are Warp scores finite and stable?
2. How large are score deltas versus CPU?
3. Does Warp preserve ranking among fixed grasps?
4. Which wrench directions disagree?
5. Are discrepancies due to known experimental limitations?

## Files to inspect first

```text
handcdo/backends/mujoco_cpu.py
handcdo/backends/mujoco_warp.py
handcdo/mujoco_eval.py
handcdo/grasp_sampling.py
scripts/evaluate_design_batch_warp.py
scripts/benchmark_mujoco_warp.py
tests/test_*warp*.py
tests/test_*validation*.py
README.md
```

## Required changes

### 1. Add comparison data structures

Create a small dataclass or serializable dict schema for CPU-vs-Warp comparison:

```python
@dataclass
class CpuWarpComparisonResult:
    design_id: str
    tool_name: str
    num_grasps: int
    cpu_backend: str
    warp_backend: str
    score_semantics: str
    max_abs_score_delta: float | None
    mean_abs_score_delta: float | None
    spearman_rank_correlation: float | None
    topk_overlap: dict[str, float]
    failure_direction_mismatch_rate: float | None
    warp_metadata: dict[str, Any]
    warnings: list[str]
```

Use existing result object conventions if available.

### 2. Add deterministic sample generation

Implement a helper to generate fixed samples:

```python
def make_validation_grasps(
    *,
    seed: int,
    num_grasps: int,
    tool_name: str,
) -> list[GraspParams]:
    ...
```

Requirements:

- deterministic with seed;
- independent of GPU availability;
- compatible with both CPU and Warp backends;
- small enough for CI/gpu smoke tests.

### 3. Add comparison logic

Implement:

```python
def compare_cpu_vs_warp_fixed_grasps(
    design: HandDesign,
    tool_name: str,
    grasps: list[GraspParams],
    config: EvaluationConfig,
    *,
    geometry_config: GeometryConfig | None = None,
    tool_assets_dir: str | Path = "assets/tools",
    warp_backend_kwargs: dict[str, Any] | None = None,
) -> CpuWarpComparisonResult:
    ...
```

Requirements:

- run CPU reference backend on exactly the same grasps;
- run Warp backend on exactly the same grasps;
- compare score arrays;
- compute mean/max absolute score deltas;
- compute rank correlation if enough samples exist;
- compute top-k overlap for small k values, for example 1, 3, 5 where valid;
- compare failure-direction details if the result schema exposes them;
- include warnings when a metric cannot be computed;
- never treat Warp as equivalent by default.

### 4. Add or extend CLI

Add a dedicated CLI or extend the existing experimental CLI with a comparison mode.

Suggested script:

```text
scripts/compare_cpu_warp_fixed_grasps.py
```

Suggested usage:

```bash
python3 scripts/compare_cpu_warp_fixed_grasps.py \
  --design preset \
  --tool hammer \
  --num-grasps 8 \
  --seed 0 \
  --nworld 8 \
  --out results/cpu_warp_validation.json
```

Requirements:

- `--help` works without `mujoco_warp`;
- running without `mujoco_warp` fails clearly only when actual comparison execution is requested;
- output JSON is deterministic except for timing fields;
- output includes `experimental_non_equivalent`.

### 5. Add docs

Update README or a dedicated docs file to explain:

- CPU remains the reference;
- Warp validation is experimental;
- score deltas are expected;
- how to run optional GPU validation;
- why Warp results are excluded from multifidelity ranking.

### 6. Add tests

CPU-only tests:

1. CLI `--help` works.
2. deterministic grasp generation is reproducible.
3. comparison metric functions work on synthetic CPU/Warp score arrays.
4. rank metrics handle ties and small sample sizes.
5. JSON schema includes required fields.
6. missing Warp dependency produces clear runtime message when actual comparison is requested.

Optional GPU tests gated by `RUN_GPU_TESTS=1`:

1. run tiny CPU-vs-Warp comparison, for example 2–4 grasps;
2. verify finite CPU and Warp scores;
3. verify output JSON exists and includes diagnostic fields;
4. do not assert tight equality unless empirically justified.

## Validation

CPU-only:

```bash
pytest -q
python3 scripts/compare_cpu_warp_fixed_grasps.py --help
```

Optional GPU:

```bash
python3 -m pip install -e ".[warp]"
RUN_GPU_TESTS=1 pytest -q -m gpu
python3 scripts/compare_cpu_warp_fixed_grasps.py --design preset --tool hammer --num-grasps 4 --seed 0 --nworld 4 --out /tmp/cpu_warp_validation.json
```

## Out of scope

Do not implement:

- automatic equivalence thresholding;
- using Warp score in final ranking;
- multifidelity best score integration;
- TPE/Optuna batch optimization;
- graph capture tuning;
- surrogate modeling;
- ROS2;
- Isaac Sim.

## Acceptance criteria

This PR is acceptable if:

1. Users can produce a CPU-vs-Warp comparison JSON on a GPU machine.
2. The comparison uses identical fixed grasps.
3. Metrics quantify score and ranking differences.
4. CPU-only tests pass.
5. Documentation clearly says Warp is experimental and non-equivalent.
