# Codex Prompt: PR15 — Multi-Tool Warp Batch Evaluation and Throughput Tuning

You are working in the `Lunar-boy/handcdo_mujoco` repository.

Start from latest `main` after PR14 has merged:

```bash
git checkout main
git pull origin main
git checkout -b pr15-warp-multitool-throughput
```

## Goal

Extend experimental MuJoCo Warp batch evaluation from one design/tool validation to multi-tool design screening, and add transparent throughput diagnostics.

This PR should make Warp useful as a high-throughput experimental evaluator while still keeping CPU as the reference and excluding Warp from multifidelity best-score aggregation.

## Files to inspect first

```text
scripts/evaluate_design_batch_warp.py
scripts/compare_cpu_warp_fixed_grasps.py
scripts/benchmark_mujoco_warp.py
handcdo/backends/mujoco_warp.py
handcdo/backends/mujoco_cpu.py
handcdo/tools.py
handcdo/grasp_sampling.py
handcdo/optimization.py
handcdo/multifidelity.py
README.md
tests/test_*warp*.py
tests/test_*cli*.py
```

Use actual existing module names. Do not add duplicate pipeline code if an equivalent orchestration helper already exists.

## Required changes

### 1. Add multi-tool experimental batch screening helper

Implement a helper such as:

```python
def evaluate_design_multi_tool_warp_batch(
    design: HandDesign,
    tool_names: list[str],
    *,
    num_grasps_per_tool: int,
    seed: int,
    config: EvaluationConfig,
    geometry_config: GeometryConfig | None = None,
    tool_assets_dir: str | Path = "assets/tools",
    warp_backend_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ...
```

Requirements:

- evaluate each tool using fixed deterministic random grasps;
- use the experimental Warp backend only when explicitly requested;
- return per-tool scores and metadata;
- aggregate an experimental screening score separately from CPU/multifidelity scores;
- keep all outputs clearly marked as experimental non-equivalent;
- handle per-tool failures without hiding them.

Suggested output shape:

```json
{
  "backend": "mujoco_warp",
  "experimental": true,
  "score_semantics": "experimental_non_equivalent",
  "include_in_multifidelity": false,
  "design_id": "...",
  "tools": {
    "hammer": {
      "num_grasps": 64,
      "mean_score": 0.0,
      "best_score": 0.0,
      "error": null,
      "metadata": {}
    }
  },
  "experimental_screening_score": 0.0,
  "throughput": {
    "total_world_steps": 0,
    "wall_time_sec": 0.0,
    "world_steps_per_second": 0.0,
    "grasps_per_second": 0.0,
    "nworld": 64,
    "chunks": 1
  }
}
```

### 2. Add throughput measurement

Add timing around:

- model/MJCF build;
- Warp model/data creation;
- close/settle;
- wrench testing;
- total wall time.

Requirements:

- report timings honestly;
- do not compare to CPU unless explicitly requested by PR14 comparison CLI;
- do not claim speedup unless both CPU and Warp timings are measured in the same command;
- keep timing fields optional or nullable if not available.

### 3. Add safe chunking and resource parameters

Support existing or new CLI/backend parameters:

```text
--nworld
--nconmax
--naconmax
--njmax
--warmup-steps
--capture-graph
--num-grasps-per-tool
--seed
--tools hammer spoon knife
```

Requirements:

- validate positive integer parameters;
- chunk grasps by `nworld`;
- report chunk count;
- fail clearly if a chunk cannot be evaluated;
- do not silently reduce sample count unless explicitly documented.

### 4. Optional graph capture support remains guarded

If the backend already has `capture_graph`, keep it guarded.

Requirements:

- default `capture_graph=False`;
- if graph capture is unsupported by the installed Warp version, raise a clear warning or diagnostic;
- do not make graph capture required for correctness;
- do not break CPU-only tests.

### 5. Extend experimental CLI

Update or add CLI functionality so users can run:

```bash
python3 scripts/evaluate_design_batch_warp.py \
  --design preset \
  --tools hammer spoon knife \
  --num-grasps-per-tool 64 \
  --seed 0 \
  --nworld 64 \
  --out results/warp_multitool_screen.json
```

Adjust argument names to match existing CLI conventions.

CLI requirements:

- `--help` works without GPU or `mujoco_warp`;
- execution requires explicit `mujoco_warp` path;
- output JSON includes schema version;
- output JSON never reuses the CPU/multifidelity score field names in a misleading way;
- output JSON marks itself as experimental.

### 6. Add tests

CPU-only tests:

1. CLI `--help` works.
2. parameter validation rejects invalid `nworld`, negative grasp counts, empty tools.
3. aggregation works on synthetic per-tool results.
4. throughput schema can be serialized.
5. experimental outputs are excluded from multifidelity fields.
6. missing `mujoco_warp` fails clearly on execution but not import/help.

Optional GPU tests gated by `RUN_GPU_TESTS=1`:

1. evaluate two tools with 2–4 grasps each;
2. verify result contains both tools;
3. verify finite scores or clear per-tool errors;
4. verify throughput fields are present;
5. verify no CPU fallback.

## Validation

CPU-only:

```bash
pytest -q
python3 scripts/evaluate_design_batch_warp.py --help
python3 scripts/compare_cpu_warp_fixed_grasps.py --help
```

Optional GPU:

```bash
python3 -m pip install -e ".[warp]"
RUN_GPU_TESTS=1 pytest -q -m gpu
python3 scripts/evaluate_design_batch_warp.py --design preset --tools hammer spoon --num-grasps-per-tool 4 --seed 0 --nworld 4 --out /tmp/warp_multitool_screen.json
```

## Out of scope

Do not implement:

- CPU-equivalent score claims;
- default optimizer backend changes;
- multifidelity `best_available_score` inclusion;
- surrogate training;
- Optuna/TPE parallelization;
- ROS2;
- Isaac Sim;
- MJX/JAX/autodiff.

## Acceptance criteria

This PR is acceptable if:

1. Multi-tool experimental Warp screening can run on a GPU machine.
2. Output JSON clearly separates experimental Warp screening score from CPU reference scores.
3. Throughput fields are honest and reproducible enough for diagnostics.
4. CPU-only CI remains green.
5. No silent CPU fallback exists.
