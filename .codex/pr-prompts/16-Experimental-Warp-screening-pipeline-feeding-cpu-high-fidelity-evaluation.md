# Codex Prompt: PR16 — Experimental Warp Screening Pipeline Feeding CPU High-Fidelity Evaluation

You are working in the `Lunar-boy/handcdo_mujoco` repository.

Start from latest `main` after PR15 has merged:

```bash
git checkout main
git pull origin main
git checkout -b pr16-experimental-warp-screening-pipeline
```

## Goal

Integrate MuJoCo Warp as an experimental high-throughput screening stage that can propose top-k candidates for subsequent CPU high-fidelity evaluation.

This PR should connect the whole pipeline without claiming Warp is CPU-equivalent.

The intended workflow is:

```text
sample many candidate hand designs
evaluate them with experimental Warp fixed-random multi-tool screening
select top-k by experimental screening score
re-evaluate selected top-k with CPU MuJoCo reference scoring
write a combined report preserving both score semantics
```

This PR should make Warp useful for narrowing the search space, not for replacing CPU reference evaluation.

## Files to inspect first

```text
handcdo/optimization.py
handcdo/multifidelity.py
handcdo/surrogate*.py
handcdo/backends/registry.py
handcdo/backends/mujoco_warp.py
handcdo/backends/mujoco_cpu.py
scripts/run_optuna_round.py
scripts/optimize_hand.py
scripts/evaluate_design_batch_warp.py
scripts/compare_cpu_warp_fixed_grasps.py
README.md
tests/test_*pipeline*.py
tests/test_*warp*.py
```

Use the actual filenames present in the repository.

## Required changes

### 1. Add an experimental screening pipeline module

Create or extend a module such as:

```text
handcdo/warp_screening.py
```

or use an existing pipeline module if one already exists.

Implement:

```python
def run_experimental_warp_screening_then_cpu_refine(
    *,
    design_sampler: Callable[..., Iterable[HandDesign]],
    num_candidates: int,
    top_k: int,
    tools: list[str],
    num_grasps_per_tool_warp: int,
    num_trials_cpu: int,
    seed: int,
    warp_backend_kwargs: dict[str, Any] | None = None,
    cpu_backend_kwargs: dict[str, Any] | None = None,
    config: EvaluationConfig | None = None,
    geometry_config: GeometryConfig | None = None,
    tool_assets_dir: str | Path = "assets/tools",
) -> dict[str, Any]:
    ...
```

Adapt the signature to existing design sampling and optimization abstractions.

Requirements:

- generate or accept candidate designs deterministically;
- run Warp screening only when explicitly requested;
- select top-k based only on `experimental_screening_score`;
- re-evaluate selected candidates with CPU reference backend;
- preserve separate score namespaces;
- never place Warp score into `best_available_score`;
- record failures and partial results clearly;
- make resumption possible if existing result utilities support it.

### 2. Add CLI for the experimental pipeline

Add a dedicated script rather than changing the default optimizer behavior.

Suggested script:

```text
scripts/run_experimental_warp_screening.py
```

Suggested usage:

```bash
python3 scripts/run_experimental_warp_screening.py \
  --num-candidates 128 \
  --top-k 8 \
  --tools hammer spoon knife \
  --num-grasps-per-tool-warp 64 \
  --num-trials-cpu 32 \
  --seed 0 \
  --nworld 64 \
  --out results/warp_screen_then_cpu_refine.json
```

CLI requirements:

- `--help` works without GPU or `mujoco_warp`;
- execution fails clearly if Warp is requested but unavailable;
- output JSON includes schema version and command metadata;
- CPU reference refinement is clearly separated from experimental Warp screening;
- no default script is changed to use Warp automatically.

### 3. Define combined report schema

The output should resemble:

```json
{
  "schema_version": "warp_screening_v1",
  "pipeline": "experimental_warp_screen_then_cpu_refine",
  "experimental": true,
  "warp_screening": {
    "backend": "mujoco_warp",
    "score_semantics": "experimental_non_equivalent",
    "include_in_multifidelity": false,
    "num_candidates": 128,
    "tools": ["hammer", "spoon", "knife"],
    "num_grasps_per_tool": 64,
    "ranked_candidates": [
      {
        "candidate_id": "candidate_0001",
        "experimental_screening_score": 0.0,
        "per_tool": {},
        "metadata": {}
      }
    ]
  },
  "cpu_refinement": {
    "backend": "mujoco_cpu",
    "score_semantics": "cpu_reference",
    "top_k": 8,
    "num_trials_cpu": 32,
    "ranked_candidates": [
      {
        "candidate_id": "candidate_0001",
        "cpu_reference_score": 0.0,
        "per_tool": {},
        "metadata": {}
      }
    ]
  },
  "analysis": {
    "rank_shift": [],
    "topk_overlap": {},
    "warnings": []
  }
}
```

Use actual existing score names where appropriate, but do not blur experimental Warp score with CPU reference score.

### 4. Add rank-shift diagnostics

Add simple diagnostics comparing Warp ranking and CPU-refined ranking for top-k candidates:

- rank shift per selected candidate;
- top-k overlap if CPU re-ranking uses the same candidate set;
- warning if CPU refinement fails for any candidate;
- warning if all Warp scores are identical or non-finite.

Do not over-interpret these metrics.

### 5. Keep multifidelity and surrogate paths guarded

If existing code has multifidelity or surrogate result pools, make sure:

- Warp screening results are not automatically inserted;
- CPU refinement results may use existing CPU result paths if compatible;
- any export clearly labels provenance;
- tests verify Warp results remain excluded from multifidelity `best_available_score`.

### 6. Add tests

CPU-only tests:

1. CLI `--help` works.
2. output schema construction works on synthetic results.
3. top-k selection works on synthetic experimental screening scores.
4. rank-shift diagnostics work.
5. invalid parameters are rejected:
   - `top_k <= 0`;
   - `top_k > num_candidates`;
   - empty tools;
   - negative grasp counts;
   - negative CPU trials.
6. missing Warp dependency fails clearly only when execution requires Warp.
7. multifidelity exclusion is preserved.

Optional GPU tests gated by `RUN_GPU_TESTS=1`:

1. run tiny screening:
   - 2–3 candidates;
   - 1–2 tools;
   - 2 grasps per tool;
   - top-k 1;
   - very small CPU refinement.
2. verify report contains both Warp and CPU sections.
3. verify CPU refinement was actually run for selected candidate(s).
4. verify Warp and CPU score semantics remain separate.

## Validation

CPU-only:

```bash
pytest -q
python3 scripts/run_experimental_warp_screening.py --help
python3 scripts/evaluate_design_batch_warp.py --help
python3 scripts/compare_cpu_warp_fixed_grasps.py --help
```

Optional GPU:

```bash
python3 -m pip install -e ".[warp]"
RUN_GPU_TESTS=1 pytest -q -m gpu
python3 scripts/run_experimental_warp_screening.py \
  --num-candidates 3 \
  --top-k 1 \
  --tools hammer \
  --num-grasps-per-tool-warp 2 \
  --num-trials-cpu 2 \
  --seed 0 \
  --nworld 2 \
  --out /tmp/warp_screen_then_cpu_refine.json
```

Adjust argument names to the actual codebase.

## Out of scope

Do not implement:

- making Warp the default backend;
- replacing CPU reference scores;
- CPU-equivalence claims;
- automatic multifidelity inclusion;
- full TPE/Optuna GPU parallelization;
- surrogate training on Warp scores unless explicitly marked experimental in a later PR;
- ROS2;
- Isaac Sim;
- MJX/JAX/autodiff.

## Acceptance criteria

This PR is acceptable if:

1. There is a dedicated experimental pipeline that screens with Warp and refines with CPU.
2. The pipeline connects design sampling, Warp multi-tool screening, top-k selection, CPU evaluation, and combined reporting.
3. Warp and CPU score semantics are separate in code and JSON.
4. Default CPU workflows are unchanged.
5. CPU-only CI passes without `mujoco_warp`.
6. Optional GPU tiny-run validation can exercise the whole pipeline.

---

# Final implementation guidance for Codex

Implement one PR at a time. Do not implement PR12–PR16 in one branch.

For every PR:

```bash
pytest -q
```

For CLI-affecting PRs:

```bash
python3 scripts/evaluate_design_batch_warp.py --help
```

For comparison/pipeline PRs:

```bash
python3 scripts/compare_cpu_warp_fixed_grasps.py --help
python3 scripts/run_experimental_warp_screening.py --help
```

For optional GPU validation only:

```bash
python3 -m pip install -e ".[warp]"
RUN_GPU_TESTS=1 pytest -q -m gpu
```

If a GPU test fails due to unavailable installed MuJoCo Warp APIs, do not fake the API. Instead, record a clear diagnostic and keep the backend experimental.