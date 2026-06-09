# Optimized PR11 Prompt Bundle: Experimental Batched MuJoCo Warp Backend

Repository target:

```text
Lunar-boy/handcdo_mujoco
```

This bundle decomposes PR11 into six small, reviewable Codex tasks.

## Why this decomposition exists

The full PR11 goal is to add an experimental MuJoCo Warp backend for batched fixed random-grasp evaluation. That is high-risk if implemented as one monolithic PR, because Codex can easily create:

- fake batching that loops over single-grasp CPU evaluations;
- silent CPU fallback mislabeled as MuJoCo Warp;
- invented MuJoCo Warp API calls;
- outputs that look CPU-equivalent but are not physically or semantically equivalent;
- contamination of existing CPU reference results and multifidelity ranking.

Therefore, this PR11 sequence is split into six stages:

| Stage | File | Purpose | Test |
|---|---|---|---|
| PR11-a | `11a-warp-utils-batched-protocol.md` | Add batched protocol and reusable Warp utility extraction only | pytest -q python3 scripts/benchmark_mujoco_warp.py --help |
| PR11-b | `11b-lazy-mujoco-warp-backend-skeleton.md` | Add lazy experimental backend skeleton and registry alias | pytest -q python3 scripts/benchmark_mujoco_warp.py --help |
| PR11-c | `11c-fixed-random-grasp-batch-orchestration.md` | Add deterministic fixed-grasp batch orchestration with dummy/CPU-testable backend | pytest -q python3 scripts/benchmark_mujoco_warp.py --help |
| PR11-d | `11d-minimal-experimental-warp-backend.md` | Implement minimal true experimental MuJoCo Warp batch backend | pytest -q python3 scripts/benchmark_mujoco_warp.py --help |
| PR11-e | `11e-experimental-warp-batch-cli.md` | Add dedicated experimental CLI and JSON output schema | pytest -q python3 scripts/benchmark_mujoco_warp.py --help python3 scripts/evaluate_design_batch_warp.py --help |
| PR11-f | `11f-cpu-warp-comparison-docs-validation.md` | Add CPU-vs-Warp comparison, docs, and optional GPU validation | pytest -q python3 scripts/benchmark_mujoco_warp.py --help python3 scripts/evaluate_design_batch_warp.py --help |

## Global semantic rules

These rules apply to every stage:

1. CPU MuJoCo remains the reference backend.
2. MuJoCo Warp is optional, experimental, and never the default backend.
3. Default installation and default tests must not require CUDA, H100, JAX, MJX, or `mujoco_warp`.
4. Never silently fall back to CPU when the user explicitly requested `mujoco_warp`.
5. Never report fake GPU speedups.
6. Never claim CPU-equivalent scoring unless exact CPU wrench semantics are implemented and regression-tested.
7. Until that later equivalence proof exists, all Warp outputs must use:

```json
"score_semantics": "experimental_non_equivalent"
```

8. Do not use Warp results in multifidelity `best_available_score`.
9. Do not modify existing CPU CLIs except for import-safe helper reuse.
10. Do not invent MuJoCo Warp APIs. Use only verified APIs from existing PR10 code, official imports, or runtime introspection.
11. Any Warp result must be excluded from CPU/multifidelity/surrogate result pools unless a later PR explicitly adds a guarded import path and comparison validation.
## Suggested usage

Copy one prompt at a time into Codex. Do not ask Codex to implement all six prompts at once.

Recommended order:

```text
11a -> 11b -> 11c -> 11d -> 11e -> 11f
```

After each stage:

```bash
pytest -q
python3 scripts/benchmark_mujoco_warp.py --help
```

For stages that add the experimental CLI:

```bash
python3 scripts/evaluate_design_batch_warp.py --help
```

For optional GPU validation, run only on a GPU node where `mujoco_warp` is installed:

```bash
python3 -m pip install -e ".[warp]"
RUN_GPU_TESTS=1 pytest -q -m gpu
```

## Important mapping note

If the repository already contains earlier PR11 prompts, this optimized bundle supersedes them conceptually. In particular, this bundle separates:

- batch orchestration;
- actual Warp backend stepping;
- CLI/output schema;
- comparison and validation.

That separation is intentional and should reduce implementation risk.
