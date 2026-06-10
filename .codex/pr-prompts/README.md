# Codex Prompt Bundle: PR12–PR16 — MuJoCo Warp GPU Backend Pipeline

Repository target:

```text
Lunar-boy/handcdo_mujoco
```

This bundle continues after the completed PR11 sequence. PR11 introduced an experimental MuJoCo Warp backend path, optional dependency isolation, fixed-random batch orchestration scaffolding, a dedicated experimental CLI, and CPU-vs-Warp comparison documentation. PR12–PR16 should now move from “safe skeleton” to “real GPU backend pipeline integration” without contaminating the CPU reference backend or multifidelity results.

## Global semantic rules

These rules apply to every PR in this bundle.

1. CPU MuJoCo remains the reference backend.
2. MuJoCo Warp remains optional, experimental, and never the default backend.
3. Default installation and default tests must not require CUDA, H100, JAX, MJX, or `mujoco_warp`.
4. Never silently fall back to CPU when the user explicitly requested `mujoco_warp`.
5. Never report fake GPU speedups.
6. Never claim CPU-equivalent scoring until exact CPU wrench semantics are implemented and validated by regression tests.
7. Until equivalence is proven, every Warp output must retain:

```json
"score_semantics": "experimental_non_equivalent"
```

8. Do not use Warp results in multifidelity `best_available_score` unless a later PR explicitly adds a guarded import path and validation proof.
9. Do not change existing CPU scoring behavior.
10. Do not invent MuJoCo Warp APIs. Use only verified APIs from existing code, official imports, runtime introspection, or guarded optional GPU tests.
11. Prefer small, reviewable changes. Each PR must be independently testable on CPU-only CI.
12. If GPU-specific validation is added, gate it behind an environment variable such as `RUN_GPU_TESTS=1` and pytest markers.

## Recommended order

```text
PR12 -> PR13 -> PR14 -> PR15 -> PR16
```

## Bundle summary

| Stage | Suggested prompt file | Purpose | Main validation |
|---|---|---|---|
| PR12 | `12-warp-real-capability-probe.md` | Replace static conservative Warp capability probing with runtime model/data probing and per-world state write smoke tests | `pytest -q`; optional `RUN_GPU_TESTS=1 pytest -q -m gpu` |
| PR13 | `13-true-warp-fixed-random-scoring.md` | Implement minimal true MuJoCo Warp fixed-random batch scoring for one design/tool path | CPU-only tests + optional GPU smoke scoring |
| PR14 | `14-cpu-warp-validation-harness.md` | Add CPU-vs-Warp validation harness measuring score delta, rank drift, and failure-direction mismatch | CLI help + deterministic comparison fixtures |
| PR15 | `15-warp-multitool-throughput.md` | Extend Warp batch evaluation to multi-tool screening and throughput reporting | Experimental CLI output schema + optional GPU benchmark |
| PR16 | `16-experimental-warp-screening-pipeline.md` | Integrate Warp as an experimental high-throughput screening stage feeding CPU high-fidelity evaluation | Pipeline-level JSON schema and guarded tests |
