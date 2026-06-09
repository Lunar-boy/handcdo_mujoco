# Codex Prompt: PR11f — Optional CPU-Equivalent Wrench Scoring for MuJoCo Warp

Use this prompt only after PR11a–PR11e are merged and stable.

Create a new branch:

```bash
git checkout main
git pull origin main
git checkout -b pr11f-warp-cpu-equivalent-wrench-scoring
```

## Goal

Attempt to make MuJoCo Warp scoring match the CPU MuJoCo reference wrench-scoring semantics.

This stage is optional and should be attempted only if the Warp backend already supports reliable per-world fixed-grasp initialization and reset.

## Important warning

Do not mark Warp results as CPU-equivalent unless the implementation actually matches CPU scoring semantics and regression comparisons are added.

The default should remain:

```json
"score_semantics": "experimental_non_equivalent"
```

Only change to:

```json
"score_semantics": "intended_cpu_equivalent"
```

when all requirements below are implemented and tested.

## CPU reference semantics to match

For each grasp:

1. Set tool pose.
2. Close hand.
3. Settle.
4. Save the post-settle state.
5. For each wrench direction:
   - reset to the saved post-settle state;
   - clear external wrench;
   - apply the same ramped Cartesian force/torque direction;
   - step simulation;
   - monitor tool translation from reference;
   - monitor tool rotation from reference;
   - terminate on threshold failure;
   - record normalized stable duration.
6. Aggregate all wrench direction results into the final grasp score.

Use the existing CPU implementation as the source of truth. Do not invent a new scoring formula.

## Required changes

Update:

```text
handcdo/backends/mujoco_warp.py
handcdo/warp_batch_eval.py
```

Add tests:

```text
tests/test_warp_wrench_score_semantics.py
```

Possibly update:

```text
scripts/compare_cpu_warp_results.py
README.md
```

## Regression comparison

Add small CPU-vs-Warp regression tests or optional comparison fixtures.

If real GPU is unavailable in CI, structure tests as:

- pure unit tests for scoring helper shape and aggregation;
- optional GPU tests skipped by default;
- comparison script tests using synthetic fixtures;
- documentation requiring H100/GPU validation before claims.

Do not require GPU for default `pytest -q`.

## Acceptance threshold

Do not choose arbitrary correlation claims.

If adding a numerical tolerance, document why. For early experimental comparison, report drift rather than enforcing strict equivalence unless deterministic behavior is demonstrated.

## Output semantics

If exact semantics are implemented but numerical drift may remain, use:

```json
"score_semantics": "intended_cpu_equivalent"
```

Also include:

```json
"score_equivalence_validated": false
```

unless actual CPU-vs-Warp regression data has been produced.

Only set:

```json
"score_equivalence_validated": true
```

when the comparison helper has verified drift against defined acceptance criteria.

## Tests

Default CPU-only tests:

1. Scoring metadata fields are present.
2. Unsupported Warp score equivalence paths fail clearly.
3. Comparison helper reports drift.
4. Optional GPU equivalence tests are skipped by default.
5. Existing CPU backend score semantics are unchanged.

Optional GPU tests:

```bash
RUN_GPU_TESTS=1 pytest -q -m gpu
```

## Validation

CPU-only:

```bash
pytest -q
python3 scripts/compare_cpu_warp_results.py --help
```

Optional GPU:

```bash
python3 -m pip install -e ".[warp]"
RUN_GPU_TESTS=1 pytest -q -m gpu
```

## Out of scope

Do not implement:

- TPE batching;
- autodiff;
- MJX/JAX backend;
- Isaac Sim;
- physical robot validation;
- Slurm production templates unless requested separately.

## Success criteria

This optional stage is successful if:

1. Warp wrench scoring matches CPU scoring structure.
2. Score semantics are accurately labeled.
3. Drift can be inspected with comparison tools.
4. No default CPU-only workflow is broken.
