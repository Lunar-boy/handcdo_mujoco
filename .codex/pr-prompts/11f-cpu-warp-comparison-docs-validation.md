# Codex Prompt: PR11-f — CPU-vs-Warp Comparison, Docs, and Validation

You are working in the `Lunar-boy/handcdo_mujoco` repository.

Start from latest `main` after PR11-a through PR11-e have merged:

```bash
git checkout main
git pull origin main
git checkout -b pr11f-cpu-warp-comparison-docs-validation
```

## Goal

Add tooling and documentation to compare experimental MuJoCo Warp outputs against CPU MuJoCo reference outputs.

This PR does not make Warp scientifically authoritative.

It adds safety rails so future users can inspect:

- score differences;
- rank drift;
- failed trials;
- missing results;
- backend metadata;
- speed metrics when available.

## Required changes

Add:

```text
scripts/compare_cpu_warp_results.py
```

Add tests:

```text
tests/test_compare_cpu_warp_results.py
```

Update:

```text
README.md
```

Optionally add:

```text
docs/mujoco_warp_experimental.md
```

## Comparison tool behavior

Support a command like:

```bash
python3 scripts/compare_cpu_warp_results.py \
  --cpu-results-dir outputs/results \
  --warp-results-dir outputs/warp_results \
  --out outputs/warp_cpu_comparison.json
```

Supported args:

```text
--cpu-results-dir      required
--warp-results-dir     required
--out                  optional JSON output path
--tools                optional comma-separated subset
--top-k                default 20
--fail-on-missing      action flag, default false
--fail-on-rank-drift   optional float threshold
--fail-on-score-diff   optional float threshold
```

The script must be import-safe and CPU-only.

It must not import `mujoco_warp`.

## Input expectations

CPU results are reference results produced by the existing CPU workflow.

Warp results are experimental JSONs produced by:

```text
scripts/evaluate_design_batch_warp.py
```

The script should tolerate partial overlap:

- design exists in CPU only;
- design exists in Warp only;
- tool exists in one backend only;
- failed trials exist in either backend;
- metadata fields missing in older outputs.

## Output schema

Produce a JSON summary like:

```json
{
  "cpu_results_dir": "outputs/results",
  "warp_results_dir": "outputs/warp_results",
  "num_cpu_designs": 0,
  "num_warp_designs": 0,
  "num_matched_designs": 0,
  "missing_in_cpu": [],
  "missing_in_warp": [],
  "score_semantics": {
    "cpu": "reference",
    "warp": "experimental_non_equivalent"
  },
  "overall": {
    "mean_abs_score_diff": null,
    "median_abs_score_diff": null,
    "max_abs_score_diff": null,
    "rank_spearman": null,
    "rank_kendall": null,
    "top_k_overlap": null
  },
  "by_tool": {},
  "by_design": [],
  "warnings": []
}
```

Do not require SciPy unless it is already a dependency. If Spearman/Kendall are not available, implement a small local rank correlation helper or set those fields to `null` with a warning.

## Comparison metrics

Where possible, compute:

- absolute score difference;
- signed score difference;
- per-tool score difference;
- missing tool count;
- failure count difference;
- design rank under CPU;
- design rank under Warp;
- rank displacement;
- top-k overlap;
- optional Spearman correlation;
- optional Kendall correlation.

Use CPU score as reference.

Do not label Warp as equivalent.

## Safety warnings

The script should warn if:

- Warp result JSON lacks `"experimental": true`;
- Warp result JSON lacks `"score_semantics"`;
- Warp result JSON claims `"intended_cpu_equivalent"`;
- backend is not `"mujoco_warp"`;
- CPU result appears to be non-CPU;
- result schemas are inconsistent;
- too few matched designs exist for meaningful rank comparison.

Warnings should go into both stdout and output JSON.

## Exit behavior

Default behavior should not fail hard unless files are unreadable.

If `--fail-on-missing` is set, exit nonzero when matched designs are missing in either direction.

If `--fail-on-rank-drift` is set, exit nonzero when max or mean rank drift exceeds the threshold. Document exactly which criterion is used.

If `--fail-on-score-diff` is set, exit nonzero when max absolute score difference exceeds the threshold.

## README/docs update

Add or extend a section explaining:

```markdown
## Comparing CPU and MuJoCo Warp results
```

State clearly:

- CPU MuJoCo remains the reference backend.
- MuJoCo Warp is experimental.
- MuJoCo Warp scores are not final scientific conclusions.
- Warp can be useful for throughput exploration on H100-class systems.
- Any claimed speedup must include hardware, `nworld`, `nconmax`, `njmax`, number of grasps, and timing method.
- Any claimed score agreement must be backed by `compare_cpu_warp_results.py`.

Include example:

```bash
python3 scripts/compare_cpu_warp_results.py \
  --cpu-results-dir outputs/results \
  --warp-results-dir outputs/warp_results \
  --out outputs/warp_cpu_comparison.json
```

## Tests

Add CPU-only tests covering:

1. Comparison script `--help` works.
2. Empty directories produce a controlled summary or helpful error.
3. Matched fake CPU/Warp JSON files produce expected score difference.
4. Missing design appears in `missing_in_cpu` or `missing_in_warp`.
5. Warning is emitted if Warp result lacks `"experimental": true`.
6. Warning is emitted if Warp result claims `"intended_cpu_equivalent"`.
7. `--fail-on-missing` exits nonzero when expected.
8. No `mujoco_warp` import is required.

## Validation

Run:

```bash
pytest -q
python3 scripts/compare_cpu_warp_results.py --help
python3 scripts/evaluate_design_batch_warp.py --help
python3 scripts/benchmark_mujoco_warp.py --help
```

Optional manual validation, if sample outputs exist:

```bash
python3 scripts/compare_cpu_warp_results.py \
  --cpu-results-dir outputs/results \
  --warp-results-dir outputs/warp_results \
  --out outputs/warp_cpu_comparison.json
```

## Out of scope

Do not implement:

- new MuJoCo Warp physics;
- score equivalence claims;
- automatic promotion of Warp scores into CPU result pools;
- multifidelity integration;
- TPE batching;
- Slurm production templates;
- JAX/MJX/autodiff;
- Isaac Sim;
- ROS;
- RL or policy learning.

## Success criteria

This stage is successful if:

1. CPU-vs-Warp comparison tooling exists.
2. It is CPU-only and import-safe.
3. It detects missing results, score drift, rank drift, and unsafe metadata.
4. Documentation clearly prevents overclaiming.
5. `pytest -q` passes without GPU dependencies.
