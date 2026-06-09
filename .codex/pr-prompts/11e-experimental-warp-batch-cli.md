# Codex Prompt: PR11-e — Experimental MuJoCo Warp Batch Evaluation CLI

You are working in the `Lunar-boy/handcdo_mujoco` repository.

Start from latest `main` after PR11-a through PR11-d have merged:

```bash
git checkout main
git pull origin main
git checkout -b pr11e-experimental-warp-batch-cli
```

## Goal

Add a dedicated experimental CLI for MuJoCo Warp batched random-grasp evaluation.

This CLI must not change existing CPU evaluation scripts or default backend behavior.

It must be explicit, conservative, and safe on CPU-only systems.

## Required changes

Add:

```text
scripts/evaluate_design_batch_warp.py
```

Add or update tests:

```text
tests/test_evaluate_design_batch_warp_cli.py
tests/test_warp_batch_result_schema.py
```

Update README with a short experimental section.

Do not modify the existing CPU evaluation CLI except for import-safe reuse of helpers.

## CLI example

Support a command like:

```bash
python3 scripts/evaluate_design_batch_warp.py \
  --design-dir outputs/designs \
  --results-dir outputs/warp_results \
  --config configs/eval_fast.yaml \
  --tools hammer,spoon,knife \
  --n-grasp-trials 128 \
  --sampler random \
  --nworld 64 \
  --nconmax 64 \
  --njmax 128 \
  --seed 0
```

## Supported arguments

Required or defaulted args:

```text
--design-dir          default outputs/designs
--design-ids          optional text file with one design id per line
--results-dir         required
--config              default configs/eval_fast.yaml
--tools               default hammer,spoon,knife
--n-grasp-trials      default 64
--sampler             choices random; default random
--nworld              default 64
--nconmax             default 64
--naconmax            optional
--njmax               default 128
--warmup-steps        default 0
--capture-graph       action flag, default false
--seed                default 0
--max-designs         optional
--require-warp        action flag
--overwrite           action flag, default false
--fail-fast           action flag, default false
```

Do not use `--continue-on-error` as a `store_true` flag with default true.

Use `--fail-fast` instead.

## CLI behavior

- `--help` must work without importing `mujoco_warp`.
- Missing `mujoco_warp` must produce a clear error only when evaluation is attempted.
- If `--require-warp` is false and Warp is unavailable, either fail cleanly with a helpful message or write a skipped availability result.
- Do not crash with an obscure raw import traceback.
- Do not silently overwrite existing results. Require `--overwrite`.
- Write one result JSON per design.
- Keep output shape compatible with existing result collection where practical.
- Do not commit generated outputs.

## Sampler policy

For this stage, support only:

```text
--sampler random
```

If the user passes `--sampler tpe`, argparse should reject it or the script should fail clearly:

```text
MuJoCo Warp batched evaluation currently supports sampler=random only.
TPE is sequential/adaptive and is intentionally out of scope for this PR.
```

Do not silently run sequential TPE through the Warp backend.

## Output schema

For each design result JSON, use a shape like:

```json
{
  "design_id": "...",
  "parameters": {},
  "hand_score": 0.0,
  "tool_results": [],
  "failed": false,
  "backend": "mujoco_warp",
  "experimental": true,
  "score_semantics": "experimental_non_equivalent",
  "warp_metadata": {}
}
```

`warp_metadata` should include:

```json
{
  "nworld": 64,
  "nconmax": 64,
  "naconmax": null,
  "njmax": 128,
  "warmup_steps": 0,
  "capture_graph": false,
  "batch_size": 64,
  "num_grasps": 128,
  "num_chunks": 2,
  "seconds_total": 0.0,
  "grasps_per_second": null,
  "world_steps_per_second": null,
  "failure_count": 0,
  "sequential_fallback": false,
  "mjcf_rewrites": []
}
```

Per-tool result should preserve the existing shape where practical:

```json
{
  "tool": "hammer",
  "best_score": 0.0,
  "best_grasp": {},
  "trials": []
}
```

Each trial should include:

- grasp parameters;
- score;
- failed flag;
- error if failed;
- wrench results if implemented;
- backend metadata if useful.

## Result safety

Do not write:

```json
"score_semantics": "intended_cpu_equivalent"
```

from this CLI.

Do not mix Warp output with CPU reference output in the same results directory unless the filename or JSON makes the backend explicit.

Recommended filename pattern:

```text
<design_id>.mujoco_warp.experimental.json
```

or:

```text
<design_id>.json
```

inside a clearly named `warp_results` directory.

## README update

Add a short section:

```markdown
## Experimental MuJoCo Warp batch evaluation
```

Explain:

- the backend is optional and experimental;
- it is intended for NVIDIA GPU throughput experiments, especially H100-class systems;
- it is most useful for fixed random-grasp batch evaluation;
- it is not the default backend;
- CPU MuJoCo remains the reference implementation;
- Warp scores should not be used as final scientific conclusions until CPU-vs-Warp comparisons are stable;
- default installation and tests remain CPU-only.

Include install example:

```bash
python3 -m pip install -e ".[test]"
python3 -m pip install -e ".[warp]"
```

Include smoke example:

```bash
python3 scripts/evaluate_design_batch_warp.py \
  --design-dir outputs/designs \
  --results-dir outputs/warp_results_smoke \
  --config configs/eval_fast.yaml \
  --tools hammer \
  --n-grasp-trials 8 \
  --sampler random \
  --nworld 8 \
  --seed 0
```

Do not claim speedup unless measured.

## Tests

CPU-only tests:

1. `python3 scripts/evaluate_design_batch_warp.py --help` works without `mujoco_warp`.
2. Parser accepts valid random-sampler arguments.
3. Parser rejects or clearly fails for unsupported TPE mode.
4. Existing result path is not overwritten unless `--overwrite` is passed.
5. Output schema helper includes required fields.
6. Missing Warp produces a helpful error or skipped result.
7. `pytest -q` passes without CUDA, H100, JAX, MJX, or `mujoco_warp`.

## Validation

Run:

```bash
pytest -q
python3 scripts/evaluate_design_batch_warp.py --help
python3 scripts/benchmark_mujoco_warp.py --help
```

Optional GPU validation:

```bash
python3 -m pip install -e ".[warp]"
python3 scripts/evaluate_design_batch_warp.py \
  --design-dir outputs/designs \
  --results-dir outputs/warp_results_smoke \
  --config configs/eval_fast.yaml \
  --tools hammer \
  --n-grasp-trials 8 \
  --sampler random \
  --nworld 8 \
  --nconmax 64 \
  --njmax 128 \
  --seed 0 \
  --overwrite
```

## Out of scope

Do not implement:

- CPU-vs-Warp comparison helper;
- exact CPU-equivalent scoring;
- TPE batching;
- Slurm production templates;
- JAX/MJX/autodiff;
- Isaac Sim;
- ROS;
- RL or policy learning.

## Success criteria

This stage is successful if:

1. A dedicated experimental Warp batch CLI exists.
2. `--help` works CPU-only.
3. Output schema is explicit and conservative.
4. Existing CPU workflows remain unchanged.
5. Default tests pass without GPU dependencies.
