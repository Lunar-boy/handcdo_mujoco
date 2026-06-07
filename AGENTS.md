# AGENTS.md

## Project

This repository is `handcdo_mujoco`, a CPU-only MuJoCo reproduction framework for the optimization infrastructure of:

> "Function-based Parametric Co-Design Optimization of Dexterous Hands" (arXiv:2604.27557)

The current scope is:

- Parametric hand design sampling.
- Primitive MJCF generation.
- MuJoCo CPU grasp and wrench evaluation.
- TPE-based co-design optimization.
- Slurm array execution for CPU HPC clusters.
- Result collection and aggregation.
- Random Forest / SHAP-based parameter analysis.

This repository is intentionally **not** an exact reproduction of the original Isaac Sim, Isaac Lab, UR5e, OptiTrack, hardware fabrication, or real robot setup.

The goal is to reproduce the paper's **optimization infrastructure** in a robust, inspectable, CPU-only MuJoCo framework suitable for research iteration and HPC batch execution.

## Reproduction boundary

Always distinguish between:

1. **Paper method**
   - What arXiv:2604.27557 describes.
   - Includes original simulator choices, hardware assumptions, fabrication workflow, and real-world validation.

2. **This repository**
   - A CPU-only MuJoCo approximation of the optimization infrastructure.
   - Uses simplified hand geometry, primitive MJCF assets, approximate grasp/wrench evaluation, and configurable contact geometry.

Do not claim numerical equivalence to the original paper unless explicitly backed by a benchmark, table, or validation experiment.

Acceptable approximations:

- MuJoCo CPU approximation of simulation-based grasp stability.
- Primitive tool and hand geometries.
- Configurable fingertip and palm contact geometry.
- TPE over mixed categorical, integer, and continuous hand parameters.
- Wrench-direction robustness tests.
- Random Forest / SHAP analysis over generated optimization results.

Out of scope by default:

- Isaac Sim or Isaac Lab integration.
- ROS integration.
- GPU-only simulation dependencies.
- Real robot control.
- UR5e-specific execution.
- OptiTrack-specific evaluation.
- Hardware fabrication workflow.
- Claims of exact physical reproduction.

## Non-negotiable constraints

- Do not add Isaac Sim, Isaac Lab, ROS, or GPU-only dependencies.
- Do not remove the existing MuJoCo CPU path.
- Do not implement MJX or MuJoCo-Warp unless the task explicitly asks for a benchmark-only prototype.
- Keep PRs small and reviewable.
- Preserve deterministic behavior when seeds are provided.
- Preserve existing CLI behavior where possible.
- Do not commit generated outputs, logs, caches, virtual environments, databases, or large mesh artifacts.
- Do not rewrite the whole repository.
- Do not silently change output schemas.
- Do not silently change scoring semantics.
- Do not rename public CLI flags unless the task explicitly asks for a breaking change.
- Do not reformat unrelated files.

## Engineering priorities

1. Preserve current behavior.
2. Improve test coverage and smoke-test reliability.
3. Maintain a clean simulator backend boundary.
4. Add configurable geometry modes only when needed by a concrete task.
5. Improve fingertip and palm contact geometry incrementally.
6. Add regression and multi-fidelity evaluation after the baseline is stable.
7. Benchmark MJX or MuJoCo-Warp only later and only as an optional path.

## Agent execution protocol

Before editing code, classify the task into one or more of the following categories.

### 1. Research-fidelity change

A change is research-fidelity related if it modifies the approximation to the paper's method.

Examples:

- Wrench scoring.
- Grasp search.
- Hand parameterization.
- Geometry or contact model.
- Tool models.
- Stability thresholds.
- Optimization objective.
- Search-space bounds.

Requirements:

- Explain which part of arXiv:2604.27557 the change is approximating.
- State whether the change is meant to improve paper fidelity or only improve engineering usability.
- Add or update tests when possible.
- Run a simulation smoke test if MuJoCo is available.
- Report whether baseline scores are expected to change.

### 2. Engineering change

A change is engineering related if it improves infrastructure without changing research semantics.

Examples:

- CLI robustness.
- Slurm execution.
- Result collection.
- Logging.
- Error handling.
- Packaging.
- Test skipping.
- Config parsing.
- File layout.
- Documentation.

Requirements:

- Preserve existing user-facing behavior unless explicitly requested.
- Keep changes minimal and localized.
- Prefer extending existing helpers over adding new subsystems.
- Run `pytest -q`.

### 3. Analysis change

A change is analysis related if it affects post-processing or interpretation.

Examples:

- Random Forest feature importance.
- SHAP analysis.
- Convergence plots.
- Benchmark comparison.
- Best-design extraction.
- CSV aggregation.

Requirements:

- Do not change simulation semantics.
- Keep analysis scripts able to run even when optional packages such as SHAP fail.
- Preserve input and output filenames unless the task explicitly asks for a new schema.
- Prefer additive output files over replacing existing outputs.

## Simplicity first

Prefer the smallest implementation that solves the current task.

Do not introduce a new abstraction layer unless at least two concrete implementations need it now.

Good patterns:

- Extend an existing config field.
- Add a small helper function.
- Add a focused test.
- Add a CLI flag with a backward-compatible default.
- Keep generated outputs under `outputs/`.

Bad patterns:

- Rewriting the hand model generator for a small geometry fix.
- Adding a new backend for speculative future use.
- Replacing MuJoCo CPU code with GPU-only code.
- Introducing Isaac Sim or ROS to appear closer to the original paper.
- Changing scoring semantics without documentation and tests.

## Surgical changes

Touch only the files required by the task.

Before editing, identify:

- The minimum file set.
- The expected behavior change.
- The verification command.

During editing:

- Do not reformat unrelated code.
- Do not rename unrelated variables.
- Do not reorganize directories unless explicitly requested.
- Do not delete working code to simplify a patch.
- Do not add large dependencies for a small feature.

After editing:

- Summarize changed files.
- Summarize behavior changes.
- Summarize tests run.
- Mention tests not run and why.

## Determinism and reproducibility

Preserve deterministic behavior whenever seeds are provided.

Rules:

- Use explicit seeds for random sampling.
- Do not introduce global random state unless necessary.
- Prefer `numpy.random.Generator` over implicit global randomness.
- Keep design IDs stable for the same parameter dictionary.
- Do not make result ordering nondeterministic.
- When parallelizing, preserve per-design failure isolation.

For optimization changes:

- Preserve Optuna study resumability.
- Preserve deterministic sampler initialization when a seed is provided.
- Do not silently change the objective direction.
- Do not silently change the average-over-tools behavior.

## Simulation semantics

The MuJoCo CPU path is the primary backend.

When changing simulation code:

- Preserve graceful failure behavior for failed designs.
- Failed simulations should produce structured failure payloads where possible.
- Wrench scores should remain bounded and interpretable.
- Contact geometry changes should be configurable when they may affect baseline scores.
- Avoid hard-coding values that should belong in `configs/default_eval.yaml` or `configs/search_space.yaml`.

When adding a geometry mode:

- Keep the existing geometry mode available.
- Add a config switch.
- Add a smoke test or unit test.
- Document expected differences from the previous mode.

## Backend policy

The backend abstraction exists to protect the MuJoCo CPU path, not to encourage speculative backend growth.

Allowed by default:

- `mujoco`
- `mujoco_cpu`

Not allowed by default:

- `isaac`
- `isaac_sim`
- `isaac_lab`
- `ros`
- `mjx`
- `mujoco_warp`

MJX or MuJoCo-Warp may only be added as an optional benchmark prototype when explicitly requested. They must not replace the default CPU backend.

## CLI and output compatibility

Preserve existing commands where possible.

Important existing workflows:

```bash
python3 scripts/generate_designs.py --n-designs 5 --output-dir outputs/designs --seed 0
python3 scripts/evaluate_design_batch.py --task-id 0 --designs-per-task 5 --design-dir outputs/designs --results-dir outputs/results --config configs/default_eval.yaml
python3 scripts/collect_results.py --results-dir outputs/results --output-csv outputs/results.csv
```

```bash
python3 scripts/run_optuna_round.py \
  --study-name handcdo-mujoco \
  --storage sqlite:///outputs/handcdo_optuna.db \
  --n-trials 20 \
  --n-grasp-trials 4 \
  --output-dir outputs \
  --seed 0 \
  --tools hammer,spoon,knife \
  --backend mujoco_cpu
```

Do not break legacy aliases such as `--backend mujoco` unless the task explicitly asks for a breaking change.

Generated outputs should remain under `outputs/` unless the user explicitly chooses another directory.

## Testing requirements

Run the smallest relevant verification for the change.

### General code changes

```bash
pytest -q
```

### Simulation or geometry changes

```bash
python3 scripts/generate_designs.py --n-designs 2 --output-dir outputs/smoke_designs --seed 0
python3 scripts/evaluate_design_batch.py --task-id 0 --designs-per-task 2 --design-dir outputs/smoke_designs --results-dir outputs/smoke_results --config configs/default_eval.yaml
python3 scripts/collect_results.py --results-dir outputs/smoke_results --output-csv outputs/smoke_results.csv
```

### Baseline-sensitive changes

For contact model, wrench scoring, grasp optimization, or objective changes, also run or update a small baseline comparison when practical.

Example:

```bash
python3 scripts/run_baseline_benchmark.py \
  --n-designs 2 \
  --n-grasp-trials 1 \
  --tools hammer \
  --seed 0 \
  --backend mujoco_cpu \
  --config configs/default_eval.yaml \
  --output-dir outputs/smoke_baseline
```

Then compare against the relevant baseline using the repository's benchmark comparison script if available.

### MuJoCo availability

If MuJoCo is not installed in the environment:

- MuJoCo-dependent tests should be skipped gracefully.
- Non-simulation tests should still run.
- Do not fail unrelated tests only because MuJoCo is unavailable.

## Dependency policy

Keep dependencies minimal.

Current dependency classes:

- Core numerical stack.
- MuJoCo CPU simulation.
- Optuna optimization.
- scikit-learn analysis.
- SHAP as optional analysis functionality.
- matplotlib for plots.
- PyYAML for config.
- pytest for tests.

Do not add heavy dependencies unless the task clearly requires them.

Avoid:

- GPU-only libraries.
- Robotics middleware.
- Simulator stacks outside the stated scope.
- Large mesh or asset packages.
- Dependencies that are difficult to install on headless HPC nodes.

## HPC and Slurm policy

This repository should remain suitable for CPU HPC batch execution.

For Slurm changes:

- Preserve array-task behavior.
- Preserve per-design failure isolation.
- Avoid assumptions about local GPUs.
- Avoid hard-coded cluster-specific paths.
- Keep logs and generated outputs out of git.
- Keep scripts usable from a clean checkout after editable install.

For long-running experiments:

- Prefer resumable workflows.
- Prefer explicit output directories.
- Prefer small smoke tests in documentation.
- Do not make CI or default tests depend on large experiments.

## Documentation policy

When behavior changes, update documentation close to the change.

Update README or relevant comments when:

- A CLI flag is added.
- A config field is added.
- Output schema changes.
- Simulation semantics change.
- A new geometry mode is added.
- A benchmark or analysis workflow changes.

Do not over-document implementation details that are likely to change.

## Code style

Use straightforward Python.

Prefer:

- Small functions.
- Dataclasses for structured configs or payloads.
- Type hints for public functions.
- Explicit error messages.
- Clear config parsing.
- Deterministic tests.

Avoid:

- Clever abstractions.
- Hidden global state.
- Broad exception swallowing without structured output.
- Large functions that mix sampling, simulation, optimization, and plotting.
- Circular imports.

## Review checklist

Before considering a change complete, verify:

- The change respects the CPU-only MuJoCo scope.
- The change does not introduce Isaac Sim, ROS, or GPU-only assumptions.
- The change is minimal and localized.
- Existing CLI behavior is preserved where possible.
- Seeds still produce deterministic behavior where relevant.
- Generated artifacts are not committed.
- Relevant tests or smoke tests were run.
- Any skipped tests are explained.
- Any expected score change is documented.
- Paper-fidelity claims are phrased as approximations unless validated.
