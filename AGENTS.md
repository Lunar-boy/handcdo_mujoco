# AGENTS.md

## Project
This repository is `handcdo_mujoco`, a CPU-only MuJoCo reproduction framework for the optimization infrastructure of "Function-based Parametric Co-Design Optimization of Dexterous Hands" (arXiv:2604.27557).

The current scope is:
- Parametric hand design sampling.
- Primitive MJCF generation.
- MuJoCo grasp/wrench evaluation.
- TPE optimization.
- Slurm array execution.
- Result collection.
- Random Forest / SHAP analysis.

The repository is intentionally not an exact reproduction of the original Isaac Sim, UR5e, OptiTrack, hardware fabrication, or real robot setup.

## Non-negotiable constraints
- Do not add Isaac Sim, Isaac Lab, ROS, or GPU-only dependencies.
- Do not remove the existing MuJoCo CPU path.
- Do not implement MJX/MuJoCo-Warp unless the task explicitly asks for a benchmark-only prototype.
- Keep PRs small and reviewable.
- Preserve deterministic behavior when seeds are provided.
- Preserve existing CLI behavior where possible.
- Do not commit generated outputs, logs, caches, virtual environments, databases, or large mesh artifacts.
- Do not rewrite the whole repository.

## Engineering priorities
1. First preserve current behavior.
2. Then add backend abstraction.
3. Then add configurable geometry modes.
4. Then add fingertip/palm contact geometry.
5. Then add regression and multi-fidelity evaluation.
6. Only later benchmark MJX-Warp.

## Test commands
Run:
```bash
pytest -q
```

For changes to simulation or geometry generation, also run a small smoke test:
```bash
python3 scripts/generate_designs.py --n-designs 2 --output-dir outputs/smoke_designs --seed 0
python3 scripts/evaluate_design_batch.py --task-id 0 --designs-per-task 2 --design-dir outputs/smoke_designs --results-dir outputs/smoke_results --config configs/default_eval.yaml
python3 scripts/collect_results.py --results-dir outputs/smoke_results --output-csv outputs/smoke_results.csv
```

If MuJoCo is not installed in the environment, tests that require MuJoCo should be skipped gracefully rather than hard-failing unrelated tests.
