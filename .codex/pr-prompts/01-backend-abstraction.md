Implement PR 1: add a simulator backend abstraction while preserving current MuJoCo CPU behavior.

Context:
This repository is a CPU-only MuJoCo reproduction of the optimization infrastructure from “Function-based Parametric Co-Design Optimization of Dexterous Hands” (arXiv:2604.27557). Do not add Isaac Sim, ROS, GPU-only dependencies, MJX, or MuJoCo-Warp in this PR.

Current repository facts:
- `scripts/run_optuna_round.py` only forwards to `handcdo.optimize_hand.main()`.
- `handcdo.optimize_hand.build_parser()` currently accepts `--backend mujoco` only.
- `handcdo.optimize_hand.run_optuna()` currently does not pass `args.backend` into `evaluate_design(...)`.
- `handcdo.optimize_hand.evaluate_design()` calls `optimize_grasp_for_tool(...)`.
- `handcdo.optimize_grasp` currently imports `evaluate_grasp`, `EvaluationConfig`, and `GraspEvaluation` directly from `handcdo.mujoco_eval`.
- `handcdo.mujoco_eval.evaluate_grasp(...)` is the current MuJoCo CPU evaluator and must remain the source of truth for current scoring behavior.

Goal:
Create a small backend layer so the current MuJoCo CPU evaluator becomes one backend named `mujoco_cpu`, while preserving existing behavior and JSON output schema.

Required changes:

1. Add package:
   - `handcdo/backends/__init__.py`
   - `handcdo/backends/base.py`
   - `handcdo/backends/mujoco_cpu.py`
   - `handcdo/backends/registry.py`

2. In `handcdo/backends/base.py`, define a `SimulatorBackend` Protocol:
   - `name: str`
   - `evaluate_grasp(design, tool_name, grasp, config) -> GraspEvaluation`

   Use `from __future__ import annotations`.
   Avoid importing MuJoCo or constructing MuJoCo models at backend package import time.
   If type imports from `mujoco_eval` are needed, prefer `typing.TYPE_CHECKING` or string annotations to avoid unnecessary runtime coupling.

3. In `handcdo/backends/mujoco_cpu.py`, implement `MujocoCpuBackend`.
   - `name = "mujoco_cpu"`
   - It should call `handcdo.mujoco_eval.evaluate_grasp(...)`.
   - Keep `handcdo.mujoco_eval.py` intact; do not move or rewrite its implementation.

4. In `handcdo/backends/registry.py`, implement:
   - `get_backend(name: str) -> SimulatorBackend`
   - Normalize accepted names:
     - `"mujoco_cpu"` -> `MujocoCpuBackend`
     - `"mujoco"` -> `MujocoCpuBackend` as a legacy alias
   - Raise `ValueError` with a clear message for unknown backend names.

5. Update `handcdo.optimize_grasp.optimize_grasp_for_tool(...)`:
   - Add optional argument `backend=None`.
   - If `backend is None`, default to `get_backend("mujoco_cpu")`.
   - Replace all direct calls to `evaluate_grasp(...)` with `backend.evaluate_grasp(...)`.
   - Preserve both TPE and random fallback behavior.
   - Preserve return payload structure exactly: `tool`, `best_score`, `best_grasp`, `trials`.

6. Update `handcdo.optimize_hand.evaluate_design(...)`:
   - Add `backend_name: str = "mujoco_cpu"` or `backend=None`.
   - Resolve the backend once per design, before the per-tool loop.
   - Pass the resolved backend into `optimize_grasp_for_tool(...)`.
   - Preserve output JSON schema exactly: `design_id`, `parameters`, `hand_score`, `tool_results`, `failed`.

7. Update `handcdo.optimize_hand.run_optuna(...)`:
   - Pass `args.backend` into `evaluate_design(...)`.
   - This is required; the CLI backend argument must not be a no-op.

8. Update CLI:
   - Change `--backend` choices to `["mujoco", "mujoco_cpu"]`.
   - Keep default as `"mujoco"` for backward compatibility.
   - Internally this should resolve to the `mujoco_cpu` backend.

9. Update README examples:
   - Prefer `--backend mujoco_cpu` in the Optuna example.
   - Mention that `mujoco` remains a legacy alias for backward compatibility.

10. Add tests in `tests/test_backends.py`:
   - Test `get_backend("mujoco_cpu").name == "mujoco_cpu"`.
   - Test `get_backend("mujoco")` returns the same backend behavior/name as `mujoco_cpu`.
   - Test unknown backend raises `ValueError`.
   - Add a fake backend class returning deterministic `GraspEvaluation`.
   - Use the fake backend to verify `optimize_grasp_for_tool(...)` calls `backend.evaluate_grasp(...)` in both TPE and random/fallback paths where feasible.
   - Add a parser test verifying both `--backend mujoco` and `--backend mujoco_cpu` are accepted.

Out of scope:
- Do not implement MJX, MuJoCo-Warp, Isaac Sim, or geometry modes.
- Do not change scoring logic.
- Do not change MJCF generation.
- Do not change Optuna objective semantics.
- Do not change result JSON schema.
- Do not rewrite the repository.

Validation:
```bash
pytest -q
python3 scripts/run_optuna_round.py \
  --study-name smoke-backend \
  --storage sqlite:///outputs/smoke_backend.db \
  --n-trials 1 \
  --n-grasp-trials 1 \
  --output-dir outputs/smoke_backend \
  --seed 0 \
  --tools hammer \
  --backend mujoco_cpu