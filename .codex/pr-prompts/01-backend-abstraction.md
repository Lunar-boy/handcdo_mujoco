Implement PR 1: add a simulator backend abstraction without changing current behavior.

Current repository facts:
- `scripts/run_optuna_round.py` only forwards to `handcdo.optimize_hand.main()`.
- `handcdo.optimize_hand.evaluate_design()` calls `optimize_grasp_for_tool(...)`.
- `handcdo.optimize_grasp` imports `evaluate_grasp`, `EvaluationConfig`, and `GraspEvaluation` directly from `handcdo.mujoco_eval`.
- `handcdo.optimize_hand.build_parser()` currently accepts `--backend mujoco` only.
- `handcdo.mujoco_eval.evaluate_grasp(...)` is the current MuJoCo CPU evaluator.

Goal:
Create a small backend layer so the current MuJoCo CPU evaluator becomes one backend named `mujoco_cpu`, while preserving existing behavior.

Required changes:
1. Create package:
   - `handcdo/backends/__init__.py`
   - `handcdo/backends/base.py`
   - `handcdo/backends/mujoco_cpu.py`
   - `handcdo/backends/registry.py`

2. In `handcdo/backends/base.py`, define a `SimulatorBackend` Protocol with:
   - `name: str`
   - `evaluate_grasp(design, tool_name, grasp, config) -> GraspEvaluation`

3. In `handcdo/backends/mujoco_cpu.py`, implement `MujocoCpuBackend`.
   - It should wrap `handcdo.mujoco_eval.evaluate_grasp`.
   - Do not move or rewrite all of `mujoco_eval.py` in this PR.

4. In `handcdo/backends/registry.py`, implement:
   - `get_backend(name: str) -> SimulatorBackend`
   - Accept both `"mujoco_cpu"` and legacy alias `"mujoco"`.
   - Raise `ValueError` for unknown backend names.

5. Update `handcdo.optimize_grasp.optimize_grasp_for_tool(...)`:
   - Add optional argument `backend=None`.
   - If backend is `None`, default to `get_backend("mujoco_cpu")`.
   - Replace direct calls to `evaluate_grasp(...)` with `backend.evaluate_grasp(...)`.
   - Preserve random fallback behavior.

6. Update `handcdo.optimize_hand.evaluate_design(...)`:
   - Add `backend_name: str = "mujoco_cpu"` or `backend=None`.
   - Resolve backend once per design, then pass it into `optimize_grasp_for_tool(...)`.
   - Preserve output JSON schema.

7. Update CLI:
   - Change `--backend` choices from `["mujoco"]` to `["mujoco", "mujoco_cpu"]`.
   - Default may remain `"mujoco"` for backward compatibility, or become `"mujoco_cpu"` if README is updated.

8. Update README examples if needed:
   - Prefer `--backend mujoco_cpu`.
   - Mention `mujoco` is a legacy alias.

9. Add tests:
   - `tests/test_backends.py`
   - Test `get_backend("mujoco_cpu")`.
   - Test `get_backend("mujoco")` returns a backend with equivalent name or behavior.
   - Test unknown backend raises `ValueError`.
   - If feasible, monkeypatch backend evaluation to verify `optimize_grasp_for_tool` calls backend, not direct `mujoco_eval.evaluate_grasp`.

Out of scope:
- Do not implement MJX, Isaac, or geometry modes.
- Do not change scoring logic.
- Do not change MJCF generation.
- Do not change Optuna objective semantics.

Validation:
```bash
pytest -q
python3 scripts/run_optuna_round.py --study-name smoke-backend --storage sqlite:///outputs/smoke_backend.db --n-trials 1 --n-grasp-trials 1 --output-dir outputs/smoke_backend --seed 0 --tools hammer --backend mujoco_cpu
```
