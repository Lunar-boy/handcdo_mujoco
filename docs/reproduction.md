# Paper-like evaluation protocol

The repository implements an explicit MuJoCo approximation of the grasp-stability objective described in arXiv:2604.27557. The protocol evaluates hammer, spoon, and knife grasps under the canonical Cartesian perturbations `+Fx`, `-Fx`, `+Fy`, `-Fy`, `+Fz`, `-Fz`, `+Tx`, `-Tx`, `+Ty`, `-Ty`, `+Tz`, and `-Tz`.

For each wrench, the score is the stable number of simulation steps divided by the configured perturbation steps, clamped to `[0, 1]`. A grasp score is the mean over the 12 wrench directions. The tool score is the best grasp-candidate score, and the design score is the mean across configured tools.

The default protocol in `configs/eval_paper_protocol.yaml` uses the paper-like geometry from `configs/eval_paper_like.yaml`, fixed force and torque magnitudes, deterministic random grasp candidates, and explicit rollout thresholds.

## Smoke evaluation

The default command uses a deterministic non-physical backend so protocol wiring, aggregation, serialization, and seed reproducibility can be checked quickly:

```bash
python3 scripts/run_paper_eval_smoke.py \
  --config configs/eval_paper_protocol.yaml \
  --num-designs 2 \
  --output outputs/paper_eval_smoke
```

Run the same protocol through the real CPU MuJoCo backend with:

```bash
python3 scripts/run_paper_eval_smoke.py \
  --config configs/eval_paper_protocol.yaml \
  --num-designs 2 \
  --output outputs/paper_eval_mujoco \
  --backend mujoco_cpu
```

The command writes:

- `results.json`: nested design, best-tool-grasp, and per-wrench results.
- `results.csv`: one row per design, tool, and wrench direction.
- `run_config.json`: resolved run inputs including backend, seed, geometry config, tools, and grasp-candidate count.

Each wrench result includes its direction, stable time in seconds, normalized stable time, and a failure reason when a displacement or rotation threshold was exceeded.

## Larger experiments

Increase `--num-designs` for a fixed random morphology batch, or use `scripts/run_optuna_round.py` for resumable morphology optimization. Keep the protocol config, geometry config, seed, backend, tool assets, and grasp budget with reported results. Generated runs belong under `outputs/` and should not be committed.

## Reproduction boundary

This protocol approximates the paper's stability objective in MuJoCo. It uses canonical 12-direction force/torque perturbations and normalized stable time. Exact scores may differ from the original implementation because the simulator backend, tool meshes, contact parameters, and grasp optimizer are not identical.

The deterministic smoke backend is only a reproducibility and schema test. It does not produce physical stability measurements. Paper-like MuJoCo scores also remain an approximation and should not be presented as numerical parity with the original simulator or hardware experiments.
