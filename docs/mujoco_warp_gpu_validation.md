# MuJoCo Warp GPU Validation

MuJoCo Warp validation is optional because the default development environment is CPU-first. Importing the project, parsing CLIs, and running ordinary tests must not require CUDA, MuJoCo Warp, Slurm, or an HPC allocation.

## CPU Tests

Run the default CPU-safe suite:

```bash
pytest -q
```

To confirm GPU tests are skipped by default:

```bash
pytest -q -rs
```

## GPU Validation On HPC/Slurm

Run strict GPU validation through the submission wrapper:

```bash
scripts/submit_mujoco_warp_gpu_validation.sh
```

For sites requiring explicit partition or account options:

```bash
scripts/submit_mujoco_warp_gpu_validation.sh --partition=<gpu-partition> --account=<account>
```

This submits `slurm/validate_mujoco_warp_gpu.sbatch`, which runs `scripts/validate_mujoco_warp_gpu.py --strict ...` on a GPU allocation. Strict success means the real MuJoCo Warp GPU backend path completed end-to-end. If the runtime lacks true fixed-grasp batching support, strict validation should fail instead of xfail.

The job writes:

```text
logs/warp_gpu_validate_<jobid>.out
logs/warp_gpu_validate_<jobid>.err
outputs/warp_gpu_validation/*.json
```

Do not run GPU pytest directly on login nodes. Direct GPU pytest commands are debug-only after `salloc`, `srun`, or the site equivalent has already provided an interactive GPU allocation:

```bash
# Debug only inside an existing GPU allocation:
RUN_GPU_TESTS=1 RUN_STRICT_WARP_INTEGRATION=1 pytest -q -m gpu
```

## Validation Script

The standard acceptance workflow is the Slurm wrapper above. The validation script still supports direct non-strict mode for local developer debugging or environment probing:

```bash
python3 scripts/validate_mujoco_warp_gpu.py --results-dir outputs/warp_gpu_validation
```

Strict mode requires full success and is what the Slurm template runs by default:

```bash
python3 scripts/validate_mujoco_warp_gpu.py \
  --results-dir outputs/warp_gpu_validation \
  --strict
```

Allow skipped validation in CPU-only environments:

```bash
python3 scripts/validate_mujoco_warp_gpu.py \
  --results-dir outputs/warp_gpu_validation \
  --allow-skip
```

Exit codes:

```text
0 = passed
0 = skipped because prerequisites are missing and --allow-skip was set
0 = xfailed in default mode because true fixed-grasp batching is unsupported, after truthful metadata checks pass
1 = backend validation failed after prerequisites were available
2 = prerequisites missing and --allow-skip was not set
```

Default mode is not weaker on successful backend runs: it still requires true batched scoring, no failed evaluations, and 12 wrench results per grasp. The only default-mode relaxation is `status: "xfailed"` for the known true fixed-grasp batching capability gate after the backend was called and metadata truthfully reports the limitation. Strict mode treats that same capability gate as `status: "failed"` with exit code `1`. Unexpected exceptions always fail in both modes.

## Slurm

Submit the generic GPU validation template through the wrapper:

```bash
scripts/submit_mujoco_warp_gpu_validation.sh
```

Pass Slurm overrides through the wrapper when needed:

```bash
scripts/submit_mujoco_warp_gpu_validation.sh --partition=gpu --time=00:20:00
```

The wrapper creates `logs/` and `outputs/warp_gpu_validation/` before calling `sbatch`. This matters because Slurm may open `#SBATCH --output` and `#SBATCH --error` paths before the sbatch script body has a chance to run. The underlying template remains generic and runs strict validation by default; add local partition, account, or module-load directives as needed.

## Report Metadata

Important fields in the JSON report and backend metadata include:

```text
backend
experimental
score_semantics
include_in_multifidelity
true_batched_scoring
per_world_state_init
failure_count
failure_reason
sequential_fallback
num_grasps
num_chunks
nworld
readback_interval
warmup_requested_steps
warmup_executed_steps
capture_graph_requested
capture_graph_enabled
capture_graph_reason
warp_capabilities
```

## Limitations

The MuJoCo Warp path remains experimental and is not claimed CPU-equivalent. GPU tests require a real CUDA and MuJoCo Warp runtime. Graph capture may remain disabled when unsupported by the dynamic readback path. Default GPU mode may xfail on runtimes that cannot support true fixed-grasp batching; strict mode is the proof that the current environment can run the real end-to-end path successfully.
