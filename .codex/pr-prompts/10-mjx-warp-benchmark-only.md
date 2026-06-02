Implement PR 10: add MJX-Warp benchmark scaffolding only, without integrating it into the main pipeline.

Important:
This PR is optional and should only be attempted after MuJoCo CPU geometry and regression tests are stable.

Goal:
Add a benchmark-only scaffold to evaluate whether MJX-Warp is worth using later. Do not replace the CPU MuJoCo backend.

Required changes:
1. Add package/module:
   - `handcdo/backends/mujoco_warp.py` or `handcdo/benchmarks/mjx_warp.py`

2. If MJX-Warp dependencies are unavailable, the module must fail gracefully and tests must skip.

3. Add script:
   - `scripts/benchmark_mjx_warp.py`

4. Benchmark design:
   - Use a fixed design JSON.
   - Use fixed tool(s).
   - Use fixed grasp samples.
   - Compare:
     - CPU MuJoCo time
     - MJX-Warp time if available
     - score correlation if score is implemented
     - failure count

5. Do not make `mujoco_warp` available in normal `--backend` choices unless the benchmark is functional and tests pass.

6. Add tests:
   - Import test that skips if dependencies are absent.
   - CLI help test.
   - No hard dependency on GPU or MJX-Warp in default test suite.

Out of scope:
- Do not replace current evaluator.
- Do not implement JAX autodiff.
- Do not require H100/GPU for pytest.
- Do not change default CLI behavior.
- Do not modify Slurm scripts to use GPU.

Validation:
```bash
pytest -q
python3 scripts/benchmark_mjx_warp.py --help
```
