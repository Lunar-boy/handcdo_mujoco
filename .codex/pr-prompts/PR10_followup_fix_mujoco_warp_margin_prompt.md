# PR10 Follow-up: Fix MuJoCo Warp MULTICCD Non-zero Margin Compatibility

You are working in the existing `handcdo_mujoco` repository after PR10 has already implemented the benchmark-only MuJoCo Warp scaffold and after the PR10 merge-blocker follow-up.

This is a **targeted bug-fix prompt**, not a new feature PR.

The Capella GPU smoke test now reaches MuJoCo Warp on an H100, but the benchmark fails during the Warp stepping stage because MuJoCo Warp rejects non-zero geom/contact margins when MULTICCD is enabled.

Keep this PR strictly benchmark-only.

Do **not** add a production MuJoCo Warp backend.

---

## Observed failure

The Capella smoke command:

```bash
python3 scripts/benchmark_mujoco_warp.py \
  --output-dir "outputs/mujoco_warp_capella_smoke_${SLURM_JOB_ID}" \
  --config configs/eval_fast.yaml \
  --tool hammer \
  --scene-mode contact_smoke \
  --steps 20 \
  --warmup-steps 2 \
  --cpu-repeats 1 \
  --warp-repeats 1 \
  --nworld 4 \
  --nconmax 64 \
  --njmax 128 \
  --require-warp \
  --overwrite
```

successfully reaches MuJoCo Warp initialization on H100:

```text
Warp 1.14.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA H100" (93 GiB, sm_90, mempool enabled)
```

But `benchmark_results.json` contains:

```json
{
  "available": true,
  "backend": "mujoco_warp",
  "error": "geom pair (palm_geom, hammer_head) has non-zero margin ((0.001, 0.001)) with MULTICCD enabled. Set margin to 0 or disable MULTICCD.",
  "exception_type": "NotImplementedError",
  "failure_count": 1,
  "failure_stage": "warp_step",
  "nconmax": 64,
  "njmax": 128,
  "nworld": 4,
  "scene_mode": "contact_smoke",
  "success": false
}
```

The sbatch script then fails because `--require-warp` correctly treats a failed Warp row as a RuntimeError:

```text
benchmark_mujoco_warp failed: RuntimeError: MuJoCo Warp is required but at least one benchmark row failed
```

---

## Root cause

The current MJCF generator likely emits a default non-zero geom margin similar to:

```xml
<default>
  <geom solref="0.012 1" solimp="0.9 0.95 0.001" margin="0.001"/>
</default>
```

CPU MuJoCo accepts this.

MuJoCo Warp with MULTICCD enabled rejects non-zero margins on colliding geom pairs.

The error is not caused by Slurm, Python environment, CUDA, H100 allocation, or missing dependencies.

The correct fix is to extend the existing benchmark-local MJCF compatibility rewrite.

---

## Required fix

Update:

```text
handcdo/benchmarks/mujoco_warp.py
```

Specifically update the existing benchmark-local MJCF rewrite logic, likely in or near:

```python
prepare_warp_compatible_mjcf(...)
```

The existing PR10 rewrite probably already handles:

```text
option.integrator: implicitfast -> Euler
```

Extend it to also rewrite non-zero margins in the benchmark-local `warp_model.xml`.

Do **not** modify the global MJCF generator.

Do **not** modify `handcdo/mjcf_generator.py`.

Do **not** modify the CPU MuJoCo model generation semantics.

Preserve the original model as:

```text
<output-dir>/model/original_model.xml
```

Apply compatibility rewrites only to:

```text
<output-dir>/model/warp_model.xml
```

Record every rewrite in `benchmark_results.json` under `mjcf_rewrites`.

---

## Margin rewrite requirements

When `allow_rewrite=True`, rewrite these benchmark-local XML attributes.

### 1. All geom margins

Rewrite every non-zero `margin` attribute on any `<geom>` element:

```xml
<geom margin="0.001"/>
```

to:

```xml
<geom margin="0"/>
```

This must include default geoms:

```xml
<default>
  <geom margin="0.001"/>
</default>
```

and world/body geoms:

```xml
<worldbody>
  <body>
    <geom margin="0.002"/>
  </body>
</worldbody>
```

### 2. All explicit contact pair margins

Rewrite every non-zero `margin` attribute on any `<pair>` element:

```xml
<contact>
  <pair geom1="a" geom2="b" margin="0.003"/>
</contact>
```

to:

```xml
<pair geom1="a" geom2="b" margin="0"/>
```

### 3. Keep zero and missing margins unchanged

Do not add `margin="0"` to elements that do not already have a margin attribute unless your existing XML rewrite infrastructure requires it.

Do not rewrite margins already equal to zero.

### 4. Record metadata

For each rewrite, append a structured metadata record.

Suggested examples:

```json
{
  "field": "geom.margin",
  "old": "0.001",
  "new": "0",
  "reason": "benchmark-local MuJoCo Warp MULTICCD compatibility"
}
```

```json
{
  "field": "pair.margin",
  "old": "0.003",
  "new": "0",
  "reason": "benchmark-local MuJoCo Warp MULTICCD compatibility"
}
```

If possible, include identifying context:

```json
{
  "field": "geom.margin",
  "element": "geom",
  "name": "palm_geom",
  "old": "0.001",
  "new": "0",
  "reason": "benchmark-local MuJoCo Warp MULTICCD compatibility"
}
```

For default geoms that have no name, use `"name": null` or omit the field.

---

## Suggested implementation shape

If the code already parses MJCF with `xml.etree.ElementTree`, add a helper similar to:

```python
from typing import Any
import xml.etree.ElementTree as ET


def _first_float(text: str) -> float | None:
    parts = text.split()
    if not parts:
        return None
    try:
        return float(parts[0])
    except ValueError:
        return None


def _rewrite_nonzero_margins_for_warp(root: ET.Element) -> list[dict[str, Any]]:
    rewrites: list[dict[str, Any]] = []

    for geom in root.findall(".//geom"):
        old = geom.get("margin")
        if old is None:
            continue
        value = _first_float(old)
        if value is None:
            continue
        if value != 0.0:
            geom.set("margin", "0")
            rewrites.append(
                {
                    "field": "geom.margin",
                    "element": "geom",
                    "name": geom.get("name"),
                    "old": old,
                    "new": "0",
                    "reason": "benchmark-local MuJoCo Warp MULTICCD compatibility",
                }
            )

    for pair in root.findall(".//pair"):
        old = pair.get("margin")
        if old is None:
            continue
        value = _first_float(old)
        if value is None:
            continue
        if value != 0.0:
            pair.set("margin", "0")
            rewrites.append(
                {
                    "field": "pair.margin",
                    "element": "pair",
                    "name": pair.get("name"),
                    "geom1": pair.get("geom1"),
                    "geom2": pair.get("geom2"),
                    "old": old,
                    "new": "0",
                    "reason": "benchmark-local MuJoCo Warp MULTICCD compatibility",
                }
            )

    return rewrites
```

Then call it inside `prepare_warp_compatible_mjcf(...)` only when rewrites are allowed:

```python
if allow_rewrite:
    rewrites.extend(_rewrite_implicitfast_integrator_for_warp(root))
    rewrites.extend(_rewrite_nonzero_margins_for_warp(root))
```

Adapt this to the existing code structure and field names.

Do not duplicate the whole XML rewrite pipeline if helpers already exist.

---

## Tests to add or update

Update:

```text
tests/test_mujoco_warp_benchmark.py
```

Add a focused unit test for margin rewrite.

Suggested test:

```python
def test_prepare_warp_compatible_mjcf_rewrites_nonzero_margins(tmp_path):
    from handcdo.benchmarks.mujoco_warp import prepare_warp_compatible_mjcf

    original = tmp_path / "original.xml"
    rewritten = tmp_path / "warp.xml"

    original.write_text(
        """<mujoco model="test">
  <option integrator="implicitfast"/>
  <default>
    <geom margin="0.001" solref="0.012 1"/>
  </default>
  <worldbody>
    <body name="body">
      <geom name="g1" type="box" size="0.1 0.1 0.1" margin="0.002"/>
      <geom name="g2" type="box" size="0.1 0.1 0.1" margin="0"/>
      <geom name="g3" type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
  <contact>
    <pair geom1="g1" geom2="g2" margin="0.003"/>
  </contact>
</mujoco>
""",
        encoding="utf-8",
    )

    result = prepare_warp_compatible_mjcf(original, rewritten, allow_rewrite=True)

    text = rewritten.read_text(encoding="utf-8")

    assert 'integrator="Euler"' in text
    assert 'margin="0.001"' not in text
    assert 'margin="0.002"' not in text
    assert 'margin="0.003"' not in text
    assert 'margin="0"' in text

    rewrites = result["mjcf_rewrites"]
    assert any(r["field"] == "option.integrator" for r in rewrites)
    assert any(r["field"] == "geom.margin" and r["old"] == "0.001" for r in rewrites)
    assert any(r["field"] == "geom.margin" and r["old"] == "0.002" for r in rewrites)
    assert any(r["field"] == "pair.margin" and r["old"] == "0.003" for r in rewrites)
```

Also add a no-rewrite test if not already present:

```python
def test_prepare_warp_compatible_mjcf_does_not_rewrite_margins_when_disabled(tmp_path):
    from handcdo.benchmarks.mujoco_warp import prepare_warp_compatible_mjcf

    original = tmp_path / "original.xml"
    rewritten = tmp_path / "warp.xml"

    original.write_text(
        """<mujoco model="test">
  <option integrator="implicitfast"/>
  <default>
    <geom margin="0.001"/>
  </default>
</mujoco>
""",
        encoding="utf-8",
    )

    result = prepare_warp_compatible_mjcf(original, rewritten, allow_rewrite=False)

    text = rewritten.read_text(encoding="utf-8")
    assert 'integrator="implicitfast"' in text
    assert 'margin="0.001"' in text
    assert result["mjcf_rewrites"] == []
```

If an equivalent no-rewrite test already exists, update it to include the margin case.

---

## README update

Update `README.md` Optional MuJoCo Warp benchmark section to mention the new benchmark-local rewrite.

Add wording like:

```markdown
The benchmark preserves the raw generated MJCF as `original_model.xml` and may create a benchmark-local
`warp_model.xml` with compatibility rewrites for MuJoCo Warp. Current compatibility rewrites include:

- `option.integrator="implicitfast"` to `Euler`;
- non-zero geom/contact-pair margins to `0` for MuJoCo Warp MULTICCD compatibility.

These rewrites are benchmark-local and do not change the default CPU MuJoCo generator or scoring path.
```

Do not overclaim physical equivalence after margin rewrite.

---

## PR10 prompt / AGENTS update

If the repository tracks PR prompts in `.codex/pr-prompts/`, update the PR10 prompt only if it is already part of this PR and easy to update.

Add a short compatibility note:

```markdown
The benchmark-local `warp_model.xml` may rewrite unsupported CPU-MuJoCo settings for MuJoCo Warp compatibility, including:

- `option.integrator="implicitfast"` to `Euler`;
- non-zero geom/contact-pair margins to `0` when required by MuJoCo Warp MULTICCD.

The original generated MJCF must be preserved as `original_model.xml`.
```

Do not make large prompt-only churn if the code/test fix is otherwise clean.

---

## Do not do these things

Do **not** disable MULTICCD unless there is already a stable, version-compatible MuJoCo Warp API for doing so and the code clearly records it.

Prefer `margin="0"` rewrite over disabling MULTICCD.

Do **not** modify `handcdo/mjcf_generator.py`.

Do **not** modify the CPU evaluator.

Do **not** modify `handcdo/backends/registry.py`.

Do **not** add `handcdo/backends/mujoco_warp.py`.

Do **not** add `mujoco_warp` to normal backend choices.

Do **not** hide the original exception if Warp still fails later.

---

## Validation commands

After implementing the fix, run locally:

```bash
.venv/bin/python -m pytest -q -m "not gpu and not slow"
.venv/bin/python scripts/benchmark_mujoco_warp.py --help
.venv/bin/python scripts/benchmark_mujoco_warp.py \
  --output-dir outputs/mujoco_warp_local_cpu_smoke \
  --config configs/eval_fast.yaml \
  --tool hammer \
  --steps 5 \
  --warmup-steps 1 \
  --cpu-repeats 1 \
  --warp-repeats 1 \
  --nworld 2 \
  --overwrite
```

The local CPU smoke may skip Warp if `mujoco_warp` is not installed.

Then rerun the Capella smoke:

```bash
VENV_PATH=.venv sbatch slurm/mujoco_warp_capella_smoke.sbatch
```

After the Capella job completes, check:

```bash
sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed
```

Expected:

```text
COMPLETED  0:0
```

Then check:

```bash
cat outputs/mujoco_warp_capella_smoke_<JOB_ID>/benchmark_results.json
```

Expected for the Warp row:

```json
{
  "backend": "mujoco_warp",
  "success": true,
  "failure_stage": null
}
```

If Warp fails with a new unsupported feature, keep the failure structured and visible in `benchmark_results.json`.

---

## Expected final report

At the end, report:

- files changed;
- new margin rewrite behavior;
- tests added or updated;
- local pytest result;
- local CPU smoke result;
- whether Capella smoke was rerun;
- if Capella was rerun, whether `warp_timing` succeeded;
- if Capella was not rerun, state that GPU/HPC validation remains pending.

Use explicit wording. Do not claim final GPU validation unless the Capella job actually completes with `ExitCode 0:0` and `warp_timing.success == true`.
