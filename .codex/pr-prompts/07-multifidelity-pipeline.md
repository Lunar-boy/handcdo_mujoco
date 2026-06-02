Implement PR 7: add multi-fidelity evaluation pipeline scripts.

Current repository facts:
- The repo supports generating designs, evaluating design batches, collecting result JSON into CSV, and running Optuna.
- Current config is single-fidelity.
- The roadmap requires fast/medium/high evaluation modes.

Goal:
Add scripts and configs for a multi-fidelity workflow:
1. Fast search/evaluation.
2. Medium re-evaluation of top candidates.
3. High re-evaluation of a smaller top set.

Required changes:
1. Add configs if they do not exist:
   - `configs/eval_fast.yaml`
   - `configs/eval_medium.yaml`
   - `configs/eval_high.yaml`

Suggested values:
```yaml
# eval_fast.yaml
simulation:
  settle_steps: 150
  close_steps: 220
  wrench_steps: 150
wrench:
  translation_threshold: 0.045
  rotation_threshold_rad: 0.55
geometry:
  finger:
    mode: capsule
    fingertip_pad_enabled: false
  palm:
    mode: box_pads
  tool:
    mode: primitive
```

```yaml
# eval_medium.yaml
simulation:
  settle_steps: 250
  close_steps: 350
  wrench_steps: 250
wrench:
  translation_threshold: 0.045
  rotation_threshold_rad: 0.55
geometry:
  finger:
    mode: capsule_tip_pad
    fingertip_pad_enabled: true
    fingertip_pad_shape: box
  palm:
    mode: pad_grid
    pad_resolution: 3
  tool:
    mode: primitive
```

```yaml
# eval_high.yaml
simulation:
  settle_steps: 350
  close_steps: 450
  wrench_steps: 400
wrench:
  translation_threshold: 0.045
  rotation_threshold_rad: 0.55
geometry:
  finger:
    mode: capsule_tip_pad
    fingertip_pad_enabled: true
    fingertip_pad_shape: box
  palm:
    mode: pad_grid
    pad_resolution: 4
  tool:
    mode: hybrid
```

2. Add script:
   - `scripts/select_top_designs.py`

Behavior:
```bash
python3 scripts/select_top_designs.py \
  --input-csv outputs/fast/results.csv \
  --top-k 100 \
  --output-design-ids outputs/medium/design_ids.txt
```

3. Add script:
   - `scripts/reevaluate_designs.py`

Behavior:
- Read design IDs or design JSON files.
- Re-evaluate with a selected config/fidelity.
- Preserve result schema.
- Add metadata fields:
  - `fidelity`
  - `backend`
  - `config_path`
  - `n_grasp_trials`
  - `seed`

4. Add script:
   - `scripts/merge_multifidelity_results.py`

Behavior:
- Merge fast/medium/high CSVs.
- Preserve separate score columns if possible:
  - `hand_score_fast`
  - `hand_score_medium`
  - `hand_score_high`

5. Add tests:
   - Select top-k from synthetic CSV.
   - Merge synthetic fidelity CSVs.
   - Re-evaluation script should fail clearly if design JSON is missing.

6. Update README:
   - Add "Multi-fidelity workflow" section.

Out of scope:
- Do not implement surrogate modeling in this PR.
- Do not implement MJX-Warp.
- Do not add mesh assets.

Validation:
```bash
pytest -q
python3 scripts/select_top_designs.py --input-csv <synthetic-or-existing-csv> --top-k 5 --output-design-ids outputs/top5.txt
```
