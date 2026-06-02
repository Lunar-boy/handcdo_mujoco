Implement PR 5: add optional palm pad grid geometry.

Current repository facts:
- `handcdo.hand_model.PalmPad` currently stores `name`, `pos`, and `size`.
- `build_hand_model` currently creates two `palm_kernel_pad_*` box pads.
- `mjcf_generator.build_mjcf_xml` writes all `hand.palm_pads` as box geoms.
- README says palm kernels are approximated as configurable local contact pads.

Goal:
Add a `pad_grid` palm contact mode controlled by geometry config. The default `box_pads` mode must remain unchanged.

Design:
- `box_pads`: existing behavior using `hand.palm_pads`.
- `pad_grid`: generate a low-resolution grid of small box pads on the palm top surface.
- Keep this simple and deterministic.
- Do not implement full Gaussian surface deformation in this PR.

Required changes:
1. Add helper in `handcdo.mjcf_generator.py`, for example:
   - `_add_palm_box_pads(palm, hand, palm_config)`
   - `_add_palm_pad_grid(palm, hand, palm_config)`

2. `pad_grid` behavior:
   - Use `palm_config.pad_resolution`, clamped or validated to a small range, e.g. 2 to 5.
   - Generate `pad_resolution x pad_resolution` box geoms.
   - Place them on top of the palm.
   - Use conservative pad height, e.g. 0.003-0.006.
   - Use `palm_config.pad_friction`.

3. Suggested naming:
```text
palm_grid_pad_r{row}_c{col}
```

4. Preserve current behavior:
   - Default config should produce the same existing palm pad XML.
   - Only `mode: pad_grid` should generate grid pads.

5. Update config:
   - `configs/geometry_medium.yaml` should use `pad_grid` with `pad_resolution: 3`.
   - `configs/geometry_high.yaml` may use `pad_grid` with `pad_resolution: 4`.

6. Add tests:
   - `tests/test_palm_pad_grid.py`
   - Default XML includes existing `palm_kernel_pad_1` and `palm_kernel_pad_2`.
   - Pad-grid XML includes `palm_grid_pad_...`.
   - For resolution 3, exactly 9 grid pads are created.
   - If MuJoCo is installed, generated XML loads.

7. Update docs:
   - Add `docs/geometry_modes.md` if it does not exist.
   - Explain that `pad_grid` is a low/medium-fidelity approximation, not full paper-level surface deformation.

Out of scope:
- Do not implement Gaussian kernel deformation.
- Do not add mesh collision.
- Do not add finger surface pieces.
- Do not change wrench scoring.

Validation:
```bash
pytest -q
```
