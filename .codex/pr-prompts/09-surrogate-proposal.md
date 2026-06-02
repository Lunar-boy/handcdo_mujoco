Implement PR 9: add surrogate-assisted candidate proposal.

Current repository facts:
- README documents Random Forest plus SHAP analysis.
- The roadmap needs surrogate-assisted proposal after enough simulation data is available.

Goal:
Add a lightweight surrogate proposal module that trains on collected simulation results and proposes new candidate designs for MuJoCo validation.

Required changes:
1. Add package:
   - `handcdo/surrogate/__init__.py`
   - `handcdo/surrogate/train.py`
   - `handcdo/surrogate/propose.py`

2. Use sklearn only if it is already a dependency. If not available, fail gracefully with a clear message or add it only if pyproject already supports analysis extras.

3. Implement:
```python
train_surrogate(results_csv: Path, output_dir: Path) -> None
```
Suggested model order:
- `RandomForestRegressor`
- `ExtraTreesRegressor`

4. Implement candidate proposal:
```python
propose_candidates(model_path, search_space, n_random, top_k, output_dir, seed)
```

5. Candidate proposal logic:
- Sample many random designs from `DesignSpace`.
- Convert design params into tabular features.
- Predict score.
- Select top-k candidates.
- Save:
  - `proposed_designs/<design_id>/design.json`
  - `proposed_candidates.csv`

6. Add script:
   - `scripts/propose_surrogate_candidates.py`

Suggested CLI:
```bash
python3 scripts/propose_surrogate_candidates.py \
  --results-csv outputs/results.csv \
  --search-space configs/search_space.yaml \
  --n-random 10000 \
  --top-k 200 \
  --output-dir outputs/surrogate_proposals \
  --seed 0
```

7. Add tests:
   - Use tiny synthetic results CSV.
   - Ensure training writes a model or clear output.
   - Ensure proposal writes exactly top-k candidate rows.
   - Ensure deterministic output with fixed seed.

8. Update README:
   - Add a short "Surrogate candidate proposal" section.
   - Explain that proposed candidates still require MuJoCo re-evaluation.

Out of scope:
- Do not replace Optuna with surrogate.
- Do not implement neural networks.
- Do not implement uncertainty unless simple and robust.
- Do not change main optimization loop.

Validation:
```bash
pytest -q
```
