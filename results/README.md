# Results

Committed result artifacts are deterministic outputs from the repository's
CPU-only toy experiments. They are included so paper claims can point to a
concrete artifact, not only a command.

Current artifacts:

- `toy_sweep_seed11.json` - machine-readable six-case scenario sweep.
- `toy_sweep_seed11.md` - human-readable table and caveats for the same sweep.
- `deep_matrix_20seed.json` - canonical 20-seed, 18-case matrix used in the
  paper and rendered artifacts.
- `deep_matrix_20seed.csv` - raw case-by-seed rows for independent checking.
- `deep_matrix_20seed.md` - human-readable multi-seed summary with 95%
  confidence intervals.
- `figures/deep_matrix_delta.svg` - critic-minus-group bar chart.
- `figures/deep_matrix_coverage.svg` - critic coverage scatter plot.

Regenerate them with:

```bash
python3 -m experiments.scenario_sweep \
  --seed 11 \
  --output-json results/toy_sweep_seed11.json \
  --output-md results/toy_sweep_seed11.md
```

Regenerate the canonical matrix with:

```bash
python3 -m experiments.deep_matrix \
  --output-json results/deep_matrix_20seed.json \
  --output-csv results/deep_matrix_20seed.csv \
  --output-md results/deep_matrix_20seed.md \
  --figures-dir results/figures
```

Rendered public paper artifacts live under `public/` and are tracked with
`public/artifact_manifest.json`.
