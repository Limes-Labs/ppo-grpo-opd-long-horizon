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
- `variance_credit_grid_seed17.json` - estimator grid separating variance
  reduction from credit assignment on the long-wait toy case, including a
  critic-free anchor-action contrast baseline over repeated toy states.
- `variance_credit_grid_seed17.md` - human-readable table and reading notes for
  the same estimator grid.
- `length_imbalance_audit_seedset.json` - horizon sweep testing within-group
  length imbalance and a length-adjusted group baseline.
- `length_imbalance_audit_seedset.md` - human-readable table for the same
  length-imbalance audit.
- `token_cost_sensitivity_20seed.json` - token-cost robustness audit over
  baseline and long-wait scenarios.
- `token_cost_sensitivity_20seed.md` - human-readable table for the same
  token-cost audit.
- `closed_loop_credit_training_10seed.json` - tabular closed-loop policy
  training audit for group, critic, length-normalized group, and coverage-gated
  credit.
- `closed_loop_credit_training_10seed.md` - human-readable table for the same
  default closed-loop audit.
- `closed_loop_credit_training_low_coverage_10seed.json` - low-replay-coverage
  stress run for the coverage-gated credit proposal.
- `closed_loop_credit_training_low_coverage_10seed.md` - human-readable table
  for the same low-coverage run.
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

Regenerate the variance/credit grid with:

```bash
python3 -m experiments.variance_credit_grid \
  --output-json results/variance_credit_grid_seed17.json \
  --output-md results/variance_credit_grid_seed17.md
```

Regenerate the robustness audits with:

```bash
python3 -m experiments.length_imbalance_audit \
  --output-json results/length_imbalance_audit_seedset.json \
  --output-md results/length_imbalance_audit_seedset.md

python3 -m experiments.token_cost_sensitivity \
  --output-json results/token_cost_sensitivity_20seed.json \
  --output-md results/token_cost_sensitivity_20seed.md
```

Regenerate the closed-loop training artifacts with:

```bash
python3 -m experiments.closed_loop_credit_training \
  --output-json results/closed_loop_credit_training_10seed.json \
  --output-md results/closed_loop_credit_training_10seed.md

python3 -m experiments.closed_loop_credit_training \
  --critic-replay-limit 80 \
  --gate-min-count 4 \
  --train-iterations 50 \
  --groups-per-iteration 16 \
  --group-size 5 \
  --max-steps 14 \
  --eval-groups 60 \
  --output-json results/closed_loop_credit_training_low_coverage_10seed.json \
  --output-md results/closed_loop_credit_training_low_coverage_10seed.md
```

Rendered public paper artifacts live under `public/` and are tracked with
`public/artifact_manifest.json`. The full paper-formatted LaTeX artifact is
`public/ppo_grpo_opd_long_horizon_latex.pdf`, with its own manifest at
`public/latex_artifact_manifest.json`.
