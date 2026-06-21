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
  confidence intervals. The group-size rows fix critic replay at 840 training
  trajectories so only evaluation sibling group size changes.
- `variance_credit_grid_seed17.json` - estimator grid separating variance
  reduction from credit assignment on the long-wait toy case, including a
  critic-free anchor-action contrast baseline over repeated toy states.
- `variance_credit_grid_seed17.md` - human-readable table and reading notes for
  the same estimator grid.
- `credit_phase_diagram_seedset.json` - broadcast-ceiling phase diagnostic over
  credit heterogeneity, critic observability, coverage, and reward contrast
  under matched train/evaluation dynamics.
- `credit_phase_diagram_seedset.md` - human-readable table and reading notes
  for the same phase diagnostic.
- `selection_regret_seedset.json` - held-out estimator-selection regret audit
  over the phase grid.
- `selection_regret_seedset.md` - human-readable table for the same
  selection-regret audit.
- `policy_gradient_fidelity_seed13.json` - exact finite-MDP policy-gradient
  audit with 200 replications, per-batch matched-KL update diagnostics, and
  policy-implied actor-coefficient rows.
- `policy_gradient_fidelity_seed13.md` - human-readable table for the same
  exact-gradient audit.
- `anchor_coverage_audit_seedset.json` - coverage sweep for the critic-free
  anchor-action contrast estimator over repeated exact toy states.
- `anchor_coverage_audit_seedset.md` - human-readable table for the same
  anchor-coverage audit.
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
- `neural_credit_generalization_seedset.json` - tiny neural value-critic audit
  trained on thresholds 1 and 3 and evaluated on held-out threshold 2.
- `neural_credit_generalization_seedset.md` - human-readable table for the same
  held-out-threshold neural audit.
- `figures/deep_matrix_delta.svg` - critic-minus-group bar chart.
- `figures/deep_matrix_coverage.svg` - critic coverage scatter plot.
- `../public/figures/deep_matrix_delta.png` and
  `../public/figures/deep_matrix_coverage.png` - rendered public PNGs used by
  the LaTeX paper. Regenerate them with `scripts/render_public_figures.py` or
  the full paper build.

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

Regenerate the broadcast-ceiling phase diagnostic with:

```bash
python3 -m experiments.credit_phase_diagram \
  --output-json results/credit_phase_diagram_seedset.json \
  --output-md results/credit_phase_diagram_seedset.md
```

Regenerate the held-out estimator-selection audit with:

```bash
python3 -m experiments.selection_regret \
  --output-json results/selection_regret_seedset.json \
  --output-md results/selection_regret_seedset.md
```

Regenerate the exact-gradient and policy-implied actor-coefficient audit with:

```bash
python3 -m experiments.policy_gradient_fidelity \
  --output-json results/policy_gradient_fidelity_seed13.json \
  --output-md results/policy_gradient_fidelity_seed13.md
```

Regenerate the robustness audits with:

```bash
python3 -m experiments.anchor_coverage_audit \
  --output-json results/anchor_coverage_audit_seedset.json \
  --output-md results/anchor_coverage_audit_seedset.md

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

Regenerate the tiny neural generalization audit with:

```bash
python3 -m experiments.neural_credit_generalization \
  --output-json results/neural_credit_generalization_seedset.json \
  --output-md results/neural_credit_generalization_seedset.md
```

The tracked public paper PDF is
`public/trajectory_rewards_are_not_token_credit.pdf`, with its manifest at
`public/paper_manifest.json`.
