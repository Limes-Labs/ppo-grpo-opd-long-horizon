# LaTeX Paper

This directory contains the full LaTeX manuscript source for the public paper:

- `main.tex` - article source with equations, figures, tables, and caveats.
- `references.bib` - bibliography for primary and near-primary sources.
- `generated/result_macros.tex`, `generated/deep_matrix_table.tex`,
  `generated/axis_summary_table.tex`, `generated/full_case_table.tex`,
  `generated/raw_seed_table.tex`, `generated/raw_error_table.tex`, and
  `generated/variance_credit_table.tex`, `generated/anchor_coverage_table.tex`,
  `generated/length_imbalance_table.tex`, `generated/token_cost_table.tex`,
  `generated/closed_loop_training_table.tex`, and
  `generated/neural_generalization_table.tex` - reproducible LaTeX inputs generated
  from `results/deep_matrix_20seed.json`,
  `results/variance_credit_grid_seed17.json`,
  `results/anchor_coverage_audit_seedset.json`,
  `results/length_imbalance_audit_seedset.json`, and
  `results/token_cost_sensitivity_20seed.json`, the closed-loop training JSON
  artifacts, and `results/neural_credit_generalization_seedset.json`.

Build the paper from the repository root:

```bash
./scripts/build_latex_paper.sh
```

The script uses `tectonic` and writes:

```text
public/trajectory_rewards_are_not_token_credit.pdf
public/paper_manifest.json
```

The script also enforces a minimum 30-page rendered paper, so a truncated build
fails before the artifact is published. `paper/build/` is ignored scratch
output. The only tracked public paper PDF is
`public/trajectory_rewards_are_not_token_credit.pdf`.
