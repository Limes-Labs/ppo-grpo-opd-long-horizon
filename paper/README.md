# LaTeX Paper

This directory contains the full LaTeX manuscript source for the public paper:

- `main.tex` - article source with equations, figures, tables, and caveats.
- `references.bib` - bibliography for primary and near-primary sources.
- `generated/result_macros.tex`, `generated/deep_matrix_table.tex`,
  `generated/axis_summary_table.tex`, `generated/full_case_table.tex`,
  `generated/raw_seed_table.tex`, and `generated/raw_error_table.tex` -
  reproducible LaTeX inputs generated from `results/deep_matrix_20seed.json`.

Build the paper from the repository root:

```bash
./scripts/build_latex_paper.sh
```

The script uses `tectonic` and writes:

```text
public/ppo_grpo_opd_long_horizon_latex.pdf
public/latex_artifact_manifest.json
```

The script also enforces a minimum 30-page rendered paper, so a truncated build
fails before the artifact is published. The LaTeX PDF is the full paper
artifact. The existing
`public/ppo_grpo_opd_long_horizon.pdf` and `.docx` files are abridged public
report artifacts generated from `scripts/build_public_artifacts.py`.
