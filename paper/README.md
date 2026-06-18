# LaTeX Paper

This directory contains the full LaTeX manuscript source for the public paper:

- `main.tex` - article source with equations, figures, tables, and caveats.
- `references.bib` - bibliography for primary and near-primary sources.
- `generated/result_macros.tex` and `generated/deep_matrix_table.tex` -
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

The LaTeX PDF is the full paper artifact. The existing
`public/ppo_grpo_opd_long_horizon.pdf` and `.docx` files are abridged public
report artifacts generated from `scripts/build_public_artifacts.py`.
