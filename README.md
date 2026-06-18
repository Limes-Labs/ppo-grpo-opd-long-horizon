# PPO, GRPO, OPD For Long-Horizon Post-Training

This is a public Limes Labs research workstream on long-horizon post-training
methods for language-model agents and reasoning systems. The focus is the method
question, not a single base model:

> For variable-length, multi-step, tool-heavy trajectories, when does a
> critic/value-model PPO setup recover useful token/state-level credit that a
> group-relative GRPO setup discards or blurs?

This is a hypothesis to test, not a settled claim. GRPO has clear practical
advantages: it removes a learned critic, reduces memory pressure, and has worked
well in verifier-heavy reasoning settings. PPO has its own costs: critic
training can be unstable, reward/process supervision can be wrong, and
trajectory storage is expensive. OPD and OPSD belong in the comparison as
complementary on-policy distillation/self-distillation tools, not as direct
drop-in replacements for policy optimization.

## What Is Here

- `PAPER.md` - Markdown working draft with formal setup, equations, taxonomy,
  cost accounting, toy results, limitations, and references.
- `paper/main.tex` and `paper/references.bib` - LaTeX manuscript source for
  the canonical full paper-formatted artifact.
- `experiments/toy_credit_assignment.py` - CPU-only toy experiment comparing a
  GRPO-style group-relative advantage with a learned critic-style TD estimator
  on variable-length trajectories.
- `experiments/scenario_sweep.py` - deterministic multi-scenario sweep covering
  critic-favorable regimes and a group-favorable counterexample.
- `experiments/deep_matrix.py` - 20-seed, 18-case matrix over horizon length,
  group size, critic budget, observability, and sparse rewards.
- `experiments/variance_credit_grid.py` - estimator grid separating variance
  reduction mechanisms from step-level credit assignment.
- `experiments/length_imbalance_audit.py` - audit of group-relative estimators
  under increasing within-group trajectory-length imbalance.
- `experiments/token_cost_sensitivity.py` - reward-shaping robustness check
  over token costs, including the zero-cost case.
- `results/toy_sweep_seed11.md` - committed sweep report for the current draft.
- `results/deep_matrix_20seed.md` - canonical multi-seed result table used for
  the public PDF/DOCX paper artifacts.
- `results/variance_credit_grid_seed17.md` - canonical result table for the
  variance-reduction versus credit-assignment grid.
- `results/length_imbalance_audit_seedset.md` - canonical length-imbalance
  audit table.
- `results/token_cost_sensitivity_20seed.md` - canonical token-cost
  sensitivity table.
- `public/ppo_grpo_opd_long_horizon.pdf` and
  `public/ppo_grpo_opd_long_horizon.docx` - abridged rendered public report
  artifacts with charts and result tables. `paper/main.tex` is the full
  manuscript source.
- `public/ppo_grpo_opd_long_horizon_latex.pdf` - full 30-page LaTeX-built
  paper with generated result appendices.
- `public/latex_artifact_manifest.json` - SHA-256 manifest for the full LaTeX
  paper artifact.
- `public/artifact_manifest.json` - SHA-256 manifest for the rendered
  artifacts and source experiment JSON.
- `tests/` - unit tests for the toy experiment and CLI artifact.
- `scripts/run_smoke.sh` - one-command smoke runner.
- `scripts/build_public_artifacts.py` - PDF/DOCX/chart builder. The core
  experiments are dependency-free; artifact generation uses `reportlab`,
  `python-docx`, and `Pillow`.
- `docs/research-roadmap.md` - integration plan for Limes AutoResearch,
  nanoGPT, EuroBench, and Parameter Golf.

## Quickstart

Requirements:

- Python 3.10 or newer
- No external Python packages

Run the smoke check:

```bash
./scripts/run_smoke.sh
```

Run tests only:

```bash
python3 -m unittest discover -s tests
```

Run the toy experiment manually:

```bash
python3 -m experiments.toy_credit_assignment \
  --scenario baseline \
  --seed 11 \
  --train-groups 120 \
  --eval-groups 40 \
  --group-size 6 \
  --max-steps 10 \
  --output runs/toy_credit_assignment_smoke.json
```

Regenerate the scenario sweep used in the paper:

```bash
python3 -m experiments.scenario_sweep \
  --seed 11 \
  --output-json results/toy_sweep_seed11.json \
  --output-md results/toy_sweep_seed11.md
```

Regenerate the canonical 20-seed matrix:

```bash
python3 -m experiments.deep_matrix \
  --output-json results/deep_matrix_20seed.json \
  --output-csv results/deep_matrix_20seed.csv \
  --output-md results/deep_matrix_20seed.md \
  --figures-dir results/figures
```

Regenerate the variance-reduction versus credit-assignment grid:

```bash
python3 -m experiments.variance_credit_grid \
  --output-json results/variance_credit_grid_seed17.json \
  --output-md results/variance_credit_grid_seed17.md
```

Regenerate the length-imbalance and token-cost audits:

```bash
python3 -m experiments.length_imbalance_audit \
  --output-json results/length_imbalance_audit_seedset.json \
  --output-md results/length_imbalance_audit_seedset.md

python3 -m experiments.token_cost_sensitivity \
  --output-json results/token_cost_sensitivity_20seed.json \
  --output-md results/token_cost_sensitivity_20seed.md
```

Regenerate PDF/DOCX paper artifacts with a Python environment that includes
`reportlab`, `python-docx`, and `Pillow`:

```bash
python3 scripts/build_public_artifacts.py \
  --matrix-json results/deep_matrix_20seed.json \
  --public-dir public
```

Build the full LaTeX paper:

```bash
./scripts/build_latex_paper.sh
```

The LaTeX build regenerates the result macros and appendix tables from
`results/deep_matrix_20seed.json` and
`results/variance_credit_grid_seed17.json`, plus the length-imbalance and
token-cost audit JSON files. It compiles the paper with `tectonic` and checks
that the rendered PDF is at least 30 pages.

The output JSON records correlation, calibrated MSE, sign accuracy, and leakage
metrics for both estimators. The toy is deliberately synthetic: it tests a
credit-assignment mechanism, not model quality.

## Current Experiment Result

The canonical 20-seed matrix runs 18 fixed regimes. The critic-style estimator
wins by mean oracle-advantage correlation in 17 regimes, but one of those is a
near tie whose 95% confidence interval crosses zero. The more careful reading is
16 clear critic-favorable cases, 1 near tie, and 1 clear group-favorable
counterexample where the critic is blind and undercovered:

| Regime | Winner | Group r | Critic r |
| --- | --- | ---: | ---: |
| horizon 4 baseline | critic | 0.495 | 0.977 |
| horizon 16 long wait | critic | 0.293 | 0.865 |
| critic budget 2 full-state | near tie by CI | 0.352 | 0.367 |
| critic budget 128 full-state | critic | 0.352 | 0.906 |
| observability blind | critic | 0.353 | 0.482 |
| blind undercovered counterexample | group | 0.510 | 0.455 |
| sparse hard, group size 8 | critic | 0.310 | 0.910 |

Read this as mechanism evidence, not a leaderboard: value information helps when
it is observable and covered; group-relative terminal rewards can be more useful
when the critic is weak or unavailable. The committed matrix includes raw
per-seed rows and 95% confidence intervals in `results/deep_matrix_20seed.csv`
and `results/deep_matrix_20seed.md`.

The PDF was rendered and visually inspected through PNG page renders. The DOCX
is structurally checked for tables, text, and embedded charts; a full visual
DOCX render was not run because LibreOffice/`soffice` is unavailable in the
current local environment.

The variance-credit grid adds the missing mechanism decomposition. In the
canonical long-wait run, a global baseline reduces the REINFORCE second moment
without creating within-trajectory credit variation, while learned critic TD
and sampled Monte Carlo value estimates create step-level variation and improve
oracle-advantage correlation over sibling group normalization.

The length-imbalance audit tests a simple length-adjusted group baseline. As
maximum horizon grows from 4 to 20, mean within-group length range grows from
1.775 to 13.439 tokens; at the longest horizon, critic TD reaches `r=0.811`
while group total return is `r=0.266` and length-adjusted group return is
`r=0.260`. The token-cost audit checks that this is not just the explicit
per-token penalty: in the long-wait zero-cost row, critic-minus-group
correlation is `+0.552` over 20 seeds.

## Working Thesis

Long-horizon agentic RL creates variable-length traces with subgoals, tool
calls, delays, retries, and compaction. In that regime:

- PPO with a value model can, in principle, support token-level or state-level
  advantage estimation and temporal credit assignment.
- GRPO can be memory-efficient and effective when groups contain informative
  reward variation, but terminal response-level group normalization can blur
  token credit in long heterogeneous rollouts.
- OPD and OPSD can provide dense distillation signals on student/on-policy
  trajectories. They are different objectives from reward optimization, although
  they can still compete as practical post-training recipes.

The first experiments here are designed to make those claims falsifiable before
moving to real language-model training.

## Source Trail

Primary and near-primary sources covered in the first outline include:

- PPO: Schulman et al., 2017, <https://arxiv.org/abs/1707.06347>
- DeepSeekMath and GRPO: Shao et al., 2024,
  <https://arxiv.org/abs/2402.03300>
- DeepSeek-R1 and GRPO RL: DeepSeek-AI, 2025,
  <https://arxiv.org/abs/2501.12948>
- OPSD: Zhao et al., 2026, <https://arxiv.org/abs/2601.18734>
- R1-Zero-like GRPO analysis / Dr. GRPO: Liu et al., 2025,
  <https://arxiv.org/abs/2503.20783>
- DAPO: Yu et al., 2025, <https://arxiv.org/abs/2503.14476>
- GRPO effective loss: Mroueh, 2025, <https://arxiv.org/abs/2503.06639>
- OPD dynamics and failure modes: Li et al., 2026,
  <https://arxiv.org/abs/2604.13016>
- StableOPD length inflation: Luo et al., 2026,
  <https://arxiv.org/abs/2604.08527>
- GRPO theory as U-statistic: Zhou et al., 2026,
  <https://arxiv.org/abs/2603.01162>
- Stabilized GRPO variants: Salmani-Zarchi et al., 2026,
  <https://arxiv.org/abs/2606.06058>
- GLM-5.2 as a vendor case study for critic-based PPO returning in compacted
  long-horizon agentic RL: Z.ai, 2026, <https://z.ai/blog/glm-5.2>
- Variance-reduction and step-credit references added in the LaTeX paper:
  REINFORCE++, RLOO/Back to Basics, Single-stream Policy Optimization,
  VinePPO, ArCHer, process supervision, GiGPO, SALT, RUDDER, Segment Policy
  Optimization, and OPPO.

## Non-Claims

This repository does not claim that PPO is always better than GRPO, that OPD is
inferior to RL, or that the toy experiment predicts frontier-model behavior. The
current claim is narrower: critic-style estimators are promising when temporal
state information is learnable; group-relative methods are attractive when group
reward contrast is reliable and value modeling is weak or too expensive.
