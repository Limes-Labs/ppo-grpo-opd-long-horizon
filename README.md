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

- `PAPER.md` - first paper-style outline with definitions, taxonomy,
  comparison table, cost accounting, failure modes, predictions, and references.
- `experiments/toy_credit_assignment.py` - CPU-only toy experiment comparing a
  GRPO-style group-relative advantage with a learned critic-style TD estimator
  on variable-length trajectories.
- `tests/` - unit tests for the toy experiment and CLI artifact.
- `scripts/run_smoke.sh` - one-command smoke runner.
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
  --seed 11 \
  --train-groups 120 \
  --eval-groups 40 \
  --group-size 6 \
  --max-steps 10 \
  --output runs/toy_credit_assignment_smoke.json
```

The output JSON records correlation, calibrated MSE, sign accuracy, and leakage
metrics for both estimators. The toy is deliberately synthetic: it tests a
credit-assignment mechanism, not model quality.

## Working Thesis

Long-horizon agentic RL creates variable-length traces with subgoals, tool
calls, delays, retries, and compaction. In that regime:

- PPO with a value model can, in principle, support token-level or state-level
  advantage estimation and temporal credit assignment.
- GRPO can be memory-efficient and effective when groups contain informative
  reward variation, but response-level group normalization can waste information
  in long rollouts by broadcasting one scalar over many heterogeneous tokens.
- OPD and OPSD can provide dense distillation signals on student/on-policy
  trajectories, but they answer a different question: how to transfer or
  self-transfer behavior, not how to optimize a policy against an external
  reward with temporal credit assignment.

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

## Non-Claims

This repository does not claim that PPO is always better than GRPO, that OPD is
inferior to RL, or that the toy experiment predicts frontier-model behavior. It
is a small public scaffold for asking sharper questions and building reproducible
evidence.

