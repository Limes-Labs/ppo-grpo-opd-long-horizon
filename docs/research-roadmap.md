# Research Roadmap

This repo should be the method-analysis layer for Limes Labs' public
post-training work. It should not become a separate island.

## Limes AutoResearch

Use `limes-autoresearch` as the run ledger and promotion gate:

- add a research-question spec for "long-horizon credit assignment"
- run toy ablations over group size, horizon length, reward sparsity, and value
  model data
- store JSON artifacts and result cards
- require negative-result notes before moving to larger models

## Limes nanoGPT

Use `limes-nanogpt` for the first neural policy experiments:

- port the toy dynamics into a tiny sequence policy
- compare response-level GRPO advantages with PPO/value-model advantages under
  identical token budgets
- add OPD/OPSD-inspired distillation only after the RL baselines are measured
- report memory and generated-token cost, not only final reward

## EuroBench

Use EuroBench when tasks require real language and agent behavior:

- define long-horizon task shards with explicit subgoals and verifiable final
  answers
- separate public smoke tasks from future hidden or reviewed tasks
- include failure annotations: wrong subgoal, tool misuse, late correction,
  repetition, and truncation
- avoid leaderboard claims until scoring and human review are stable

## Parameter Golf

Use Parameter Golf for efficiency pressure:

- charge critic, verifier, teacher, and reference-model costs
- track artifact-size and training-time budgets
- ask whether a method improves reward per byte, per generated token, and per
  wall-clock minute
- preserve simple baselines so clever methods do not hide compute overhead

## First Milestones

1. Stabilize this repo's toy experiment and source outline.
2. Add AutoResearch configs that replay toy ablations.
3. Port one tiny PPO-vs-GRPO comparison into `limes-nanogpt`.
4. Draft one EuroBench long-horizon task schema with process labels.
5. Publish a result card with limitations and at least one negative finding.

## Experiment Ladder

The next scientific goal is a phase diagram, not a single winner.

1. **Toy estimator phase:** sweep horizon, wait/no-op load, reward sparsity,
   group size, critic state coverage, and critic data budget. Keep oracle
   advantages so estimator quality is measurable.
2. **Tiny neural phase:** move the same dynamics into `limes-nanogpt` with
   actual learned policies, PPO-style value heads, and GRPO-style group
   objectives under identical generated-token budgets.
3. **Distillation phase:** add OPD/OPSD-style teacher signals only after PPO and
   GRPO baselines are stable. Charge teacher/self-teacher compute explicitly.
4. **Benchmark phase:** use EuroBench tasks with verifiable subgoals, tool
   state, retries, and process labels. Report failures by category rather than
   only aggregate score.
5. **Efficiency phase:** use Parameter Golf-style accounting to ask which method
   improves reward per byte, generated token, wall-clock minute, and extra model
   multiplier.
