# Trajectory Rewards Are Not Token Credit

**Status:** Markdown companion summary for public draft v0.7, 2026-06-18
**Repository:** `Limes-Labs/ppo-grpo-opd-long-horizon`
**Scope:** method research for long-horizon language-model post-training, not a
model-specific report.

The canonical full paper artifact is `paper/main.tex` and the rendered LaTeX
PDF under `public/`. This Markdown file is kept as a readable companion summary.

## Abstract

Recent reasoning-model training has made critic-free reinforcement learning,
especially Group Relative Policy Optimization (GRPO), a central public topic.
GRPO is attractive because it avoids a learned value model and has produced
strong results in verifier-heavy math and reasoning settings. This paper studies
where that trade may change. In long-horizon, variable-length, agentic
trajectories, a single response-level group-relative advantage can blur
heterogeneous token roles: useful subgoals, harmful detours, retries, tool
calls, and no-op padding may all receive the same scalar update. Critic-based
PPO can in principle use token/state-level value estimates to recover temporal
credit, but only when the value target is learnable, sufficiently covered, and
worth its memory and stability cost. On-policy distillation (OPD) and
on-policy self-distillation (OPSD) are best treated as complementary
post-training objectives: they can transfer dense teacher/self-teacher signals
on student-visited trajectories, but they are not the same objective as reward
optimization.

We contribute a taxonomy, cost-accounting checklist, failure-mode table,
dependency-free estimator audits, a structural critic-free anchor-action
contrast baseline, an exploratory coverage-gated credit estimator, and a
tabular closed-loop training audit. In a 20-seed, 18-case matrix with known
oracle advantages, a critic-style TD estimator has higher mean
oracle-advantage correlation in 17 regimes. A confidence-interval reading makes
that 16 clear critic-favorable cases, 1 near tie, and 1 clear group-favorable
counterexample where the critic is blind and undercovered. The result supports
a conditional thesis: critic methods are promising for long heterogeneous
rollouts, but GRPO remains scientifically plausible and often cheaper when
reward contrast is reliable and value modeling is weak.

## 1. Contributions

1. **Method taxonomy.** We separate PPO-style and GRPO-style reward
   optimization from OPD/OPSD distillation by objective, signal granularity,
   variance-reduction mechanism, and extra-model cost.
2. **Conditional thesis.** We argue against a blanket "PPO beats GRPO" claim.
   The practical question is when temporal state information is more valuable
   than the critic's cost and instability.
3. **Cost accounting.** We list the model, rollout, verifier, teacher, and
   storage costs that should be charged before comparing methods.
4. **Toy evidence.** We provide fast CPU experiments with known oracle
   advantages, including a 20-seed matrix, variance/credit decomposition,
   anchor-coverage audit, length and token-cost robustness checks, and tabular
   closed-loop training.
5. **New baseline ideas.** We test anchor-action contrast and coverage-gated
   credit as modest candidate mechanisms, not as solved replacements for PPO or
   GRPO.
6. **Research protocol.** We define testable predictions for future Limes Labs
   work in `limes-autoresearch`, `limes-nanogpt`, EuroBench, and Parameter Golf.

## 2. Operational Definition of Long Horizon

For this workstream, a trajectory is long-horizon when at least one of the
following holds:

- The response has many decision tokens or tool actions before reward arrives.
- Reward is delayed until a final answer, test result, verifier call, or human
  preference judgment.
- Intermediate tokens have mixed causal roles: useful substeps, mistakes,
  corrections, retrieval/tool calls, no-op waiting, formatting, or repetition.
- The task state changes over time through files, tools, memory, environment
  observations, or compacted summaries.
- Credit assignment must distinguish "the trace eventually succeeded" from
  "this token/action helped the trace succeed."

This definition is intentionally broader than context length. A short trace with
tool state and delayed reward can be long-horizon; a long monologue with dense
teacher labels may not be.

## 3. Formal Setup

Let a policy model generate a trajectory

```text
tau = (s_0, a_0, s_1, a_1, ..., s_T)
```

where states may include prompt context, partial response, tool outputs, memory,
or verifier state. Let `R(tau)` be an outcome reward, possibly combined with
per-token KL or process rewards. The central question is how to estimate an
advantage signal for updating the policy.

### 3.1 PPO With A Critic

PPO optimizes a clipped policy-gradient surrogate [1]:

```math
L^{PPO}(\theta) =
E_t \left[
  \min \left(
    \rho_t(\theta) A_t,
    \operatorname{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) A_t
  \right)
\right],
```

where

```math
\rho_t(\theta) =
\frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}.
```

In actor-critic use, the advantage can be estimated from a value model:

```math
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
```

or with generalized advantage estimation [2]:

```math
A_t^{GAE(\gamma,\lambda)}
= \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}.
```

The critic is not free. It adds parameters, memory, training targets, forward
passes, and a new failure mode: a wrong value model can confidently assign wrong
credit.

### 3.2 GRPO

GRPO samples a group of completions for the same prompt, scores them, and
normalizes reward within the group [5, 6]. A simplified response-level advantage
for completion `i` is:

```math
\hat{A}_i^{GRPO} =
\frac{R_i - \mu_G}{\sigma_G + \epsilon},
```

where `mu_G` and `sigma_G` are computed over the sampled group. In common
language-model implementations, this scalar is then applied across the tokens of
that completion, often alongside KL regularization.

This critic-free design is memory-efficient and can work well with verifiable
rewards. But in terminal-reward, long-response settings, it may not distinguish
which parts of a successful response helped.

### 3.3 OPD

On-policy distillation trains the student on its own sampled outputs, with a
teacher providing dense supervision on those student-visited states. A typical
objective minimizes a divergence such as:

```math
L^{OPD} =
E_{x \sim D,\; y \sim \pi_s(\cdot|x)}
\left[
  \sum_t D_{KL}\left(
    \pi_t(\cdot | x, y_{<t})
    \;\|\;
    \pi_s(\cdot | x, y_{<t})
  \right)
\right].
```

GKD and related work motivate this on-policy framing as a way to reduce
train/inference distribution mismatch in autoregressive distillation [12].
Recent OPD papers study when the teacher actually provides new useful
information and when training becomes unstable [14, 15].

### 3.4 OPSD

OPSD removes the separate teacher model but not the teacher signal. A single
model plays two contextual roles: a teacher role conditioned on privileged
verified traces or solution information, and a student role conditioned on the
ordinary problem. Training minimizes a per-token divergence between those roles
over student rollouts [13]. This can be token-efficient, but it relies on
privileged traces and remains a distillation objective rather than direct reward
optimization.

## 4. Method Comparison

| Method | Objective | Signal granularity | Extra model/signal | Main strength | Main risk |
| --- | --- | --- | --- | --- | --- |
| PPO + critic | Optimize policy against reward/KL | token, state, trajectory | value model, reward/reference | temporal credit assignment | critic cost and miscalibration |
| GRPO | Optimize policy with group-normalized rewards | group/response scalar, sometimes token-broadcast | reward/verifier/reference, no critic | memory-efficient RLVR | weak intra-rollout credit, group pathologies |
| OPD | Distill teacher on student rollouts | token/logit/representation | teacher model | dense supervision on visited states | teacher mismatch, teacher cost |
| OPSD | Distill privileged self-teacher into ordinary student | token/logit | privileged traces, same model roles | no separate teacher model | privileged-context leakage, length/truncation issues |

## 5. Related Work

PPO was introduced as a simpler trust-region-like policy optimization method
[1], building on trust-region policy optimization [3]. PPO became central to
public RLHF practice in summarization and instruction-following systems [4, 16].

DeepSeekMath introduced GRPO as a PPO variant that removes the critic and
reduces memory usage for mathematical reasoning [5]. DeepSeek-R1 used
GRPO-style RL with rule-based rewards for reasoning, while also documenting
readability and language-mixing issues in pure RL and using multi-stage training
for DeepSeek-R1 [6]. Subsequent GRPO work has analyzed effective losses and
success amplification [7], response-length optimization bias and Dr. GRPO [8],
DAPO-style dynamic sampling and clipping changes [9], U-statistic theory [10],
and stabilized variants for low-dispersion or constrained reward settings
[11, 17, 18].

Z.ai's GLM-5.2 report is useful as an industry case study, not as independent
causal evidence [19]. The report describes long-horizon coding-agent rollouts
with compaction into variable numbers of sub-traces and says the training setup
moved from group-wise optimization to critic-based PPO with token-level
advantages for individual rollouts. This matches the failure mode studied here:
once rollouts have variable length and different internal token roles, a
response-level group scalar becomes less directly aligned with token credit.
The same report also describes OPD infrastructure, including parallel OPD
training to merge multiple expert models, so it should not be read as "PPO
replaces OPD." It is better evidence for a hybrid training stack.

OPD and OPSD form a neighboring but distinct branch. Generalized Knowledge
Distillation trains on student-generated outputs with teacher feedback [12].
OPSD uses privileged self-teacher contexts over student rollouts [13].
Recent OPD studies identify teacher/student compatibility, genuine teacher
novelty, length inflation, and truncation collapse as central issues [14, 15].

## 6. Cost Accounting Checklist

A fair comparison must charge more than final benchmark score:

- policy forward/backward passes and optimizer state
- old-policy log probabilities for clipped objectives
- reference-policy KL cost
- reward model, verifier, or unit-test calls
- group size and discarded/filtered samples
- value-model parameters, training data, and inference passes
- teacher/self-teacher logits or hidden states for OPD/OPSD
- generated tokens, failed rollouts, retries, and overlong traces
- trajectory storage, KV cache, and tool/environment state
- anti-hacking filters, LLM judges, sandboxing, blocked-call handling, and
  dummy observations when invalid tool calls are intercepted
- human or synthetic process-label cost
- engineering complexity and debugging cost

For Limes Labs, the near-term reporting unit should be:

```text
reward improvement per generated token,
per wall-clock minute,
per extra model-memory multiplier,
and per reproducible artifact.
```

## 7. Toy Experiment

### 7.1 Environment

The toy environment in `experiments/toy_credit_assignment.py` creates
variable-length trajectories with three token/action types:

- `help`: increases a hidden progress score.
- `harm`: decreases the score.
- `wait`: consumes a token without changing the score.

A terminal verifier returns success if the final score reaches a prompt-specific
threshold. The dynamics are known, so the experiment computes an oracle
state-action advantage:

```math
A^*(s_t,a_t) =
-c + V^*(s_{t+1}, h-1) - V^*(s_t, h),
```

where `c` is token cost and `h` is remaining horizon.

### 7.2 Estimators

The experiment compares:

1. **Group-relative estimator.** Normalize terminal returns within a prompt
   group and broadcast the scalar to every token in the trajectory.
2. **Critic-style estimator.** Fit a tabular value model from sampled
   returns-to-go, then compute a one-step TD advantage.

This is not a PPO-vs-GRPO training benchmark. It is a measurement of advantage
estimator quality under controlled dynamics.

### 7.3 Metrics

The report tracks:

- Pearson correlation with the oracle advantage
- calibrated MSE after affine rescaling
- raw MSE
- sign accuracy
- mean absolute advantage on `wait` vs active tokens
- within-trajectory estimator variance
- zero-variance group fraction
- critic exact-state hit rate

### 7.4 Smoke Sweep And Deep Matrix

The deterministic smoke sweep in `results/toy_sweep_seed11.md` runs six regimes
to keep CI fast. The canonical analysis for the paper is the 20-seed matrix in
`results/deep_matrix_20seed.md` and `results/deep_matrix_20seed.csv`. It runs
18 fixed cases across horizon length, group size, critic budget, observability,
a designed counterexample, and sparse-reward conditions.

The table reports mean Pearson correlation with the oracle token-level
advantage. Delta is `critic r - group r`; the confidence interval is an
across-seed 95% interval for that delta. "Mean winner" reports the sign of the
mean delta; "CI read" reports whether that delta is clearly positive, clearly
negative, or crosses zero.

| Case | Axis | Mean winner | CI read | Group r | Critic r | Delta r | 95% CI | Critic hit |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| horizon_04_baseline | horizon | critic | critic_clear | 0.495 | 0.977 | 0.482 | +/- 0.011 | 1.00 |
| horizon_08_baseline | horizon | critic | critic_clear | 0.390 | 0.925 | 0.535 | +/- 0.015 | 1.00 |
| horizon_12_baseline | horizon | critic | critic_clear | 0.335 | 0.866 | 0.531 | +/- 0.012 | 0.99 |
| horizon_16_long_wait | horizon | critic | critic_clear | 0.293 | 0.865 | 0.571 | +/- 0.013 | 1.00 |
| group_size_02_long_wait | group_size | critic | critic_clear | 0.213 | 0.771 | 0.557 | +/- 0.022 | 0.99 |
| group_size_04_long_wait | group_size | critic | critic_clear | 0.301 | 0.851 | 0.550 | +/- 0.014 | 0.99 |
| group_size_08_long_wait | group_size | critic | critic_clear | 0.346 | 0.910 | 0.564 | +/- 0.011 | 1.00 |
| group_size_12_long_wait | group_size | critic | critic_clear | 0.358 | 0.940 | 0.582 | +/- 0.007 | 1.00 |
| critic_budget_002_full | critic_budget | critic | near_tie | 0.352 | 0.367 | 0.015 | +/- 0.027 | 0.48 |
| critic_budget_008_full | critic_budget | critic | critic_clear | 0.352 | 0.459 | 0.107 | +/- 0.027 | 0.88 |
| critic_budget_032_full | critic_budget | critic | critic_clear | 0.352 | 0.735 | 0.383 | +/- 0.021 | 0.98 |
| critic_budget_128_full | critic_budget | critic | critic_clear | 0.352 | 0.906 | 0.554 | +/- 0.017 | 1.00 |
| observability_full | observability | critic | critic_clear | 0.352 | 0.864 | 0.513 | +/- 0.020 | 0.99 |
| observability_coarse | observability | critic | critic_clear | 0.356 | 0.738 | 0.382 | +/- 0.015 | 1.00 |
| observability_blind | observability | critic | critic_clear | 0.353 | 0.482 | 0.129 | +/- 0.017 | 1.00 |
| blind_undercovered_counterexample | counterexample | group | group_clear | 0.510 | 0.455 | -0.055 | +/- 0.015 | 0.69 |
| sparse_hard_group04 | sparse_reward | critic | critic_clear | 0.283 | 0.846 | 0.563 | +/- 0.014 | 0.99 |
| sparse_hard_group08 | sparse_reward | critic | critic_clear | 0.310 | 0.910 | 0.600 | +/- 0.008 | 1.00 |

Across the 18 cases, critic-style TD has the higher mean correlation in 17
cases, group-relative estimation wins by mean in one counterexample, and the
mean critic-minus-group correlation is 0.420. The confidence-interval reading is
more cautious: 16 clear critic-favorable cases, 1 near tie, and 1 clear
group-favorable case. The weakest critic-favorable case is
`critic_budget_002_full`, where the delta is only 0.015 +/- 0.027 and the exact
state hit rate is 0.48. Scientifically, this is close enough that the correct
reading is "coverage-limited critic roughly ties group-relative," not "PPO
dominates."

### 7.5 Interpretation

The toy supports four mechanism-level claims:

1. **Long heterogeneous traces punish scalar broadcast.** As wait-heavy traces
   grow, response-level group advantages correlate less with token-level oracle
   credit.
2. **Value estimates help when state is informative.** Full and coarse critics
   recover temporal structure because the state contains progress information.
3. **Coverage is the hinge.** Better critic coverage turns small or ambiguous
   advantages into large positive deltas; poor coverage can erase the benefit.
4. **Critics fail when observability or coverage fails.** A blind,
   undercovered critic can lose to group-relative terminal outcome information.

## 8. What Is Better, And Why?

The answer is conditional.

**PPO-style critic methods are better candidates when:**

- trajectories are long, variable-length, and internally heterogeneous;
- there are meaningful states or prefixes from which future return can be
  predicted;
- the critic has enough coverage and is evaluated for calibration;
- process rewards, tool states, or subgoal labels exist;
- memory budget allows value-model training and rollout storage.

**GRPO-style methods are better candidates when:**

- rewards are verifiable, cheap, and have useful within-prompt variation;
- trajectories are short or internally homogeneous;
- value modeling is too expensive, unobserved, or undercovered;
- group sampling can be scaled cheaply;
- stabilized variants handle low dispersion, length bias, and clipping details.

**OPD/OPSD are better candidates when:**

- a teacher or privileged self-teacher has genuinely new token-level knowledge;
- the goal is to transfer behavior rather than directly optimize an external
  reward;
- dense supervision is cheaper or more stable than RL;
- rollout length and truncation are actively controlled.

The likely best system is hybrid: use rejection sampling or GRPO-like RL to
discover successes, use value/process models where temporal credit matters, and
distill verified successful behaviors back into cheaper policies.

## 9. Failure Modes

### PPO / critic failures

- value overfitting to reward artifacts
- stale advantages after policy drift
- process reward misspecification
- critic collapse on long sparse trajectories
- high memory and engineering cost
- false confidence from calibrated-looking but wrong values

### GRPO failures

- zero-variance collapse when all group rewards match
- low-variance amplification under z-score normalization
- response-length bias and overlong incorrect answers
- scalar reward broadcast over harmful or irrelevant tokens
- group composition confounding prompt difficulty with policy quality
- weak signal when semantic diversity receives identical reward

### OPD / OPSD failures

- teacher and student thinking patterns are incompatible
- teacher provides no new information beyond the student
- on-policy rollouts inflate in length and truncate
- privileged traces leak solution structure
- KL copying transfers teacher mistakes
- dense token supervision hides high teacher-compute cost

## 10. Testable Predictions

1. Increasing horizon length and no-op padding should reduce the correlation of
   response-level GRPO advantages with process labels.
2. Increasing group size should help GRPO only when it increases meaningful
   reward dispersion.
3. Adding process or subgoal rewards should narrow the PPO/GRPO credit gap.
4. Degrading critic observability should produce regimes where GRPO wins.
5. OPD/OPSD should improve token efficiency when teacher signals are novel and
   rollout lengths are controlled, but degrade when teacher/student support
   diverges.
6. Hybrid pipelines should outperform single-method pipelines when they charge
   all teacher, critic, verifier, and rollout costs honestly.

## 11. Reproducibility

Run all tests:

```bash
python3 -m unittest discover -s tests
```

Run the smoke gate:

```bash
./scripts/run_smoke.sh
```

Regenerate the sweep:

```bash
python3 -m experiments.scenario_sweep \
  --seed 11 \
  --output-json results/toy_sweep_seed11.json \
  --output-md results/toy_sweep_seed11.md
```

Regenerate the 20-seed matrix and SVG figures:

```bash
python3 -m experiments.deep_matrix \
  --output-json results/deep_matrix_20seed.json \
  --output-csv results/deep_matrix_20seed.csv \
  --output-md results/deep_matrix_20seed.md \
  --figures-dir results/figures
```

Regenerate the public PDF and DOCX artifacts:

```bash
python3 scripts/build_public_artifacts.py \
  --matrix-json results/deep_matrix_20seed.json \
  --public-dir public
```

The core experiment code has no external Python dependencies. The PDF/DOCX
builder requires `reportlab`, `python-docx`, and `Pillow`. The published
numbers are deterministic for the committed code and Python's `random.Random`
behavior.

The rendered PDF and DOCX are abridged public report artifacts derived from the
canonical matrix, not a full export of this `PAPER.md` manuscript. The PDF was
rendered to PNG pages and visually inspected in the local environment. The DOCX
is structurally checked for text, tables, and embedded chart images; a visual
DOCX render was not run because LibreOffice/`soffice` is unavailable locally.

## 12. Limitations

This paper is not a claim about frontier-scale training. The toy environment has
a finite known state, an oracle value function for evaluation, and a tabular
critic. Real post-training adds neural approximation, KL controllers, reward
model errors, truncation, tool failures, distributed rollout systems, stale
policy samples, optimizer details, and data contamination risks. The toy
therefore supports mechanism hypotheses, not leaderboard conclusions.

The GRPO baseline is intentionally simple. Future work should compare
leave-one-out baselines, length-normalized variants, DAPO/Dr. GRPO-style
corrections, process rewards, and dynamic sampling before making stronger
claims.

The GLM-5.2 discussion is based on a public vendor report and local text
snapshot, not on independent reproduction of its training stack. We use it as a
case study for problem structure, not as proof that PPO caused the reported
performance.

## 13. Safety and Ethics

Long-horizon agentic RL can improve tool use and reasoning, but it can also
amplify reward hacking, deception, unsafe tool strategies, and benchmark
overfitting. Public Limes experiments should start with toy and benchmark-safe
settings, publish negative results, avoid hidden capability claims, and separate
benign evaluation tasks from cyber, persuasion, or critical-infrastructure
actions that could create misuse risk.

## 14. Conclusion

The evidence so far favors a conditional research program: critic-based PPO is a
strong candidate for long heterogeneous trajectories when value targets are
learnable and adequately covered; GRPO is attractive when group reward contrast
is reliable and a critic is too costly or poorly observed; OPD/OPSD are
complementary distillation tools that can transfer dense behavior signals but do
not replace reward optimization as a concept. The next step is to move from this
finite-state toy into tiny neural policies, then into reproducible language and
agent benchmarks with honest cost accounting.

## References

[1] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov.
"Proximal Policy Optimization Algorithms." 2017.
<https://arxiv.org/abs/1707.06347>

[2] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel.
"High-Dimensional Continuous Control Using Generalized Advantage Estimation."
2015. <https://arxiv.org/abs/1506.02438>

[3] John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter
Abbeel. "Trust Region Policy Optimization." 2015.
<https://arxiv.org/abs/1502.05477>

[4] Long Ouyang et al. "Training language models to follow instructions with
human feedback." 2022. <https://arxiv.org/abs/2203.02155>

[5] Zhihong Shao et al. "DeepSeekMath: Pushing the Limits of Mathematical
Reasoning in Open Language Models." 2024. <https://arxiv.org/abs/2402.03300>

[6] DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
Reinforcement Learning." 2025. <https://arxiv.org/abs/2501.12948>

[7] Youssef Mroueh. "Reinforcement Learning with Verifiable Rewards: GRPO's
Effective Loss, Dynamics, and Success Amplification." 2025.
<https://arxiv.org/abs/2503.06639>

[8] Zichen Liu et al. "Understanding R1-Zero-Like Training: A Critical
Perspective." 2025. <https://arxiv.org/abs/2503.20783>

[9] Qiying Yu et al. "DAPO: An Open-Source LLM Reinforcement Learning System at
Scale." 2025. <https://arxiv.org/abs/2503.14476>

[10] Hongyi Zhou et al. "Demystifying Group Relative Policy Optimization: Its
Policy Gradient is a U-Statistic." 2026. <https://arxiv.org/abs/2603.01162>

[11] Mohammad Mahdi Salmani-Zarchi et al. "MDP-GRPO: Stabilized Group Relative
Policy Optimization for Multi-Constraint Instruction Following." 2026.
<https://arxiv.org/abs/2606.06058>

[12] Rishabh Agarwal et al. "On-Policy Distillation of Language Models:
Learning from Self-Generated Mistakes." 2023.
<https://arxiv.org/abs/2306.13649>

[13] Siyan Zhao et al. "Self-Distilled Reasoner: On-Policy Self-Distillation
for Large Language Models." 2026. <https://arxiv.org/abs/2601.18734>

[14] Yaxuan Li et al. "Rethinking On-Policy Distillation of Large Language
Models: Phenomenology, Mechanism, and Recipe." 2026.
<https://arxiv.org/abs/2604.13016>

[15] Feng Luo et al. "Demystifying OPD: Length Inflation and Stabilization
Strategies for Large Language Models." 2026.
<https://arxiv.org/abs/2604.08527>

[16] Nisan Stiennon et al. "Learning to summarize from human feedback." 2020.
<https://arxiv.org/abs/2009.01325>

[17] Tue Le, Nghi D. Q. Bui, Linh Ngo Van, Trung Le. "Token-Regulated Group
Relative Policy Optimization for Stable Reinforcement Learning in Large
Language Models." 2025. <https://arxiv.org/abs/2511.00066>

[18] Roger Girgis et al. "Constrained Group Relative Policy Optimization." 2026.
<https://arxiv.org/abs/2602.05863>

[19] Z.ai. "GLM-5.2: Built for Long-Horizon Tasks." 2026.
<https://z.ai/blog/glm-5.2>
