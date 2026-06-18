# PPO vs GRPO vs OPD/OPSD For Long-Horizon Post-Training

Status: first public outline, 2026-06-18.

## Abstract

Recent reasoning-model post-training has made critic-free methods such as Group
Relative Policy Optimization (GRPO) highly visible. GRPO is attractive because
it removes the value model used in PPO-style actor-critic training and can work
well with verifier rewards. This outline asks where that trade changes: in
long-horizon, variable-length, agentic trajectories, response-level
group-relative advantages may discard useful temporal structure. The working
hypothesis is that critic-based PPO becomes more attractive when state/token
credit assignment matters, while GRPO remains useful for memory-efficient
verifier-driven training and OPD/OPSD provide complementary distillation signals.

## Definitions

**PPO.** Proximal Policy Optimization optimizes a clipped policy-gradient
surrogate over on-policy samples. In actor-critic RLHF-style use, a value model
estimates returns and supports advantage estimates, often with temporal
difference or GAE-style credit assignment. Source: Schulman et al. 2017.

**GRPO.** Group Relative Policy Optimization samples multiple completions for a
prompt and normalizes rewards within the group, avoiding a separate value model.
DeepSeekMath introduced GRPO as a PPO variant for mathematical reasoning, and
DeepSeek-R1 used GRPO-style RL for reasoning models.

**OPD.** On-policy distillation lets the student generate trajectories under its
own policy, then uses a teacher signal on those student-visited states. The
signal can be token-level KL, outcome-guided, representation-level, or hybrid.

**OPSD.** On-policy self-distillation is a self-teaching variant where a single
model plays teacher and student under different contexts, for example teacher
access to privileged verified traces and student access only to the question.

## Taxonomy

| Method | Main objective | Typical signal granularity | Extra model | Strength | Long-horizon risk |
| --- | --- | --- | --- | --- | --- |
| PPO with critic | Policy optimization against reward/KL | token, state, trajectory | value/critic, often reward/reference | temporal credit assignment | critic cost and miscalibration |
| GRPO | Policy optimization with group-normalized rewards | usually response/group scalar broadcast to tokens | reference/reward, no critic | memory-efficient verifier RL | group variance, length bias, weak intra-rollout credit |
| OPD | Distill teacher behavior on student rollouts | token/logit or representation | teacher | dense supervision on visited states | teacher cost, mismatch, length inflation |
| OPSD | Distill privileged self-teacher into ordinary self-student | token/logit | same model in teacher/student roles | no external teacher, uses verified traces | privileged-context leakage, not reward optimization |

## Cost Accounting

Any fair comparison should charge:

- policy forward/backward cost and optimizer state
- reference-policy log-prob cost for KL control
- reward/verifier calls and reward-model training
- group size and number of rollouts per prompt
- value-model parameters, forward passes, training targets, and memory for PPO
- teacher forward passes and logits/hidden states for OPD/OPSD
- trajectory length, truncation, KV cache, and storage
- failed rollouts, retries, and dynamic sampling filters

The first Limes experiments should report "tokens generated per useful gradient
signal" and "additional model-memory multiplier" alongside task metrics.

## Failure Modes

**PPO failure modes.** Value targets can be noisy or biased; critics can overfit
reward-model artifacts; stale advantages can destabilize updates; process
rewards can teach shortcuts; full actor-critic training can exceed small-lab
memory budgets.

**GRPO failure modes.** Homogeneous binary rewards can produce zero-variance
groups; z-score normalization can amplify low-dispersion noise; response-level
advantages can reward every token in a successful long trace, including delay or
repetition; group composition can confound prompt difficulty with policy quality;
length pressure can be biased if not handled carefully.

**OPD/OPSD failure modes.** Teacher and student may share too little or too much
behavioral support; a teacher may add no genuinely new capability; rollouts can
inflate in length and hit truncation; token-level KL can copy teacher mistakes;
privileged teacher contexts can leak solution structure unless controlled.

## Toy Experiment

The included toy experiment samples variable-length trajectories with helpful,
harmful, and wait tokens. A terminal verifier gives success/failure; known
dynamics give an oracle state-action advantage. Two estimators are compared:

- **group-relative:** normalize terminal returns within a prompt group and
  broadcast the resulting scalar over all tokens in each trajectory.
- **critic-style:** learn a tabular value model from sampled returns-to-go and
  compute a one-step TD advantage.

The expected result is not "PPO wins"; it is narrower: when a trajectory contains
heterogeneous token roles, a learned value estimator can recover more temporal
structure than a response-level group scalar.

## Testable Predictions

1. On variable-length traces with many neutral or corrective tokens, critic-style
   advantages should show higher correlation with process labels than
   response-level group advantages.
2. GRPO should improve when group sampling increases reward dispersion and when
   rewards contain process or subgoal information.
3. GRPO should struggle most when binary rewards are homogeneous within groups or
   when long successful traces contain many irrelevant tokens.
4. OPD/OPSD should be strongest when the teacher has genuinely new, reliable
   token-level information on student-visited states.
5. A hybrid recipe may be best: GRPO or rejection sampling for broad exploration,
   PPO/value models for temporal credit, OPD/OPSD for distilling verified
   successful behavior back into smaller or cheaper policies.

## Roadmap For The Paper

- Expand each source summary into a one-page evidence card.
- Reproduce the toy experiment with ablations over group size, horizon length,
  reward sparsity, and value-model data.
- Move from tabular dynamics to tiny language/action policies in
  `limes-nanogpt`.
- Use `limes-autoresearch` to log runs and prevent cherry-picking.
- Connect to EuroBench agentic tasks once they expose verifiable subgoals.

## References

- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov.
  "Proximal Policy Optimization Algorithms." 2017.
  <https://arxiv.org/abs/1707.06347>
- Zhihong Shao et al. "DeepSeekMath: Pushing the Limits of Mathematical
  Reasoning in Open Language Models." 2024. <https://arxiv.org/abs/2402.03300>
- DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
  Reinforcement Learning." 2025. <https://arxiv.org/abs/2501.12948>
- Zichen Liu et al. "Understanding R1-Zero-Like Training: A Critical
  Perspective." 2025. <https://arxiv.org/abs/2503.20783>
- Qiying Yu et al. "DAPO: An Open-Source LLM Reinforcement Learning System at
  Scale." 2025. <https://arxiv.org/abs/2503.14476>
- Youssef Mroueh. "Reinforcement Learning with Verifiable Rewards: GRPO's
  Effective Loss, Dynamics, and Success Amplification." 2025.
  <https://arxiv.org/abs/2503.06639>
- Siyan Zhao et al. "Self-Distilled Reasoner: On-Policy Self-Distillation for
  Large Language Models." 2026. <https://arxiv.org/abs/2601.18734>
- Yaxuan Li et al. "Rethinking On-Policy Distillation of Large Language Models:
  Phenomenology, Mechanism, and Recipe." 2026.
  <https://arxiv.org/abs/2604.13016>
- Feng Luo et al. "Demystifying OPD: Length Inflation and Stabilization
  Strategies for Large Language Models." 2026.
  <https://arxiv.org/abs/2604.08527>
- Hongyi Zhou et al. "Demystifying Group Relative Policy Optimization: Its
  Policy Gradient is a U-Statistic." 2026.
  <https://arxiv.org/abs/2603.01162>
- Mohammad Mahdi Salmani-Zarchi et al. "MDP-GRPO: Stabilized Group Relative
  Policy Optimization for Multi-Constraint Instruction Following." 2026.
  <https://arxiv.org/abs/2606.06058>

