# Sequence-Policy Training Audit

A shared one-hidden-layer autoregressive MLP policy samples action tokens
from prompt and prefix-derived sequence state. The audit compares group
broadcast credit with learned neural value-TD credit under matched rollout
budgets. This is a synthetic sequence-policy mechanism check, not an
LLM-scale transformer benchmark.

| Method | Init R | Final R | Delta R | Success | Wait frac. | Critic MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Group broadcast | 0.171 | 0.811 | 0.640 | 0.912 | 0.044 | 0.000 |
| Neural value TD | 0.171 | 0.839 | 0.668 | 0.940 | 0.004 | 0.065 |

Summary:
- Policy family: autoregressive_mlp
- 10 paired seeds
- Neural minus group final return: 0.028
- Neural minus group final return 95% paired CI: [0.007, 0.050]
- Neural minus group success: 0.028
- Neural minus group wait fraction: -0.040
