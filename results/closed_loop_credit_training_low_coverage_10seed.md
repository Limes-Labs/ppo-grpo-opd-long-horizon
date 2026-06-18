# Closed-Loop Credit Training

A tabular softmax policy is trained directly in the toy environment under
matched generated-token budgets. Coverage-gated credit is the hybrid
proposal: use critic TD only for states with replay coverage, otherwise
fall back to group-relative credit.

| Method | Initial return | Final return | Delta return | Final success | Final wait | Critic frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| group_total | 0.508 | 0.769 | 0.261 | 0.928 | 0.539 | 0.000 |
| group_length_norm | 0.508 | 0.754 | 0.246 | 0.913 | 0.538 | 0.000 |
| critic_td | 0.508 | 0.780 | 0.272 | 0.940 | 0.560 | 0.945 |
| coverage_gated | 0.508 | 0.777 | 0.269 | 0.937 | 0.548 | 0.644 |

Summary:
- Best by final return: critic_td
- Coverage-gated minus group return: 0.008
- Coverage-gated minus group success: 0.008
