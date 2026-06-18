# Closed-Loop Credit Training

A tabular softmax policy is trained directly in the toy environment under
matched generated-token budgets. Coverage-gated credit is the hybrid
proposal: use critic TD only for states with replay coverage, otherwise
fall back to group-relative credit.

| Method | Initial return | Final return | Delta return | Final success | Final wait | Critic frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| group_total | 0.515 | 0.785 | 0.271 | 0.949 | 0.539 | 0.000 |
| group_length_norm | 0.515 | 0.782 | 0.267 | 0.945 | 0.539 | 0.000 |
| critic_td | 0.515 | 0.810 | 0.296 | 0.974 | 0.563 | 0.998 |
| coverage_gated | 0.515 | 0.809 | 0.294 | 0.972 | 0.562 | 0.996 |

Summary:
- Best by final return: critic_td
- Coverage-gated minus group return: 0.024
- Coverage-gated minus group success: 0.024
