# Policy-gradient fidelity audit

Finite MDP with exact value and numerical exact policy gradient.

| Method | Bias/|g*| | Var trace | Cosine | KL-step dJ | Adv. r | Cal. MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `reinforce_return` | 0.108 | 0.022 | 0.994 | 0.190 | 0.430 | 0.037 |
| `sibling_loo_return` | 0.071 | 0.017 | 0.998 | 0.191 | 0.386 | 0.038 |
| `prefix_value_baseline` | 0.061 | 0.013 | 0.998 | 0.191 | 0.429 | 0.037 |
| `brpo_combined_baseline` | 0.063 | 0.014 | 0.998 | 0.191 | 0.416 | 0.037 |
| `critic_td` | 0.028 | 0.003 | 1.000 | 0.192 | 1.000 | 0.000 |
| `vimpo_equal_ref` | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045 |
| `vimpo_stale_ref` | 0.750 | 0.001 | 0.661 | 0.125 | 0.583 | 0.030 |

## Diagnostics

- Exact gradient norm: 0.160
- Base return: 0.506
- Mean emitted tokens: 5.388
- Null token fraction: 0.072
- Stale reference KL: 0.040
