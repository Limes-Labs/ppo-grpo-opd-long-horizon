# Policy-gradient fidelity audit

Finite MDP with exact value and occupancy-measure policy gradient.

| Method | Mean err/|g*| | 95% CI | Var trace | Cosine | KL-step dJ | Adv. r | Cal. MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `reinforce_return` | 0.062 | 0.020 | 0.018 | 0.998 | 0.193 | 0.433 | 0.036 |
| `sibling_loo_return` | 0.052 | 0.014 | 0.013 | 0.999 | 0.192 | 0.387 | 0.037 |
| `prefix_value_baseline` | 0.045 | 0.011 | 0.010 | 0.999 | 0.192 | 0.432 | 0.036 |
| `brpo_combined_baseline` | 0.047 | 0.013 | 0.011 | 0.999 | 0.192 | 0.419 | 0.036 |
| `learned_value_td` | 0.052 | 0.014 | 0.006 | 0.999 | 0.192 | 0.895 | 0.009 |
| `oracle_value_td` | 0.016 | 0.008 | 0.002 | 1.000 | 0.192 | 1.000 | 0.000 |

## Policy-implied actor coefficients

- `vimpo_actor_equal_ref`: ref KL 0.000, cos 0.000, mean err 1.000
- `vimpo_actor_fixed_ref_near`: ref KL 0.003, cos 0.658, mean err 0.896
- `vimpo_actor_fixed_ref_mid`: ref KL 0.018, cos 0.658, mean err 0.786
- `vimpo_actor_fixed_ref_far`: ref KL 0.060, cos 0.658, mean err 0.765

## Diagnostics

- Exact gradient norm: 0.160
- Finite-difference relative check error: 0.000
- Base return: 0.506
- Mean emitted tokens: 5.378
- Null token fraction: 0.070
