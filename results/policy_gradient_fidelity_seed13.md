# Policy-gradient fidelity audit

Finite MDP with exact value and occupancy-measure policy gradient.

| Method | Rel. mean err. | 95% CI | Var trace | Cosine | Mean batch dJ | P5 batch dJ | Neg. batch | Null abs A | Adv. r | Cal. MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `reinforce_return` | 0.075 | 0.020 | 0.018 | 0.997 | 0.156 | 0.132 | 0.000 | 0.610 | 0.432 | 0.036 |
| `sibling_loo_return` | 0.050 | 0.015 | 0.013 | 0.999 | 0.163 | 0.146 | 0.000 | 0.471 | 0.388 | 0.038 |
| `prefix_value_baseline` | 0.049 | 0.013 | 0.010 | 0.999 | 0.168 | 0.151 | 0.000 | 0.493 | 0.431 | 0.036 |
| `brpo_combined_baseline` | 0.048 | 0.014 | 0.011 | 0.999 | 0.167 | 0.150 | 0.000 | 0.488 | 0.419 | 0.037 |
| `learned_value_td` | 0.043 | 0.014 | 0.006 | 0.999 | 0.175 | 0.158 | 0.000 | 0.000 | 0.889 | 0.009 |
| `oracle_value_td` | 0.026 | 0.009 | 0.002 | 1.000 | 0.186 | 0.176 | 0.000 | 0.000 | 1.000 | 0.000 |

## Policy-implied actor coefficients

- `vimpo_actor_equal_ref`: ref KL 0.000, cos 0.000, mean err 1.000
- `vimpo_actor_fixed_ref_near`: ref KL 0.003, cos 0.659, mean err 0.896
- `vimpo_actor_fixed_ref_mid`: ref KL 0.018, cos 0.659, mean err 0.786
- `vimpo_actor_fixed_ref_far`: ref KL 0.060, cos 0.659, mean err 0.764

## Diagnostics

- Exact gradient norm: 0.160
- Finite-difference relative check error: 0.000
- Base return: 0.506
- Mean emitted tokens: 5.370
- Null token fraction: 0.069
