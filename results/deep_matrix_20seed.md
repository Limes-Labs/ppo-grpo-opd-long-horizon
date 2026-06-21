# Deep Toy Matrix

This report aggregates the toy credit-assignment comparison across
20 seeds and 18 fixed cases.
Positive delta means the critic-style TD estimator has higher oracle
advantage correlation than the group-relative estimator.
For group-size rows, critic replay is fixed at 840 training trajectories
while only evaluation sibling group size changes.

## Summary

- Critic wins by mean correlation: 17
- Group wins by mean correlation: 1
- Clear critic-favorable cases by 95% CI: 16
- Near-tie cases by 95% CI: 1
- Clear group-favorable cases by 95% CI: 1
- Mean critic-minus-group correlation: 0.426

| Case | Axis | Mean winner | CI read | Group r | Critic r | Delta r | 95% CI | Critic hit | Wait frac | Success |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| horizon_04_baseline | horizon | critic | critic_clear | 0.495 | 0.977 | 0.482 | +/- 0.011 | 1.00 | 0.30 | 0.43 |
| horizon_08_baseline | horizon | critic | critic_clear | 0.390 | 0.925 | 0.535 | +/- 0.015 | 1.00 | 0.37 | 0.55 |
| horizon_12_baseline | horizon | critic | critic_clear | 0.335 | 0.866 | 0.531 | +/- 0.012 | 0.99 | 0.41 | 0.63 |
| horizon_16_long_wait | horizon | critic | critic_clear | 0.293 | 0.865 | 0.571 | +/- 0.013 | 1.00 | 0.54 | 0.68 |
| group_size_02_long_wait | group_size | critic | critic_clear | 0.213 | 0.902 | 0.689 | +/- 0.014 | 0.99 | 0.51 | 0.66 |
| group_size_04_long_wait | group_size | critic | critic_clear | 0.301 | 0.889 | 0.588 | +/- 0.016 | 1.00 | 0.50 | 0.65 |
| group_size_08_long_wait | group_size | critic | critic_clear | 0.346 | 0.894 | 0.549 | +/- 0.011 | 1.00 | 0.50 | 0.64 |
| group_size_12_long_wait | group_size | critic | critic_clear | 0.358 | 0.892 | 0.534 | +/- 0.009 | 1.00 | 0.50 | 0.65 |
| critic_budget_002_full | critic_budget | critic | near_tie | 0.352 | 0.367 | 0.015 | +/- 0.027 | 0.48 | 0.39 | 0.60 |
| critic_budget_008_full | critic_budget | critic | critic_clear | 0.352 | 0.459 | 0.107 | +/- 0.027 | 0.88 | 0.39 | 0.60 |
| critic_budget_032_full | critic_budget | critic | critic_clear | 0.352 | 0.735 | 0.383 | +/- 0.021 | 0.98 | 0.39 | 0.60 |
| critic_budget_128_full | critic_budget | critic | critic_clear | 0.352 | 0.906 | 0.554 | +/- 0.017 | 1.00 | 0.39 | 0.60 |
| observability_full | observability | critic | critic_clear | 0.352 | 0.864 | 0.513 | +/- 0.020 | 0.99 | 0.39 | 0.60 |
| observability_coarse | observability | critic | critic_clear | 0.356 | 0.738 | 0.382 | +/- 0.015 | 1.00 | 0.45 | 0.61 |
| observability_blind | observability | critic | critic_clear | 0.353 | 0.482 | 0.129 | +/- 0.017 | 1.00 | 0.44 | 0.61 |
| blind_undercovered_counterexample | counterexample | group | group_clear | 0.510 | 0.455 | -0.055 | +/- 0.015 | 0.69 | 0.33 | 0.43 |
| sparse_hard_group04 | sparse_reward | critic | critic_clear | 0.283 | 0.846 | 0.563 | +/- 0.014 | 0.99 | 0.42 | 0.47 |
| sparse_hard_group08 | sparse_reward | critic | critic_clear | 0.310 | 0.910 | 0.600 | +/- 0.008 | 1.00 | 0.42 | 0.47 |

## Interpretation

- The critic estimator is strongest when state coverage is high and traces
  contain mixed token roles.
- Group-relative estimation remains competitive when the critic is blind,
  poorly covered, or when terminal group outcomes carry most of the signal.
- These experiments measure estimator fidelity, not closed-loop policy
  improvement under PPO/GRPO clipping, KL control, or optimizer effects.
