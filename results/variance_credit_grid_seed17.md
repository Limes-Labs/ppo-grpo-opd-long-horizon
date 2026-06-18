# Variance Reduction vs Credit Assignment Grid

This CPU-only toy run separates variance reduction from credit
assignment. It compares trajectory-level baselines with step-level
estimators against the known oracle advantage. It is estimator-fidelity
evidence, not a closed-loop training result.

## Configuration

- `seed`: 17
- `scenario_name`: long_wait
- `scenario_description`: Long successful traces often contain uninformative wait tokens.
- `critic_observation`: full
- `train_groups`: 160
- `eval_groups`: 48
- `group_size`: 6
- `max_steps`: 14
- `branches_per_state`: 32
- `token_cost`: 0.02

## Results

| Estimator | Variance reduction | Credit | r | MSE | Sign | Wait leak | Within-traj var | Second moment |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| REINFORCE return | none | trajectory | 0.357 | 0.02082 | 0.643 | 1.157 | 0.000 | 0.474 |
| Global baseline | global/running | trajectory | 0.357 | 0.02082 | 0.643 | 0.875 | 0.000 | 0.189 |
| Sibling group norm | sibling group | trajectory | 0.319 | 0.02144 | 0.631 | 0.855 | 0.000 | 0.898 |
| Leave-one-out | sibling group | trajectory | 0.332 | 0.02123 | 0.631 | 0.799 | 0.000 | 0.215 |
| Dense progress reward | external dense signal | step | 0.674 | 0.01303 | 0.469 | 0.000 | 0.416 | 0.480 |
| Anchor action contrast | anchor state group | step | 0.715 | 0.01166 | 0.592 | 0.439 | 0.042 | 0.045 |
| Learned critic TD | learned critic V(s) | step | 0.870 | 0.00581 | 0.723 | 0.356 | 0.031 | 0.031 |
| Sampled MC value | sampled value | step | 0.849 | 0.00665 | 0.665 | 0.563 | 0.034 | 0.033 |
| Oracle advantage ceiling | oracle value | step | 1.000 | 0.00000 | 1.000 | 0.250 | 0.025 | 0.024 |

## Reading

- Global baselines can reduce the second moment of a trajectory-level
  policy-gradient signal without creating step-level credit.
- Sibling groups can be memory-light, but in this long-wait setting their
  scalar advantages still have zero within-trajectory variation.
- Anchor-action contrast is a critic-free structural batch estimator
  over repeated exact toy states; unsupported anchors fall back to zero.
- Step-level estimators create intra-trajectory variation and reduce
  wait-token leakage when they have useful state or process signal.
- The oracle row is a ceiling, included only to calibrate estimator fidelity.

## Summary

- Best non-oracle estimator: `critic_td`.
- Best step-level estimator: `critic_td`.
- Best trajectory-level estimator: `reinforce_return`.
- Step best minus trajectory best correlation: 0.513.
- Global baseline minus REINFORCE second moment: -0.286.
