# Tiny Neural Generalization Audit

The neural value critic is trained on thresholds 1 and 3, then evaluated
on held-out threshold 2. Exact tabular state lookup cannot help on the
evaluation threshold; the value model must generalize from features.

| Estimator | Pearson r | Cal. MSE | Sign | Wait leak | Within var |
| --- | ---: | ---: | ---: | ---: | ---: |
| Group relative | 0.364 | 0.0318 | 0.674 | 0.917 | 0.000 |
| Neural critic TD | 0.842 | 0.0104 | 0.875 | 0.418 | 0.038 |

Summary:
- Held-out exact-state fraction: 1.000
- Neural minus group Pearson r: 0.479
- Train value MSE: 0.1789
