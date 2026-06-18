# Length Imbalance Audit

This table audits whether simple length correction rescues a
trajectory-level group scalar as rollouts become longer and more
imbalanced. `Per-token r` is group-normalized total return divided by
trajectory length; it is still broadcast to every token.

| Max steps | Len range | Wait frac | Group r | Per-token r | Critic r | Critic - group | Critic - per-token |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1.775 | 0.336 | 0.488 | 0.500 | 0.981 | 0.492 | 0.481 |
| 8 | 4.851 | 0.431 | 0.389 | 0.389 | 0.940 | 0.551 | 0.551 |
| 12 | 7.725 | 0.495 | 0.330 | 0.328 | 0.890 | 0.561 | 0.562 |
| 16 | 10.696 | 0.539 | 0.293 | 0.289 | 0.849 | 0.556 | 0.560 |
| 20 | 13.439 | 0.573 | 0.266 | 0.260 | 0.811 | 0.545 | 0.551 |

Summary:
- Critic wins vs group total: 5 / 5
- Critic wins vs group per-token: 5 / 5
- Delta growth, shortest to longest horizon: 0.052
- Length-range growth, shortest to longest horizon: 11.664
