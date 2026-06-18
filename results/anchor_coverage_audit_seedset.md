# Anchor Coverage Audit

This audit sweeps evaluation batch size to change exact repeated
state-action coverage for anchor-action contrast. It tests when this
critic-free structural batch estimator beats trajectory-level sibling
group normalization, and whether it closes the gap to critic TD.

| Eval groups | Support | Sibling r | Anchor r | Critic r | A-G | C-A | Anchor wait |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.413 | 0.361 | 0.134 | 0.805 | -0.228 | 0.672 | 0.775 |
| 4 | 0.510 | 0.341 | 0.253 | 0.836 | -0.088 | 0.583 | 0.839 |
| 8 | 0.675 | 0.329 | 0.389 | 0.852 | 0.060 | 0.463 | 0.556 |
| 16 | 0.813 | 0.326 | 0.544 | 0.845 | 0.219 | 0.301 | 0.510 |
| 32 | 0.915 | 0.321 | 0.654 | 0.861 | 0.334 | 0.206 | 0.476 |
| 48 | 0.945 | 0.318 | 0.710 | 0.867 | 0.392 | 0.156 | 0.453 |
| 64 | 0.962 | 0.315 | 0.751 | 0.865 | 0.437 | 0.113 | 0.423 |

Summary:
- First eval-groups value where anchor beats sibling: 8.
- First eval-groups value where support >= 0.80: 16.
- Rows where critic remains above anchor: 7.
- Maximum anchor r: 0.751.

Reading:
- Anchor contrast is coverage-sensitive. At low exact-anchor coverage,
  it can be worse than sibling group normalization.
- With enough repeated anchors, it becomes a strong critic-free
  step-level baseline, but critic TD remains higher in this sweep.
