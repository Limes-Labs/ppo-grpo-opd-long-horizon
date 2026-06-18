# Token-Cost Sensitivity

This table checks whether critic TD still improves oracle-credit
alignment when the explicit per-token cost is removed.

| Scenario | Cost | Group r | Critic r | Delta | 95% CI | Group wait | Critic wait |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.00 | 0.336 | 0.882 | 0.546 | 0.010 | 0.930 | 0.465 |
| baseline | 0.01 | 0.332 | 0.882 | 0.550 | 0.010 | 0.937 | 0.465 |
| baseline | 0.02 | 0.335 | 0.882 | 0.546 | 0.010 | 0.928 | 0.465 |
| baseline | 0.05 | 0.339 | 0.882 | 0.543 | 0.010 | 0.902 | 0.465 |
| long_wait | 0.00 | 0.335 | 0.888 | 0.552 | 0.015 | 0.871 | 0.371 |
| long_wait | 0.01 | 0.327 | 0.888 | 0.561 | 0.016 | 0.911 | 0.371 |
| long_wait | 0.02 | 0.330 | 0.888 | 0.557 | 0.016 | 0.898 | 0.371 |
| long_wait | 0.05 | 0.334 | 0.888 | 0.554 | 0.016 | 0.868 | 0.371 |

Summary:
- Clear positive rows: 8 / 8
- Minimum delta minus CI95: 0.532
- Long-wait zero-cost delta: 0.552
