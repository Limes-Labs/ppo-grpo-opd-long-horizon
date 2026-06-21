# Held-out estimator-selection regret

The selection rule fits a critic-cost penalty on half the phase-grid
cells and evaluates regret on held-out cells. Within each cell,
alternating seeds split audit MSE estimates from held-out regret.

- Train cells: 24
- Held-out cells: 24
- Fitted lambda cost: 0.00050

| Policy | Held-out regret | Max regret | Accuracy | Critic rate |
| --- | ---: | ---: | ---: | ---: |
| audit_mse_cost | 0.00031 | 0.00745 | 0.95833 | 0.29167 |
| always_group | 0.00566 | 0.07478 | 0.75000 | 0.00000 |
| always_critic | 0.00234 | 0.01666 | 0.25000 | 1.00000 |
