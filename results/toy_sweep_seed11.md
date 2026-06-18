# Toy Scenario Sweep

This generated report summarizes a deterministic CPU-only sweep over toy
credit-assignment regimes. It measures estimator quality against the
known oracle advantage from the toy dynamics; it is not a closed-loop
PPO or GRPO training benchmark.

Seed: `11`

| Case | Winner | Group r | Critic r | Group MSE | Critic MSE | Zero-var groups | Critic state hit | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| short_dense_full_critic | critic | 0.442 | 0.974 | 0.04193 | 0.00271 | 0.00 | 1.00 | Shorter, denser traces reduce the cost of response-level scalar credit. |
| baseline_full_critic | critic | 0.353 | 0.898 | 0.03329 | 0.00737 | 0.00 | 1.00 | Baseline mixed horizons with a fully observed tabular value model. |
| long_wait_full_critic | critic | 0.286 | 0.908 | 0.02060 | 0.00392 | 0.00 | 1.00 | Long wait-heavy traces test whether response-level rewards praise no-op tokens. |
| sparse_hard_full_critic | critic | 0.310 | 0.916 | 0.02240 | 0.00398 | 0.00 | 1.00 | Sparse success tests whether either estimator has enough signal. |
| coarse_critic_partial_state | critic | 0.356 | 0.736 | 0.02268 | 0.01191 | 0.00 | 1.00 | A partially observed critic checks whether value information remains useful. |
| blind_critic_counterexample | group | 0.525 | 0.503 | 0.04634 | 0.04777 | 0.00 | 0.68 | A blind, undercovered critic is a counterexample where terminal group outcomes win. |

## Readout

- Critic wins in this toy mean the value estimator had enough relevant state
  information to recover temporal structure.
- Group wins mean terminal group outcome information was more useful than
  the critic's state abstraction in that regime.
- Zero-variance group rates diagnose when group normalization has no
  within-prompt reward contrast.
- Critic state hit rate diagnoses whether the learned value table is using
  exact state estimates or fallbacks.

## Caveats

The toy has a known finite-state oracle and a tabular critic, so it is much
cleaner than neural long-horizon RL. The results support mechanism-level
hypotheses only. Stronger GRPO variants, process rewards, KL/clipping
details, optimizer effects, and closed-loop policy learning remain future
work.
