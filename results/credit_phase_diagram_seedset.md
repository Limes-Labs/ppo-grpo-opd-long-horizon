# Broadcast ceiling phase diagram

This diagnostic estimates two boundary quantities in the finite toy MDP:
within-trajectory credit heterogeneity and held-out critic reliability.
The group estimator is trajectory-constant, so its absolute correlation
with exact behavior-policy advantage should not exceed the broadcast
ceiling implied by the heterogeneity index.

## Summary

- Seeds: 5
- Cells: 32
- Clear critic cells: 15
- Clear group cells: 2
- Near ties: 15

| Cell | H_credit | Ceiling | Group r | Critic r | Delta r | CI | Read | Mechanism |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| low_full_low_contrast_matched | 0.671 | 0.573 | 0.457 | 0.466 | 0.009 | +/- 0.081 | near_tie | process_structural_or_hybrid |
| low_full_low_contrast_stale | 0.671 | 0.573 | 0.457 | 0.485 | 0.028 | +/- 0.058 | near_tie | process_structural_or_hybrid |
| low_full_low_sparse_matched | 0.743 | 0.506 | 0.364 | 0.359 | -0.005 | +/- 0.052 | near_tie | process_structural_or_hybrid |
| low_full_low_sparse_stale | 0.743 | 0.506 | 0.364 | 0.346 | -0.018 | +/- 0.058 | near_tie | process_structural_or_hybrid |
| low_full_high_contrast_matched | 0.671 | 0.573 | 0.457 | 0.968 | 0.511 | +/- 0.034 | critic_clear | critic_or_sampled_value |
| low_full_high_contrast_stale | 0.671 | 0.573 | 0.457 | 0.957 | 0.500 | +/- 0.024 | critic_clear | critic_or_sampled_value |
| low_full_high_sparse_matched | 0.743 | 0.506 | 0.364 | 0.962 | 0.598 | +/- 0.011 | critic_clear | critic_or_sampled_value |
| low_full_high_sparse_stale | 0.743 | 0.506 | 0.364 | 0.943 | 0.579 | +/- 0.025 | critic_clear | critic_or_sampled_value |
| low_blind_low_contrast_matched | 0.671 | 0.573 | 0.457 | 0.477 | 0.019 | +/- 0.023 | near_tie | process_structural_or_hybrid |
| low_blind_low_contrast_stale | 0.671 | 0.573 | 0.457 | 0.516 | 0.059 | +/- 0.060 | near_tie | process_structural_or_hybrid |
| low_blind_low_sparse_matched | 0.743 | 0.506 | 0.364 | 0.309 | -0.055 | +/- 0.023 | group_clear | process_structural_or_hybrid |
| low_blind_low_sparse_stale | 0.743 | 0.506 | 0.364 | 0.288 | -0.076 | +/- 0.029 | group_clear | process_structural_or_hybrid |
| low_blind_high_contrast_matched | 0.671 | 0.573 | 0.457 | 0.577 | 0.120 | +/- 0.036 | critic_clear | process_structural_or_hybrid |
| low_blind_high_contrast_stale | 0.671 | 0.573 | 0.457 | 0.561 | 0.104 | +/- 0.041 | critic_clear | process_structural_or_hybrid |
| low_blind_high_sparse_matched | 0.743 | 0.506 | 0.364 | 0.406 | 0.043 | +/- 0.036 | near_tie | process_structural_or_hybrid |
| low_blind_high_sparse_stale | 0.743 | 0.506 | 0.364 | 0.396 | 0.032 | +/- 0.041 | near_tie | process_structural_or_hybrid |
| high_full_low_contrast_matched | 0.845 | 0.393 | 0.277 | 0.336 | 0.059 | +/- 0.024 | critic_clear | process_structural_or_hybrid |
| high_full_low_contrast_stale | 0.845 | 0.393 | 0.277 | 0.295 | 0.019 | +/- 0.074 | near_tie | process_structural_or_hybrid |
| high_full_low_sparse_matched | 0.879 | 0.347 | 0.283 | 0.288 | 0.006 | +/- 0.045 | near_tie | process_structural_or_hybrid |
| high_full_low_sparse_stale | 0.879 | 0.347 | 0.283 | 0.245 | -0.037 | +/- 0.026 | near_tie | process_structural_or_hybrid |
| high_full_high_contrast_matched | 0.845 | 0.393 | 0.277 | 0.826 | 0.549 | +/- 0.030 | critic_clear | critic_or_sampled_value |
| high_full_high_contrast_stale | 0.845 | 0.393 | 0.277 | 0.726 | 0.450 | +/- 0.079 | critic_clear | critic_or_sampled_value |
| high_full_high_sparse_matched | 0.879 | 0.347 | 0.283 | 0.802 | 0.519 | +/- 0.041 | critic_clear | critic_or_sampled_value |
| high_full_high_sparse_stale | 0.879 | 0.347 | 0.283 | 0.740 | 0.457 | +/- 0.047 | critic_clear | critic_or_sampled_value |
| high_blind_low_contrast_matched | 0.845 | 0.393 | 0.277 | 0.339 | 0.062 | +/- 0.023 | critic_clear | process_structural_or_hybrid |
| high_blind_low_contrast_stale | 0.845 | 0.393 | 0.277 | 0.338 | 0.061 | +/- 0.020 | critic_clear | process_structural_or_hybrid |
| high_blind_low_sparse_matched | 0.879 | 0.347 | 0.283 | 0.277 | -0.006 | +/- 0.057 | near_tie | process_structural_or_hybrid |
| high_blind_low_sparse_stale | 0.879 | 0.347 | 0.283 | 0.235 | -0.048 | +/- 0.028 | near_tie | process_structural_or_hybrid |
| high_blind_high_contrast_matched | 0.845 | 0.393 | 0.277 | 0.408 | 0.132 | +/- 0.031 | critic_clear | process_structural_or_hybrid |
| high_blind_high_contrast_stale | 0.845 | 0.393 | 0.277 | 0.399 | 0.122 | +/- 0.030 | critic_clear | process_structural_or_hybrid |
| high_blind_high_sparse_matched | 0.879 | 0.347 | 0.283 | 0.321 | 0.038 | +/- 0.037 | near_tie | process_structural_or_hybrid |
| high_blind_high_sparse_stale | 0.879 | 0.347 | 0.283 | 0.303 | 0.021 | +/- 0.044 | near_tie | process_structural_or_hybrid |

## Reading

- High credit heterogeneity lowers the best possible token-level fidelity
  of any trajectory-constant estimator.
- A critic crosses that ceiling only when its held-out TD signal is
  reliable under the current observation and coverage regime.
- When heterogeneity is high and critic reliability is low, the diagnostic
  points to process rewards, structural anchors, or adaptive hybrids.
