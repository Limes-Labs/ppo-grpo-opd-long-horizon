# Broadcast ceiling phase diagram

This diagnostic estimates two boundary quantities in the finite toy MDP:
within-trajectory credit heterogeneity and held-out critic reliability.
The group estimator is trajectory-constant, so its absolute correlation
with exact behavior-policy advantage should not exceed the broadcast
ceiling implied by the heterogeneity index.

## Summary

- Seeds: 5
- Cells: 48
- H_credit range: 0.000 to 0.880
- Clear critic cells: 11
- Clear group cells: 3
- Near ties: 34

| Cell | Target H | H_credit | Ceiling | Group r | Critic r | Delta r | CI | Read | Mechanism |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| h005_non_privileged_low_contrast_matched | 0.050 | 0.000 | 1.000 | 0.901 | 1.000 | 0.099 | +/- 0.015 | critic_clear | either_by_cost |
| h005_non_privileged_low_sparse_matched | 0.050 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | +/- 0.000 | near_tie | group_or_global |
| h005_non_privileged_high_contrast_matched | 0.050 | 0.000 | 1.000 | 0.901 | 1.000 | 0.099 | +/- 0.015 | critic_clear | either_by_cost |
| h005_non_privileged_high_sparse_matched | 0.050 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | +/- 0.000 | near_tie | group_or_global |
| h005_blind_low_contrast_matched | 0.050 | 0.000 | 1.000 | 0.901 | 1.000 | 0.099 | +/- 0.015 | critic_clear | either_by_cost |
| h005_blind_low_sparse_matched | 0.050 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | +/- 0.000 | near_tie | group_or_global |
| h005_blind_high_contrast_matched | 0.050 | 0.000 | 1.000 | 0.901 | 1.000 | 0.099 | +/- 0.015 | critic_clear | either_by_cost |
| h005_blind_high_sparse_matched | 0.050 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | +/- 0.000 | near_tie | group_or_global |
| h015_non_privileged_low_contrast_matched | 0.150 | 0.246 | 0.868 | 0.665 | 0.765 | 0.100 | +/- 0.148 | near_tie | either_by_cost |
| h015_non_privileged_low_sparse_matched | 0.150 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | +/- 0.000 | near_tie | group_or_global |
| h015_non_privileged_high_contrast_matched | 0.150 | 0.246 | 0.868 | 0.665 | 0.999 | 0.334 | +/- 0.043 | critic_clear | either_by_cost |
| h015_non_privileged_high_sparse_matched | 0.150 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | +/- 0.000 | near_tie | group_or_global |
| h015_blind_low_contrast_matched | 0.150 | 0.246 | 0.868 | 0.665 | 0.689 | 0.024 | +/- 0.084 | near_tie | either_by_cost |
| h015_blind_low_sparse_matched | 0.150 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | +/- 0.000 | near_tie | group_or_global |
| h015_blind_high_contrast_matched | 0.150 | 0.246 | 0.868 | 0.665 | 0.758 | 0.093 | +/- 0.085 | near_tie | either_by_cost |
| h015_blind_high_sparse_matched | 0.150 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | +/- 0.000 | near_tie | group_or_global |
| h030_non_privileged_low_contrast_matched | 0.300 | 0.251 | 0.865 | 0.730 | 0.788 | 0.058 | +/- 0.078 | near_tie | either_by_cost |
| h030_non_privileged_low_sparse_matched | 0.300 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | +/- 0.000 | near_tie | group_or_global |
| h030_non_privileged_high_contrast_matched | 0.300 | 0.251 | 0.865 | 0.730 | 0.999 | 0.269 | +/- 0.037 | critic_clear | either_by_cost |
| h030_non_privileged_high_sparse_matched | 0.300 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | +/- 0.000 | near_tie | group_or_global |
| h030_blind_low_contrast_matched | 0.300 | 0.251 | 0.865 | 0.730 | 0.738 | 0.008 | +/- 0.078 | near_tie | either_by_cost |
| h030_blind_low_sparse_matched | 0.300 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | +/- 0.000 | near_tie | group_or_global |
| h030_blind_high_contrast_matched | 0.300 | 0.251 | 0.865 | 0.730 | 0.766 | 0.035 | +/- 0.051 | near_tie | either_by_cost |
| h030_blind_high_sparse_matched | 0.300 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | +/- 0.000 | near_tie | group_or_global |
| h050_non_privileged_low_contrast_matched | 0.500 | 0.497 | 0.709 | 0.624 | 0.548 | -0.077 | +/- 0.095 | near_tie | process_structural_or_hybrid |
| h050_non_privileged_low_sparse_matched | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | +/- 0.000 | near_tie | group_or_global |
| h050_non_privileged_high_contrast_matched | 0.500 | 0.497 | 0.709 | 0.624 | 0.997 | 0.373 | +/- 0.014 | critic_clear | critic_or_sampled_value |
| h050_non_privileged_high_sparse_matched | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | +/- 0.000 | near_tie | group_or_global |
| h050_blind_low_contrast_matched | 0.500 | 0.497 | 0.709 | 0.624 | 0.521 | -0.103 | +/- 0.061 | group_clear | process_structural_or_hybrid |
| h050_blind_low_sparse_matched | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | +/- 0.000 | near_tie | group_or_global |
| h050_blind_high_contrast_matched | 0.500 | 0.497 | 0.709 | 0.624 | 0.595 | -0.029 | +/- 0.030 | near_tie | process_structural_or_hybrid |
| h050_blind_high_sparse_matched | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | +/- 0.000 | near_tie | group_or_global |
| h070_non_privileged_low_contrast_matched | 0.700 | 0.704 | 0.543 | 0.410 | 0.303 | -0.107 | +/- 0.101 | near_tie | process_structural_or_hybrid |
| h070_non_privileged_low_sparse_matched | 0.700 | 0.830 | 0.410 | 0.273 | 0.362 | 0.089 | +/- 0.095 | near_tie | process_structural_or_hybrid |
| h070_non_privileged_high_contrast_matched | 0.700 | 0.704 | 0.543 | 0.410 | 0.924 | 0.515 | +/- 0.054 | critic_clear | critic_or_sampled_value |
| h070_non_privileged_high_sparse_matched | 0.700 | 0.830 | 0.410 | 0.273 | 0.864 | 0.590 | +/- 0.051 | critic_clear | critic_or_sampled_value |
| h070_blind_low_contrast_matched | 0.700 | 0.704 | 0.543 | 0.410 | 0.285 | -0.125 | +/- 0.064 | group_clear | process_structural_or_hybrid |
| h070_blind_low_sparse_matched | 0.700 | 0.830 | 0.410 | 0.273 | 0.362 | 0.089 | +/- 0.095 | near_tie | process_structural_or_hybrid |
| h070_blind_high_contrast_matched | 0.700 | 0.704 | 0.543 | 0.410 | 0.411 | 0.002 | +/- 0.053 | near_tie | process_structural_or_hybrid |
| h070_blind_high_sparse_matched | 0.700 | 0.830 | 0.410 | 0.273 | 0.384 | 0.111 | +/- 0.083 | near_tie | process_structural_or_hybrid |
| h090_non_privileged_low_contrast_matched | 0.900 | 0.843 | 0.396 | 0.301 | 0.249 | -0.052 | +/- 0.070 | near_tie | process_structural_or_hybrid |
| h090_non_privileged_low_sparse_matched | 0.900 | 0.880 | 0.346 | 0.273 | 0.241 | -0.032 | +/- 0.076 | near_tie | process_structural_or_hybrid |
| h090_non_privileged_high_contrast_matched | 0.900 | 0.843 | 0.396 | 0.301 | 0.859 | 0.558 | +/- 0.033 | critic_clear | critic_or_sampled_value |
| h090_non_privileged_high_sparse_matched | 0.900 | 0.880 | 0.346 | 0.273 | 0.848 | 0.576 | +/- 0.047 | critic_clear | critic_or_sampled_value |
| h090_blind_low_contrast_matched | 0.900 | 0.843 | 0.396 | 0.301 | 0.189 | -0.112 | +/- 0.049 | group_clear | process_structural_or_hybrid |
| h090_blind_low_sparse_matched | 0.900 | 0.880 | 0.346 | 0.273 | 0.215 | -0.057 | +/- 0.113 | near_tie | process_structural_or_hybrid |
| h090_blind_high_contrast_matched | 0.900 | 0.843 | 0.396 | 0.301 | 0.249 | -0.052 | +/- 0.036 | near_tie | process_structural_or_hybrid |
| h090_blind_high_sparse_matched | 0.900 | 0.880 | 0.346 | 0.273 | 0.312 | 0.039 | +/- 0.091 | near_tie | process_structural_or_hybrid |

## Reading

- High credit heterogeneity lowers the best possible token-level fidelity
  of any trajectory-constant estimator.
- A critic crosses that ceiling only when its held-out TD signal is
  reliable under the current observation and coverage regime.
- When heterogeneity is high and critic reliability is low, the diagnostic
  points to process rewards, structural anchors, or adaptive hybrids.
