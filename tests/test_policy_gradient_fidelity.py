import math
import unittest

from experiments.policy_gradient_fidelity import (
    ACTIONS,
    MDPConfig,
    TabularSoftmaxPolicy,
    exact_advantage,
    exact_policy_gradient,
    finite_difference_policy_gradient,
    run_policy_gradient_audit,
    vimpo_signal,
)


class PolicyGradientFidelityTests(unittest.TestCase):
    def test_true_null_action_has_zero_exact_advantage(self) -> None:
        config = MDPConfig(threshold=2, horizon=5)
        policy = TabularSoftmaxPolicy(config)

        self.assertIn("null", ACTIONS)
        for score in range(-2, 4):
            for remaining in range(1, config.horizon + 1):
                self.assertAlmostEqual(
                    exact_advantage(policy, score, remaining, "null"),
                    0.0,
                    places=12,
                )

    def test_vimpo_actor_signal_is_zero_at_reference_initialization(self) -> None:
        config = MDPConfig(threshold=2, horizon=5)
        policy = TabularSoftmaxPolicy(config)
        reference = policy.copy()

        exact = exact_policy_gradient(policy)
        self.assertGreater(exact.norm, 1e-3)

        max_abs_signal = 0.0
        for score in range(-2, 4):
            for remaining in range(1, config.horizon + 1):
                for action in ACTIONS:
                    max_abs_signal = max(
                        max_abs_signal,
                        abs(vimpo_signal(policy, reference, score, remaining, action)),
                    )
        self.assertAlmostEqual(max_abs_signal, 0.0, places=12)

    def test_analytic_gradient_matches_finite_difference_check(self) -> None:
        config = MDPConfig(threshold=2, horizon=4)
        policy = TabularSoftmaxPolicy(config)

        exact = exact_policy_gradient(policy)
        finite_difference = finite_difference_policy_gradient(policy)

        self.assertLess(finite_difference.minus(exact).norm / exact.norm, 1e-8)

    def test_audit_reports_gradient_bias_variance_and_matched_kl(self) -> None:
        result = run_policy_gradient_audit(
            seed=13,
            batches=12,
            groups_per_batch=6,
            group_size=4,
            replications=3,
            config=MDPConfig(threshold=2, horizon=4),
        )

        self.assertEqual(result["config"]["actions"], list(ACTIONS))
        self.assertGreater(result["exact_gradient"]["norm"], 1e-3)
        self.assertGreater(result["exact_gradient"]["base_return"], 0.0)
        self.assertLess(result["exact_gradient"]["finite_difference_relative_error"], 1e-8)
        self.assertEqual(result["config"]["replications"], 3)

        metrics = {entry["method"]: entry["metrics"] for entry in result["estimators"]}
        for required in [
            "reinforce_return",
            "sibling_loo_return",
            "prefix_value_baseline",
            "brpo_combined_baseline",
            "learned_value_td",
            "oracle_value_td",
        ]:
            self.assertIn(required, metrics)
            for key in [
                "relative_mean_error_norm",
                "relative_mean_error_norm_ci95",
                "variance_trace",
                "variance_trace_ci95",
                "gradient_cosine",
                "matched_kl_improvement",
                "batch_matched_kl_improvement_mean",
                "batch_matched_kl_improvement_p05",
                "batch_negative_update_probability",
                "null_abs_credit_mean",
                "advantage_correlation",
            ]:
                self.assertTrue(math.isfinite(metrics[required][key]), (required, key))
            self.assertGreaterEqual(
                metrics[required]["batch_negative_update_probability"],
                0.0,
            )
            self.assertLessEqual(
                metrics[required]["batch_negative_update_probability"],
                1.0,
            )

        self.assertLess(
            metrics["prefix_value_baseline"]["variance_trace"],
            metrics["reinforce_return"]["variance_trace"],
        )
        self.assertLess(
            metrics["brpo_combined_baseline"]["variance_trace"],
            metrics["reinforce_return"]["variance_trace"],
        )
        self.assertGreater(metrics["learned_value_td"]["critic_state_hit_rate"], 0.0)
        self.assertLess(
            metrics["oracle_value_td"]["variance_trace"],
            metrics["reinforce_return"]["variance_trace"],
        )
        self.assertGreater(
            metrics["sibling_loo_return"]["matched_kl_improvement"],
            0.0,
        )
        self.assertGreater(
            metrics["oracle_value_td"]["matched_kl_improvement"],
            0.0,
        )

        policy_implied = {
            entry["method"]: entry for entry in result["policy_implied_signals"]
        }
        self.assertIn("vimpo_actor_equal_ref", policy_implied)
        self.assertIn("vimpo_actor_fixed_ref_far", policy_implied)
        self.assertAlmostEqual(
            policy_implied["vimpo_actor_equal_ref"]["metrics"]["mean_gradient_norm"],
            0.0,
            places=12,
        )
        self.assertEqual(
            policy_implied["vimpo_actor_equal_ref"]["metrics"]["gradient_cosine"],
            0.0,
        )
        self.assertGreater(
            policy_implied["vimpo_actor_fixed_ref_far"]["reference_kl"],
            policy_implied["vimpo_actor_fixed_ref_near"]["reference_kl"],
        )

        position = result["position_diagnostics"]["overall"]
        for required in ["group_mean", "prefix_budget", "brpo_combined", "critic_value"]:
            self.assertIn(required, position)
            self.assertGreater(position[required]["count"], 0)
        self.assertLess(
            position["brpo_combined"]["score_weighted_residual_second_moment"],
            position["group_mean"]["score_weighted_residual_second_moment"],
        )
        self.assertLess(
            position["critic_value"]["residual_variance_ratio"],
            position["group_mean"]["residual_variance_ratio"],
        )


if __name__ == "__main__":
    unittest.main()
