import unittest

from experiments.credit_phase_diagram import run_phase_diagram
from experiments.selection_regret import run_selection_regret


class SelectionRegretTests(unittest.TestCase):
    def test_selection_regret_beats_at_least_one_static_policy(self) -> None:
        phase = run_phase_diagram(
            seeds=[11, 29, 47],
            heterogeneity_levels=["low", "high"],
            observability_levels=["non_privileged", "blind"],
            coverage_levels=["low", "high"],
            reward_levels=["contrast", "sparse"],
            drift_levels=["matched"],
            eval_groups=12,
            group_size=4,
        )
        result = run_selection_regret(phase)

        heldout = result["heldout_metrics"]
        self.assertEqual(result["config"]["heldout_cell_count"], 8)
        heldout_rewards = {
            row["cell_name"]: (
                "sparse" if "_sparse_" in row["cell_name"] else "contrast"
            )
            for row in result["heldout_choices"]
        }
        self.assertEqual(set(heldout_rewards.values()), {"contrast", "sparse"})
        self.assertIn("reward-paired split", result["config"]["cell_split"])
        self.assertIn("audit_mse_cost", heldout)
        self.assertIn("always_group", heldout)
        self.assertIn("always_critic", heldout)
        self.assertLessEqual(
            heldout["audit_mse_cost"]["mean_regret"],
            max(
                heldout["always_group"]["mean_regret"],
                heldout["always_critic"]["mean_regret"],
            ),
        )
        self.assertGreaterEqual(heldout["audit_mse_cost"]["selection_accuracy"], 0.0)
        self.assertLessEqual(heldout["audit_mse_cost"]["selection_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
