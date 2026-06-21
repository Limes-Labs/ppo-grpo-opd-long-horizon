import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.toy_credit_assignment import StepRecord, Trajectory
from experiments.credit_phase_diagram import credit_heterogeneity, run_phase_diagram


def make_step(trajectory_id: int, remaining: int, advantage: float) -> StepRecord:
    return StepRecord(
        trajectory_id=trajectory_id,
        prompt_id=0,
        threshold=1,
        action="help",
        start_score=0,
        next_score=1,
        remaining_before=remaining,
        remaining_after=max(0, remaining - 1),
        step_reward=0.0,
        terminal_reward=0.0,
        total_return=0.0,
        return_to_go=0.0,
        oracle_advantage=advantage,
    )


def make_trajectory(trajectory_id: int, advantages: list[float]) -> Trajectory:
    return Trajectory(
        trajectory_id=trajectory_id,
        prompt_id=0,
        threshold=1,
        length=len(advantages),
        final_score=0,
        terminal_reward=0.0,
        total_return=0.0,
        steps=[
            make_step(trajectory_id, len(advantages) - index, advantage)
            for index, advantage in enumerate(advantages)
        ],
    )


class CreditPhaseDiagramTests(unittest.TestCase):
    def test_credit_heterogeneity_matches_broadcast_ceiling_theorem(self) -> None:
        trajectories = [
            make_trajectory(1, [1.0, -1.0]),
            make_trajectory(2, [1.0, 1.0]),
        ]

        diagnostic = credit_heterogeneity(trajectories)

        self.assertAlmostEqual(diagnostic["total_variance"], 0.75)
        self.assertAlmostEqual(diagnostic["within_trajectory_variance"], 0.50)
        self.assertAlmostEqual(diagnostic["credit_heterogeneity"], 2 / 3)
        self.assertAlmostEqual(
            diagnostic["broadcast_ceiling_correlation"],
            math.sqrt(1 / 3),
        )

    def test_phase_diagram_reports_boundary_quantities(self) -> None:
        result = run_phase_diagram(
            seeds=[11, 29],
            heterogeneity_levels=["low", "high"],
            observability_levels=["non_privileged", "blind"],
            coverage_levels=["low", "high"],
            reward_levels=["contrast"],
            drift_levels=["matched"],
            eval_groups=12,
            group_size=4,
        )

        self.assertEqual(result["summary"]["cell_count"], 8)
        self.assertGreaterEqual(result["summary"]["critic_clear_cells"], 1)
        self.assertGreaterEqual(result["summary"]["group_clear_cells"], 1)

        rows = result["aggregate_rows"]
        high_rows = [row for row in rows if row["heterogeneity"] == "high"]
        low_rows = [row for row in rows if row["heterogeneity"] == "low"]
        self.assertGreater(
            sum(row["credit_heterogeneity"] for row in high_rows) / len(high_rows),
            sum(row["credit_heterogeneity"] for row in low_rows) / len(low_rows),
        )
        for row in rows:
            self.assertIn("actor_observation_schema", row["seed_results"][0])
            self.assertIn("critic_observation_schema", row["seed_results"][0])
            if row["observability"] == "non_privileged":
                self.assertFalse(row["critic_is_privileged"])
            self.assertGreaterEqual(row["credit_heterogeneity"], 0.0)
            self.assertLessEqual(row["credit_heterogeneity"], 1.0)
            self.assertGreaterEqual(row["broadcast_ceiling_correlation"], 0.0)
            self.assertLessEqual(row["broadcast_ceiling_correlation"], 1.0)
            self.assertLessEqual(
                abs(row["group_correlation"]),
                row["broadcast_ceiling_correlation"] + 1e-9,
            )
            self.assertIn(
                row["recommended_mechanism"],
                {
                    "group_or_global",
                    "either_by_cost",
                    "critic_or_sampled_value",
                    "process_structural_or_hybrid",
                },
            )

    def test_calibrated_low_heterogeneity_and_group_size_deconfounding(self) -> None:
        small_group = run_phase_diagram(
            seeds=[11],
            heterogeneity_levels=["h005", "h030"],
            observability_levels=["non_privileged"],
            coverage_levels=["high"],
            reward_levels=["contrast"],
            drift_levels=["matched"],
            eval_groups=12,
            group_size=3,
        )
        large_group = run_phase_diagram(
            seeds=[11],
            heterogeneity_levels=["h005", "h030"],
            observability_levels=["non_privileged"],
            coverage_levels=["high"],
            reward_levels=["contrast"],
            drift_levels=["matched"],
            eval_groups=12,
            group_size=6,
        )

        h005 = next(
            row for row in small_group["aggregate_rows"] if row["heterogeneity"] == "h005"
        )
        self.assertLess(h005["credit_heterogeneity"], 0.20)
        self.assertFalse(h005["critic_is_privileged"])

        small_counts = {
            row["heterogeneity"]: row["train_trajectories"]
            for row in small_group["aggregate_rows"]
        }
        large_counts = {
            row["heterogeneity"]: row["train_trajectories"]
            for row in large_group["aggregate_rows"]
        }
        self.assertEqual(small_counts, large_counts)
        self.assertEqual(set(small_counts.values()), {600})

    def test_cli_writes_phase_diagram_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "phase.json"
            output_md = Path(tmpdir) / "phase.md"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.credit_phase_diagram",
                    "--seeds",
                    "11,29",
                    "--heterogeneity-levels",
                    "low,high",
                    "--observability-levels",
                    "full,blind",
                    "--coverage-levels",
                    "low,high",
                    "--reward-levels",
                    "contrast",
                    "--drift-levels",
                    "matched",
                    "--eval-groups",
                    "12",
                    "--group-size",
                    "4",
                    "--output-json",
                    str(output_json),
                    "--output-md",
                    str(output_md),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("phase cells:", completed.stdout)
            payload = json.loads(output_json.read_text())
            self.assertEqual(payload["summary"]["cell_count"], 8)
            self.assertIn("Broadcast ceiling phase diagram", output_md.read_text())


if __name__ == "__main__":
    unittest.main()
