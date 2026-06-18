import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.toy_credit_assignment import StepRecord
from experiments.variance_credit_grid import build_anchor_action_contrast, run_grid


def make_step(
    trajectory_id: int,
    *,
    action: str,
    start_score: int,
    remaining_before: int,
    return_to_go: float,
) -> StepRecord:
    return StepRecord(
        trajectory_id=trajectory_id,
        prompt_id=0,
        threshold=3,
        action=action,
        start_score=start_score,
        next_score=start_score,
        remaining_before=remaining_before,
        remaining_after=remaining_before - 1,
        step_reward=0.0,
        terminal_reward=return_to_go,
        total_return=return_to_go,
        return_to_go=return_to_go,
        oracle_advantage=0.0,
    )


class VarianceCreditGridTests(unittest.TestCase):
    def test_anchor_action_contrast_matches_leave_one_out_formula(self) -> None:
        steps = [
            make_step(
                1, action="help", start_score=0, remaining_before=3, return_to_go=1.0
            ),
            make_step(
                2, action="help", start_score=0, remaining_before=3, return_to_go=0.5
            ),
            make_step(
                3, action="harm", start_score=0, remaining_before=3, return_to_go=-0.5
            ),
            make_step(
                4, action="harm", start_score=0, remaining_before=3, return_to_go=-1.0
            ),
            make_step(
                5, action="wait", start_score=2, remaining_before=2, return_to_go=0.25
            ),
        ]

        estimates, counts = build_anchor_action_contrast(steps)
        expected = {
            (1, 3): 0.5 - ((0.5 - 0.5 - 1.0) / 3),
            (2, 3): 1.0 - ((1.0 - 0.5 - 1.0) / 3),
            (3, 3): -1.0 - ((1.0 + 0.5 - 1.0) / 3),
            (4, 3): -0.5 - ((1.0 + 0.5 - 0.5) / 3),
        }

        for key, value in expected.items():
            self.assertAlmostEqual(estimates[key], value)
        self.assertEqual(estimates[(5, 2)], 0.0)
        self.assertEqual(counts["unique_anchor_states"], 2.0)
        self.assertEqual(counts["unique_anchor_actions"], 3.0)
        self.assertAlmostEqual(counts["supported_step_fraction"], 4 / 5)

    def test_grid_separates_variance_reduction_from_step_credit(self) -> None:
        result = run_grid(
            seed=17,
            train_groups=80,
            eval_groups=24,
            group_size=6,
            max_steps=12,
            branches_per_state=16,
        )
        estimators = {entry["name"]: entry for entry in result["estimators"]}

        reinforce = estimators["reinforce_return"]["metrics"]
        global_baseline = estimators["global_baseline"]["metrics"]
        sibling = estimators["sibling_group_norm"]["metrics"]
        anchor = estimators["anchor_action_contrast"]["metrics"]
        critic = estimators["critic_td"]["metrics"]
        sampled = estimators["sampled_mc_td"]["metrics"]

        self.assertLess(
            global_baseline["estimate_second_moment"],
            reinforce["estimate_second_moment"],
        )
        for name in [
            "reinforce_return",
            "global_baseline",
            "sibling_group_norm",
            "leave_one_out",
        ]:
            self.assertAlmostEqual(
                estimators[name]["metrics"]["within_trajectory_variance"],
                0.0,
                places=12,
            )

        self.assertGreater(
            critic["pearson_correlation"],
            sibling["pearson_correlation"] + 0.40,
        )
        self.assertGreater(
            sampled["pearson_correlation"],
            sibling["pearson_correlation"] + 0.35,
        )
        self.assertGreater(
            anchor["pearson_correlation"],
            sibling["pearson_correlation"] + 0.25,
        )
        self.assertLess(
            anchor["pearson_correlation"],
            critic["pearson_correlation"],
        )
        self.assertGreater(anchor["within_trajectory_variance"], 0.0)
        self.assertLess(
            critic["wait_to_active_abs_ratio"],
            sibling["wait_to_active_abs_ratio"],
        )
        self.assertGreater(
            result["sample_counts"]["anchor_action"]["supported_step_fraction"],
            0.75,
        )

    def test_grid_is_deterministic(self) -> None:
        first = run_grid(
            seed=29,
            train_groups=60,
            eval_groups=16,
            group_size=5,
            max_steps=10,
            branches_per_state=12,
        )
        second = run_grid(
            seed=29,
            train_groups=60,
            eval_groups=16,
            group_size=5,
            max_steps=10,
            branches_per_state=12,
        )

        self.assertEqual(first["estimators"], second["estimators"])
        self.assertEqual(first["summary"], second["summary"])

    def test_cli_writes_json_and_markdown_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "grid.json"
            output_md = Path(tmpdir) / "grid.md"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.variance_credit_grid",
                    "--seed",
                    "17",
                    "--train-groups",
                    "60",
                    "--eval-groups",
                    "16",
                    "--group-size",
                    "5",
                    "--max-steps",
                    "10",
                    "--branches-per-state",
                    "12",
                    "--output-json",
                    str(output_json),
                    "--output-md",
                    str(output_md),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("best:", completed.stdout)
            payload = json.loads(output_json.read_text())
            names = {entry["name"] for entry in payload["estimators"]}
            self.assertIn("critic_td", names)
            self.assertIn("sampled_mc_td", names)
            self.assertIn("anchor_action_contrast", names)
            self.assertIn("sibling_group_norm", names)
            self.assertIn("Variance Reduction vs Credit Assignment", output_md.read_text())


if __name__ == "__main__":
    unittest.main()
