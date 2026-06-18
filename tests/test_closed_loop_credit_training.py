import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.closed_loop_credit_training import TrainingConfig, run_closed_loop


class ClosedLoopCreditTrainingTests(unittest.TestCase):
    def test_coverage_gated_beats_group_in_small_closed_loop_probe(self) -> None:
        result = run_closed_loop(
            seeds=[11, 29, 47],
            config=TrainingConfig(
                train_iterations=35,
                groups_per_iteration=12,
                group_size=4,
                max_steps=10,
                eval_groups=40,
                eval_every=35,
            ),
        )
        summaries = {row["method"]: row for row in result["method_summaries"]}

        self.assertGreater(
            summaries["coverage_gated"]["final_return"],
            summaries["group_total"]["final_return"],
        )
        self.assertGreater(
            summaries["critic_td"]["final_return"],
            summaries["group_total"]["final_return"],
        )
        self.assertGreater(
            summaries["coverage_gated"]["final_critic_fraction"],
            0.0,
        )

    def test_closed_loop_is_deterministic(self) -> None:
        config = TrainingConfig(
            train_iterations=12,
            groups_per_iteration=8,
            group_size=3,
            max_steps=8,
            eval_groups=20,
            eval_every=12,
        )
        first = run_closed_loop(
            seeds=[11, 29],
            methods=["group_total", "coverage_gated"],
            config=config,
        )
        second = run_closed_loop(
            seeds=[11, 29],
            methods=["group_total", "coverage_gated"],
            config=config,
        )
        self.assertEqual(first["method_summaries"], second["method_summaries"])
        self.assertEqual(first["summary"], second["summary"])

    def test_cli_writes_artifacts_and_autoresearch_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "closed.json"
            output_md = Path(tmpdir) / "closed.md"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.closed_loop_credit_training",
                    "--seeds",
                    "11",
                    "29",
                    "--train-iterations",
                    "8",
                    "--groups-per-iteration",
                    "6",
                    "--group-size",
                    "3",
                    "--max-steps",
                    "8",
                    "--eval-groups",
                    "12",
                    "--eval-every",
                    "8",
                    "--output-json",
                    str(output_json),
                    "--output-md",
                    str(output_md),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("coverage_minus_group_return=", completed.stdout)
            payload = json.loads(output_json.read_text())
            self.assertIn("coverage_gated", {row["method"] for row in payload["method_summaries"]})
            self.assertIn("Closed-Loop Credit Training", output_md.read_text())


if __name__ == "__main__":
    unittest.main()
