import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.toy_credit_assignment import run_experiment


class ToyCreditAssignmentTests(unittest.TestCase):
    def test_value_model_estimator_beats_group_relative_on_toy_oracle(self) -> None:
        result = run_experiment(
            seed=11,
            train_groups=120,
            eval_groups=40,
            group_size=6,
            max_steps=10,
        )
        metrics = result["metrics"]
        group = metrics["group_relative"]
        critic = metrics["critic_value_model"]

        self.assertGreater(
            critic["pearson_correlation"],
            group["pearson_correlation"] + 0.20,
        )
        self.assertLess(
            critic["calibrated_mse"],
            group["calibrated_mse"],
        )
        self.assertEqual(result["sample_counts"]["eval_trajectories"], 240)

    def test_experiment_is_deterministic_for_same_seed(self) -> None:
        first = run_experiment(
            seed=5,
            train_groups=80,
            eval_groups=24,
            group_size=5,
            max_steps=9,
        )
        second = run_experiment(
            seed=5,
            train_groups=80,
            eval_groups=24,
            group_size=5,
            max_steps=9,
        )

        self.assertEqual(first["metrics"], second["metrics"])
        self.assertEqual(first["sample_counts"], second["sample_counts"])

    def test_cli_writes_json_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "toy.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.toy_credit_assignment",
                    "--seed",
                    "3",
                    "--train-groups",
                    "60",
                    "--eval-groups",
                    "16",
                    "--group-size",
                    "4",
                    "--max-steps",
                    "8",
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("pearson:", completed.stdout)
            payload = json.loads(output.read_text())
            self.assertIn("group_relative", payload["metrics"])
            self.assertIn("critic_value_model", payload["metrics"])


if __name__ == "__main__":
    unittest.main()

