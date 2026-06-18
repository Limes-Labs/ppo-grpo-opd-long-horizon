import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.token_cost_sensitivity import run_sensitivity


class TokenCostSensitivityTests(unittest.TestCase):
    def test_zero_cost_long_wait_still_favors_critic(self) -> None:
        result = run_sensitivity(
            seeds=[11, 29, 47, 83, 131],
            scenarios=["long_wait"],
            token_costs=[0.0, 0.05],
            train_groups=70,
            eval_groups=16,
            group_size=5,
            max_steps=12,
        )
        rows = result["aggregate_rows"]

        for row in rows:
            self.assertGreater(row["delta_r"] - row["delta_ci95"], 0.25)
            self.assertLess(row["critic_wait_leak"], row["group_wait_leak"])
            self.assertAlmostEqual(row["group_within_var"], 0.0, places=12)
            self.assertGreater(row["critic_within_var"], 0.0)

    def test_sensitivity_is_deterministic(self) -> None:
        kwargs = {
            "seeds": [11, 29],
            "scenarios": ["baseline"],
            "token_costs": [0.0, 0.02],
            "train_groups": 40,
            "eval_groups": 10,
            "group_size": 4,
            "max_steps": 8,
        }
        first = run_sensitivity(**kwargs)
        second = run_sensitivity(**kwargs)
        self.assertEqual(first["aggregate_rows"], second["aggregate_rows"])
        self.assertEqual(first["summary"], second["summary"])

    def test_cli_writes_json_and_markdown_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "cost.json"
            output_md = Path(tmpdir) / "cost.md"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.token_cost_sensitivity",
                    "--seeds",
                    "11",
                    "29",
                    "--scenarios",
                    "long_wait",
                    "--token-costs",
                    "0.0",
                    "0.02",
                    "--train-groups",
                    "30",
                    "--eval-groups",
                    "8",
                    "--group-size",
                    "4",
                    "--max-steps",
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

            self.assertIn("clear_positive:", completed.stdout)
            payload = json.loads(output_json.read_text())
            self.assertEqual(len(payload["aggregate_rows"]), 2)
            self.assertIn("Token-Cost Sensitivity", output_md.read_text())


if __name__ == "__main__":
    unittest.main()
