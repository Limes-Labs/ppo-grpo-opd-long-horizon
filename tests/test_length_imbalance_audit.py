import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.length_imbalance_audit import run_audit


class LengthImbalanceAuditTests(unittest.TestCase):
    def test_length_imbalance_grows_and_critic_keeps_step_signal(self) -> None:
        result = run_audit(
            seeds=[11, 29, 47],
            horizons=[4, 12, 20],
            train_groups=50,
            eval_groups=12,
            group_size=5,
        )
        rows = result["horizon_summaries"]
        shortest = rows[0]
        longest = rows[-1]

        self.assertGreater(
            longest["mean_group_length_range"],
            shortest["mean_group_length_range"] + 5.0,
        )
        self.assertGreater(
            longest["critic_minus_group_total_r"],
            0.25,
        )
        self.assertGreater(
            longest["critic_minus_group_per_token_r"],
            0.25,
        )
        self.assertAlmostEqual(longest["group_total_within_var"], 0.0, places=12)
        self.assertAlmostEqual(longest["group_per_token_within_var"], 0.0, places=12)
        self.assertGreater(longest["critic_within_var"], 0.0)

    def test_audit_is_deterministic(self) -> None:
        kwargs = {
            "seeds": [11, 29],
            "horizons": [4, 16],
            "train_groups": 40,
            "eval_groups": 10,
            "group_size": 4,
        }
        first = run_audit(**kwargs)
        second = run_audit(**kwargs)
        self.assertEqual(first["horizon_summaries"], second["horizon_summaries"])
        self.assertEqual(first["summary"], second["summary"])

    def test_cli_writes_json_and_markdown_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "length.json"
            output_md = Path(tmpdir) / "length.md"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.length_imbalance_audit",
                    "--seeds",
                    "11",
                    "29",
                    "--horizons",
                    "4",
                    "12",
                    "--train-groups",
                    "30",
                    "--eval-groups",
                    "8",
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

            self.assertIn("wins:", completed.stdout)
            payload = json.loads(output_json.read_text())
            self.assertEqual(len(payload["horizon_summaries"]), 2)
            self.assertIn("Length Imbalance Audit", output_md.read_text())


if __name__ == "__main__":
    unittest.main()
