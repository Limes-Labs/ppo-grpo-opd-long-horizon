import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.anchor_coverage_audit import run_audit


class AnchorCoverageAuditTests(unittest.TestCase):
    def test_anchor_needs_repeated_state_action_coverage(self) -> None:
        result = run_audit(
            seeds=[11, 29, 47],
            eval_groups_values=[2, 8, 32],
            train_groups=80,
            group_size=5,
            max_steps=12,
            branches_per_state=8,
        )
        rows = result["aggregate_rows"]
        low, middle, high = rows

        self.assertLess(
            low["supported_step_fraction"],
            middle["supported_step_fraction"],
        )
        self.assertLess(
            middle["supported_step_fraction"],
            high["supported_step_fraction"],
        )
        self.assertLess(low["anchor_minus_sibling_r"], 0.0)
        self.assertGreater(high["anchor_minus_sibling_r"], 0.10)
        self.assertGreater(high["critic_minus_anchor_r"], 0.0)
        self.assertGreater(high["anchor_within_var"], 0.0)

    def test_audit_is_deterministic(self) -> None:
        kwargs = {
            "seeds": [11, 29],
            "eval_groups_values": [2, 16],
            "train_groups": 60,
            "group_size": 4,
            "max_steps": 10,
            "branches_per_state": 6,
        }
        first = run_audit(**kwargs)
        second = run_audit(**kwargs)
        self.assertEqual(first["aggregate_rows"], second["aggregate_rows"])
        self.assertEqual(first["summary"], second["summary"])

    def test_cli_writes_json_and_markdown_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "anchor.json"
            output_md = Path(tmpdir) / "anchor.md"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.anchor_coverage_audit",
                    "--seeds",
                    "11",
                    "29",
                    "--eval-groups-values",
                    "2",
                    "16",
                    "--train-groups",
                    "40",
                    "--group-size",
                    "4",
                    "--max-steps",
                    "10",
                    "--branches-per-state",
                    "6",
                    "--output-json",
                    str(output_json),
                    "--output-md",
                    str(output_md),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("coverage:", completed.stdout)
            payload = json.loads(output_json.read_text())
            self.assertEqual(len(payload["aggregate_rows"]), 2)
            self.assertIn("Anchor Coverage Audit", output_md.read_text())


if __name__ == "__main__":
    unittest.main()
