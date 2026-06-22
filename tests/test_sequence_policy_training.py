import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.sequence_policy_training import (
    SequencePolicyConfig,
    run_sequence_policy_audit,
)


class SequencePolicyTrainingTests(unittest.TestCase):
    def test_neural_td_beats_group_broadcast_for_sequence_policy(self) -> None:
        result = run_sequence_policy_audit(
            seeds=[11, 29, 47],
            config=SequencePolicyConfig(
                train_iterations=8,
                groups_per_iteration=8,
                group_size=3,
                max_steps=8,
                eval_groups=40,
                eval_every=8,
                policy_hidden_size=6,
                critic_hidden_size=6,
                learning_rate=0.02,
                critic_epochs=3,
                max_critic_examples=300,
            ),
        )
        summaries = {row["method"]: row for row in result["method_summaries"]}

        self.assertIn("group_broadcast", summaries)
        self.assertIn("neural_value_td", summaries)
        self.assertEqual(result["config"]["policy_family"], "autoregressive_mlp")
        self.assertGreater(
            summaries["neural_value_td"]["final_return"],
            summaries["group_broadcast"]["final_return"],
        )
        self.assertGreater(
            summaries["neural_value_td"]["return_improvement"],
            summaries["group_broadcast"]["return_improvement"],
        )
        self.assertLess(
            summaries["neural_value_td"]["final_wait_fraction"],
            summaries["group_broadcast"]["final_wait_fraction"],
        )
        self.assertEqual(result["summary"]["paired_seed_count"], 3)
        self.assertEqual(len(result["summary"]["neural_minus_group_return_ci95"]), 2)

    def test_sequence_policy_audit_is_deterministic(self) -> None:
        config = SequencePolicyConfig(
            train_iterations=12,
            groups_per_iteration=6,
            group_size=3,
            max_steps=8,
            eval_groups=16,
            eval_every=12,
            policy_hidden_size=6,
            critic_hidden_size=6,
            critic_epochs=5,
            max_critic_examples=600,
        )
        first = run_sequence_policy_audit(
            seeds=[11, 29],
            methods=["group_broadcast", "neural_value_td"],
            config=config,
        )
        second = run_sequence_policy_audit(
            seeds=[11, 29],
            methods=["group_broadcast", "neural_value_td"],
            config=config,
        )

        self.assertEqual(first["method_summaries"], second["method_summaries"])
        self.assertEqual(first["summary"], second["summary"])

    def test_cli_writes_json_and_markdown_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "sequence.json"
            output_md = Path(tmpdir) / "sequence.md"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.sequence_policy_training",
                    "--seeds",
                    "11",
                    "29",
                    "--train-iterations",
                    "10",
                    "--groups-per-iteration",
                    "6",
                    "--group-size",
                    "3",
                    "--max-steps",
                    "8",
                    "--eval-groups",
                    "12",
                    "--eval-every",
                    "10",
                    "--policy-hidden-size",
                    "6",
                    "--critic-hidden-size",
                    "6",
                    "--critic-epochs",
                    "5",
                    "--max-critic-examples",
                    "600",
                    "--output-json",
                    str(output_json),
                    "--output-md",
                    str(output_md),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("neural_minus_group_return=", completed.stdout)
            payload = json.loads(output_json.read_text())
            self.assertEqual(payload["summary"]["paired_seed_count"], 2)
            self.assertIn(
                "neural_value_td",
                {row["method"] for row in payload["method_summaries"]},
            )
            markdown = output_md.read_text()
            self.assertIn("Sequence-Policy Training Audit", markdown)
            self.assertIn("paired seeds", markdown)
            self.assertIn("95% paired CI", markdown)


if __name__ == "__main__":
    unittest.main()
