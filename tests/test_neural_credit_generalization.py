import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.neural_credit_generalization import (
    NeuralGeneralizationConfig,
    run_neural_generalization,
)


class NeuralCreditGeneralizationTests(unittest.TestCase):
    def test_neural_critic_generalizes_to_heldout_threshold(self) -> None:
        result = run_neural_generalization(
            seeds=[11, 17, 23],
            config=NeuralGeneralizationConfig(
                train_thresholds=(1, 3),
                eval_thresholds=(2,),
                train_groups=60,
                eval_groups=16,
                group_size=5,
                max_steps=10,
                hidden_size=8,
                epochs=35,
                learning_rate=0.02,
                max_train_examples=2500,
            ),
        )

        group = result["aggregate"]["estimators"]["group_relative"]
        neural = result["aggregate"]["estimators"]["neural_critic_td"]
        counts = result["aggregate"]["sample_counts"]

        self.assertGreaterEqual(counts["heldout_exact_state_fraction"], 0.99)
        self.assertGreaterEqual(neural["pearson_correlation"], 0.75)
        self.assertGreater(
            neural["pearson_correlation"],
            group["pearson_correlation"] + 0.30,
        )
        self.assertLess(neural["calibrated_mse"], group["calibrated_mse"])
        self.assertAlmostEqual(
            group["within_trajectory_variance"],
            0.0,
            places=12,
        )
        self.assertGreater(neural["within_trajectory_variance"], 1e-4)
        self.assertLessEqual(neural["wait_to_active_abs_ratio"], 0.60)
        self.assertLess(
            neural["wait_to_active_abs_ratio"],
            group["wait_to_active_abs_ratio"],
        )

    def test_neural_generalization_is_deterministic(self) -> None:
        config = NeuralGeneralizationConfig(
            train_groups=25,
            eval_groups=8,
            group_size=4,
            max_steps=8,
            hidden_size=6,
            epochs=12,
            max_train_examples=900,
        )
        first = run_neural_generalization(seeds=[11, 17], config=config)
        second = run_neural_generalization(seeds=[11, 17], config=config)

        self.assertEqual(first["aggregate"], second["aggregate"])
        self.assertEqual(first["seed_results"], second["seed_results"])

    def test_cli_writes_json_and_markdown_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "neural.json"
            output_md = Path(tmpdir) / "neural.md"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.neural_credit_generalization",
                    "--seeds",
                    "11",
                    "--train-thresholds",
                    "1",
                    "3",
                    "--eval-thresholds",
                    "2",
                    "--train-groups",
                    "30",
                    "--eval-groups",
                    "8",
                    "--group-size",
                    "4",
                    "--max-steps",
                    "8",
                    "--hidden-size",
                    "6",
                    "--epochs",
                    "16",
                    "--max-train-examples",
                    "1000",
                    "--output-json",
                    str(output_json),
                    "--output-md",
                    str(output_md),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("neural_r=", completed.stdout)
            payload = json.loads(output_json.read_text())
            self.assertIn("neural_critic_td", payload["aggregate"]["estimators"])
            self.assertIn("Tiny Neural Generalization Audit", output_md.read_text())


if __name__ == "__main__":
    unittest.main()
