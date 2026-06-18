import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "autoresearch/coverage_gated_credit_spec.json"
CONFIG = ROOT / "autoresearch/closed_loop_credit_training_config.json"


class AutoResearchConfigTests(unittest.TestCase):
    def test_coverage_gated_spec_and_config_are_consistent(self) -> None:
        spec = json.loads(SPEC.read_text())
        config = json.loads(CONFIG.read_text())

        metric_names = {metric["name"] for metric in spec["metrics"]}
        config_metrics = set(config["metric_keys"])
        self.assertEqual(config_metrics, metric_names)
        self.assertEqual(
            spec["promotion_gate"]["metric"],
            "coverage_minus_group_return",
        )
        self.assertIn("experiments.closed_loop_credit_training", config["command"])
        self.assertIn("--critic-replay-limit", config["command"])
        self.assertIn("--gate-min-count", config["command"])


if __name__ == "__main__":
    unittest.main()
