import json
import tempfile
import unittest
from pathlib import Path

from experiments.scenario_sweep import build_markdown_report, run_sweep
from experiments.toy_credit_assignment import SCENARIOS, run_experiment


class ScenarioSweepTests(unittest.TestCase):
    def test_named_scenarios_change_observability_and_horizon(self) -> None:
        self.assertEqual(SCENARIOS["blind_critic"].critic_observation, "blind")
        self.assertGreater(
            SCENARIOS["long_wait"].wait_bias_after_success,
            SCENARIOS["short_dense"].wait_bias_after_success,
        )

        result = run_experiment(
            seed=13,
            scenario_name="blind_critic",
            train_groups=80,
            eval_groups=24,
            group_size=5,
            max_steps=9,
        )

        self.assertEqual(result["config"]["scenario_name"], "blind_critic")
        self.assertIn("critic_observation", result["config"])

    def test_sweep_runs_multiple_scenarios_and_writes_interpretable_results(self) -> None:
        result = run_sweep(seed=11)

        self.assertGreaterEqual(len(result["cases"]), 5)
        by_name = {case["case_name"]: case for case in result["cases"]}
        self.assertIn("long_wait_full_critic", by_name)
        self.assertIn("blind_critic_counterexample", by_name)

        long_wait = by_name["long_wait_full_critic"]["metrics"]
        self.assertGreater(
            long_wait["critic_value_model"]["pearson_correlation"],
            long_wait["group_relative"]["pearson_correlation"],
        )

        blind = by_name["blind_critic_counterexample"]["metrics"]
        self.assertLess(
            blind["critic_value_model"]["pearson_correlation"],
            blind["group_relative"]["pearson_correlation"],
        )

    def test_markdown_report_contains_methods_and_caveats(self) -> None:
        result = run_sweep(seed=11)
        markdown = build_markdown_report(result)

        self.assertIn("# Toy Scenario Sweep", markdown)
        self.assertIn("long_wait_full_critic", markdown)
        self.assertIn("blind_critic_counterexample", markdown)
        self.assertIn("Caveats", markdown)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "sweep.json"
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            payload = json.loads(output.read_text())
            self.assertEqual(payload["seed"], 11)


if __name__ == "__main__":
    unittest.main()

