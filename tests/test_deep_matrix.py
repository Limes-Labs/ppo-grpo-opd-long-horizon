import tempfile
import unittest
from pathlib import Path

from experiments.deep_matrix import (
    DEFAULT_DEEP_CASES,
    build_deep_markdown_report,
    render_deep_charts,
    run_deep_matrix,
)


class DeepMatrixTests(unittest.TestCase):
    def test_deep_matrix_aggregates_multiple_seeds_and_axes(self) -> None:
        cases = DEFAULT_DEEP_CASES[:4]
        result = run_deep_matrix(seeds=[3, 7], cases=cases)

        self.assertEqual(result["seed_count"], 2)
        self.assertEqual(result["case_count"], len(cases))
        self.assertEqual(len(result["cases"]), len(cases))
        self.assertIn("overall", result)

        for case in result["cases"]:
            self.assertIn("mean_critic_minus_group_correlation", case)
            self.assertIn("ci95_critic_minus_group_correlation", case)
            self.assertIn(case["winner_by_mean_correlation"], {"critic", "group", "tie"})
            self.assertIn(
                case["evidence_by_ci95"],
                {"critic_clear", "group_clear", "near_tie"},
            )

    def test_deep_report_and_charts_are_generated(self) -> None:
        result = run_deep_matrix(seeds=[3, 7], cases=DEFAULT_DEEP_CASES[:4])
        markdown = build_deep_markdown_report(result)

        self.assertIn("# Deep Toy Matrix", markdown)
        self.assertIn("95% CI", markdown)
        self.assertIn("Case", markdown)
        self.assertIn("CI read", markdown)

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            charts = render_deep_charts(result, outdir)
            self.assertGreaterEqual(len(charts), 2)
            for chart in charts:
                self.assertTrue(chart.exists())
                self.assertGreater(chart.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
