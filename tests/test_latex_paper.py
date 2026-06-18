import re
import unittest
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper/main.tex"
BIB = ROOT / "paper/references.bib"
LATEX_PDF = ROOT / "public/ppo_grpo_opd_long_horizon_latex.pdf"
LATEX_MANIFEST = ROOT / "public/latex_artifact_manifest.json"
MACROS = ROOT / "paper/generated/result_macros.tex"
TABLE = ROOT / "paper/generated/deep_matrix_table.tex"
AXIS_TABLE = ROOT / "paper/generated/axis_summary_table.tex"
FULL_CASE_TABLE = ROOT / "paper/generated/full_case_table.tex"
RAW_SEED_TABLE = ROOT / "paper/generated/raw_seed_table.tex"
RAW_ERROR_TABLE = ROOT / "paper/generated/raw_error_table.tex"
VARIANCE_CREDIT_TABLE = ROOT / "paper/generated/variance_credit_table.tex"
ANCHOR_COVERAGE_TABLE = ROOT / "paper/generated/anchor_coverage_table.tex"
LENGTH_IMBALANCE_TABLE = ROOT / "paper/generated/length_imbalance_table.tex"
TOKEN_COST_TABLE = ROOT / "paper/generated/token_cost_table.tex"
CLOSED_LOOP_TABLE = ROOT / "paper/generated/closed_loop_training_table.tex"
MATRIX = ROOT / "results/deep_matrix_20seed.json"
VARIANCE_CREDIT = ROOT / "results/variance_credit_grid_seed17.json"
ANCHOR_COVERAGE = ROOT / "results/anchor_coverage_audit_seedset.json"
LENGTH_IMBALANCE = ROOT / "results/length_imbalance_audit_seedset.json"
TOKEN_COST = ROOT / "results/token_cost_sensitivity_20seed.json"
CLOSED_LOOP = ROOT / "results/closed_loop_credit_training_10seed.json"
CLOSED_LOOP_LOW = ROOT / "results/closed_loop_credit_training_low_coverage_10seed.json"


class LatexPaperTests(unittest.TestCase):
    def test_latex_source_has_real_paper_structure(self) -> None:
        text = TEX.read_text()

        for required in [
            r"\documentclass[11pt]{article}",
            r"\begin{abstract}",
            r"\section{Introduction}",
            r"\section{Definitions and Method Taxonomy}",
            r"\subsection{Variance reduction versus credit assignment}",
            r"\section{Cost Accounting}",
            r"\section{Toy Experiment}",
            r"\section{Results}",
            r"\section{Failure Modes and Testable Predictions}",
            r"\section{Threats to Validity}",
            r"\section{Reproducibility}",
            r"\section{Limitations}",
            r"\section{Roadmap for Limes Labs}",
            r"\section{Conclusion}",
            r"\appendix",
            r"\section{Full Case Summary}",
            r"\section{Raw Seed-Level Rows}",
            r"\section{Raw Error and Dispersion Rows}",
            r"\bibliography{references}",
        ]:
            self.assertIn(required, text)

        self.assertIn(r"\includegraphics[width=\linewidth]{deep_matrix_delta.png}", text)
        self.assertIn(r"\DeepClearCriticCases", text)
        self.assertIn("clear critic-favorable", text)
        self.assertIn(r"\input{generated/variance_credit_table.tex}", text)
        self.assertIn(r"\input{generated/anchor_coverage_table.tex}", text)
        self.assertIn(r"\input{generated/length_imbalance_table.tex}", text)
        self.assertIn(r"\input{generated/token_cost_table.tex}", text)
        self.assertIn(r"\input{generated/closed_loop_training_table.tex}", text)
        self.assertIn("coverage-gated credit", text)
        self.assertIn("near tie", text)
        self.assertIn("not independent causal evidence", text)
        self.assertIn("not a production", text)
        self.assertIn("controlled estimator-fidelity toy", text)
        self.assertIn("anchor-action contrast", text)
        self.assertIn("coverage-gated credit", text)
        self.assertIn("tabular closed-loop training", text)
        self.assertIn(r"\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)", text)
        self.assertIn(r"\paragraph{Full clipped \ppo/\grpo\ training.}", text)
        self.assertIn(r"\paragraph{Tiny neural policy.}", text)
        self.assertNotIn(r"\paragraph{Closed-loop toy training.}", text)
        self.assertNotIn(r"\mathcal{r}_t + \gamma V_\phi", text)

    def test_generated_latex_inputs_match_matrix_json(self) -> None:
        matrix = json.loads(MATRIX.read_text())
        macros = MACROS.read_text()
        table = TABLE.read_text()

        expected = {
            "DeepSeedCount": matrix["seed_count"],
            "DeepCaseCount": matrix["case_count"],
            "DeepMeanCriticWins": matrix["overall"]["critic_wins_by_mean_correlation"],
            "DeepClearCriticCases": matrix["overall"]["clear_critic_cases_by_ci95"],
            "DeepNearTieCases": matrix["overall"]["near_tie_cases_by_ci95"],
            "DeepClearGroupCases": matrix["overall"]["clear_group_cases_by_ci95"],
        }
        for name, value in expected.items():
            self.assertIn(rf"\newcommand{{\{name}}}{{{value}}}", macros)

        self.assertIn(r"critic\_budget\_002\_full", table)
        self.assertIn("near tie", table)
        self.assertIn(r"blind\_undercovered\_counterexample", table)
        self.assertIn("group clear", table)

        self.assertIn("critic\\_budget", AXIS_TABLE.read_text())
        self.assertIn(r"\begin{longtable}", FULL_CASE_TABLE.read_text())
        raw_seed = RAW_SEED_TABLE.read_text()
        self.assertIn(r"\begin{longtable}", raw_seed)
        self.assertGreaterEqual(raw_seed.count(r"\\"), 360)
        raw_error = RAW_ERROR_TABLE.read_text()
        self.assertIn(r"\begin{longtable}", raw_error)
        self.assertIn("Group MSE", raw_error)
        self.assertGreaterEqual(raw_error.count(r"\\"), 360)

        variance_credit = json.loads(VARIANCE_CREDIT.read_text())
        variance_table = VARIANCE_CREDIT_TABLE.read_text()
        self.assertIn("Critic TD", variance_table)
        self.assertIn("Sampled MC", variance_table)
        self.assertIn("Anchor ctr.", variance_table)
        self.assertIn(r"\begin{tabular}", variance_table)
        self.assertEqual(
            variance_credit["summary"]["best_non_oracle_by_correlation"],
            "critic_td",
        )
        variance_estimators = {
            entry["name"]: entry["metrics"] for entry in variance_credit["estimators"]
        }
        self.assertGreater(
            variance_estimators["anchor_action_contrast"]["pearson_correlation"],
            variance_estimators["sibling_group_norm"]["pearson_correlation"],
        )
        self.assertLess(
            variance_estimators["anchor_action_contrast"]["pearson_correlation"],
            variance_estimators["critic_td"]["pearson_correlation"],
        )

        anchor_coverage = json.loads(ANCHOR_COVERAGE.read_text())
        anchor_table = ANCHOR_COVERAGE_TABLE.read_text()
        self.assertIn("Anchor $r$", anchor_table)
        self.assertEqual(
            anchor_coverage["summary"]["first_eval_groups_anchor_beats_sibling"],
            8,
        )
        self.assertEqual(
            anchor_coverage["summary"]["critic_above_anchor_rows"],
            len(anchor_coverage["aggregate_rows"]),
        )

        length = json.loads(LENGTH_IMBALANCE.read_text())
        length_table = LENGTH_IMBALANCE_TABLE.read_text()
        self.assertIn(r"$H_{\max}$", length_table)
        self.assertEqual(
            length["summary"]["critic_wins_vs_group_total"],
            len(length["horizon_summaries"]),
        )

        token_cost = json.loads(TOKEN_COST.read_text())
        token_table = TOKEN_COST_TABLE.read_text()
        self.assertIn("Wait$_G$", token_table)
        self.assertEqual(
            token_cost["summary"]["clear_positive_rows"],
            token_cost["summary"]["row_count"],
        )

        closed_loop = json.loads(CLOSED_LOOP.read_text())
        closed_loop_low = json.loads(CLOSED_LOOP_LOW.read_text())
        closed_table = CLOSED_LOOP_TABLE.read_text()
        self.assertIn("Cov-gated", closed_table)
        self.assertEqual(
            closed_loop["summary"]["best_by_final_return"],
            "critic_td",
        )
        self.assertGreater(
            closed_loop["summary"]["coverage_minus_group_return"],
            0.0,
        )
        self.assertGreater(
            closed_loop_low["summary"]["coverage_minus_group_return"],
            0.0,
        )

    def test_bibliography_covers_required_sources(self) -> None:
        bib = BIB.read_text()
        for key in [
            "schulman2017ppo",
            "shao2024deepseekmath",
            "deepseek2025r1",
            "zhao2026opsd",
            "li2026opd",
            "luo2026opd",
            "hu2025reinforcepp",
            "ahmadian2024backtobasics",
            "xu2025singlestream",
            "kazemnejad2024vineppo",
            "zhou2024archer",
            "lightman2023verify",
            "wang2023mathshepherd",
            "yuan2024freeprocess",
            "feng2025gigpo",
            "li2025salt",
            "arjona2018rudder",
            "guo2025segmentpo",
            "li2026oppo",
            "zai2026glm52",
        ]:
            self.assertRegex(bib, rf"@\w+\{{{re.escape(key)},")

    def test_latex_pdf_artifact_exists_when_committed(self) -> None:
        self.assertTrue(LATEX_PDF.exists())
        self.assertGreater(LATEX_PDF.stat().st_size, 100_000)
        self.assertEqual(LATEX_PDF.read_bytes()[:4], b"%PDF")

        manifest = json.loads(LATEX_MANIFEST.read_text())
        output = manifest["outputs"]["public/ppo_grpo_opd_long_horizon_latex.pdf"]
        self.assertEqual(output["bytes"], LATEX_PDF.stat().st_size)
        self.assertEqual(manifest["checks"]["pdf_header"], "%PDF")
        self.assertTrue(manifest["checks"]["pdf_generated"])
        self.assertGreaterEqual(manifest["checks"]["page_count"], 30)
        self.assertTrue(manifest["checks"]["page_count_ok"])
        self.assertIn("paper/main.tex", manifest["inputs"])
        self.assertIn("results/deep_matrix_20seed.json", manifest["inputs"])
        self.assertIn("paper/generated/raw_seed_table.tex", manifest["inputs"])
        self.assertIn("paper/generated/raw_error_table.tex", manifest["inputs"])
        self.assertIn("paper/generated/variance_credit_table.tex", manifest["inputs"])
        self.assertIn("results/variance_credit_grid_seed17.json", manifest["inputs"])
        self.assertIn("paper/generated/anchor_coverage_table.tex", manifest["inputs"])
        self.assertIn("results/anchor_coverage_audit_seedset.json", manifest["inputs"])
        self.assertIn("paper/generated/length_imbalance_table.tex", manifest["inputs"])
        self.assertIn("paper/generated/token_cost_table.tex", manifest["inputs"])
        self.assertIn("paper/generated/closed_loop_training_table.tex", manifest["inputs"])
        self.assertIn("results/length_imbalance_audit_seedset.json", manifest["inputs"])
        self.assertIn("results/token_cost_sensitivity_20seed.json", manifest["inputs"])
        self.assertIn("results/closed_loop_credit_training_10seed.json", manifest["inputs"])
        self.assertIn("results/closed_loop_credit_training_low_coverage_10seed.json", manifest["inputs"])


if __name__ == "__main__":
    unittest.main()
