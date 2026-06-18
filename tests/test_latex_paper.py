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
MATRIX = ROOT / "results/deep_matrix_20seed.json"


class LatexPaperTests(unittest.TestCase):
    def test_latex_source_has_real_paper_structure(self) -> None:
        text = TEX.read_text()

        for required in [
            r"\documentclass[11pt]{article}",
            r"\begin{abstract}",
            r"\section{Introduction}",
            r"\section{Definitions and Method Taxonomy}",
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
        self.assertIn(r"\DeepClearCriticCases\ clear critic-favorable cases", text)
        self.assertIn("near tie", text)
        self.assertIn("not independent causal evidence", text)
        self.assertIn("not a closed-loop", text)

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

    def test_bibliography_covers_required_sources(self) -> None:
        bib = BIB.read_text()
        for key in [
            "schulman2017ppo",
            "shao2024deepseekmath",
            "deepseek2025r1",
            "zhao2026opsd",
            "li2026opd",
            "luo2026opd",
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


if __name__ == "__main__":
    unittest.main()
