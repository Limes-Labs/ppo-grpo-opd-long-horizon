import re
import unittest
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper/main.tex"
BIB = ROOT / "paper/references.bib"
LATEX_PDF = ROOT / "public/trajectory_rewards_are_not_token_credit.pdf"
LATEX_MANIFEST = ROOT / "public/paper_manifest.json"
MACROS = ROOT / "paper/generated/result_macros.tex"
TABLE = ROOT / "paper/generated/deep_matrix_table.tex"
VARIANCE_CREDIT_TABLE = ROOT / "paper/generated/variance_credit_table.tex"
ANCHOR_COVERAGE_TABLE = ROOT / "paper/generated/anchor_coverage_table.tex"
LENGTH_IMBALANCE_TABLE = ROOT / "paper/generated/length_imbalance_table.tex"
TOKEN_COST_TABLE = ROOT / "paper/generated/token_cost_table.tex"
CLOSED_LOOP_TABLE = ROOT / "paper/generated/closed_loop_training_table.tex"
NEURAL_TABLE = ROOT / "paper/generated/neural_generalization_table.tex"
PHASE_TABLE = ROOT / "paper/generated/credit_phase_table.tex"
POLICY_GRADIENT_TABLE = ROOT / "paper/generated/policy_gradient_table.tex"
POLICY_IMPLIED_TABLE = ROOT / "paper/generated/policy_implied_table.tex"
POLICY_BASELINE_TABLE = ROOT / "paper/generated/policy_baseline_table.tex"
MATRIX = ROOT / "results/deep_matrix_20seed.json"
VARIANCE_CREDIT = ROOT / "results/variance_credit_grid_seed17.json"
ANCHOR_COVERAGE = ROOT / "results/anchor_coverage_audit_seedset.json"
LENGTH_IMBALANCE = ROOT / "results/length_imbalance_audit_seedset.json"
TOKEN_COST = ROOT / "results/token_cost_sensitivity_20seed.json"
CLOSED_LOOP = ROOT / "results/closed_loop_credit_training_10seed.json"
CLOSED_LOOP_LOW = ROOT / "results/closed_loop_credit_training_low_coverage_10seed.json"
NEURAL = ROOT / "results/neural_credit_generalization_seedset.json"
PHASE = ROOT / "results/credit_phase_diagram_seedset.json"
POLICY_GRADIENT = ROOT / "results/policy_gradient_fidelity_seed13.json"


class LatexPaperTests(unittest.TestCase):
    def test_latex_source_has_real_paper_structure(self) -> None:
        text = TEX.read_text()

        for required in [
            r"\documentclass[11pt]{article}",
            r"\begin{abstract}",
            r"\section{Introduction}",
            r"\section{Definitions and Method Taxonomy}",
            r"\subsection{Policy-implied values}",
            r"\subsection{Variance reduction versus credit assignment}",
            r"\section{Compute and Information Costs}",
            r"\section{Formal Limitation: Information, Reliability, and Broadcast}",
            r"\section{Toy Experiment}",
            r"\section{Results}",
            r"\section{Estimator Regimes and Selection Criteria}",
            r"\section{Failure Modes and Testable Predictions}",
            r"\section{Threats to Validity}",
            r"\section{Reproducibility}",
            r"\section{Limitations}",
            r"\section{Conclusion}",
            r"\appendix",
            r"\section{Case Design and Artifact Mapping}",
            r"\section{Benchmark Reporting Protocol}",
            r"\bibliography{references}",
        ]:
            self.assertIn(required, text)

        self.assertIn(r"\author{Anonymous Authors\\Anonymous Affiliation}", text)
        self.assertIn("The Broadcast Ceiling", text)
        self.assertIn(r"\vimpo", text)
        self.assertIn(r"\brpo", text)
        self.assertIn(r"\includegraphics[width=\linewidth]{deep_matrix_delta.png}", text)
        self.assertIn(r"\DeepClearCriticCases", text)
        self.assertIn("critic-favorable", text)
        self.assertIn(r"\input{generated/policy_gradient_table.tex}", text)
        self.assertIn(r"\input{generated/policy_implied_table.tex}", text)
        self.assertIn(r"\input{generated/policy_baseline_table.tex}", text)
        self.assertIn(r"\input{generated/variance_credit_table.tex}", text)
        self.assertIn(r"\input{generated/anchor_coverage_table.tex}", text)
        self.assertIn(r"\input{generated/length_imbalance_table.tex}", text)
        self.assertIn(r"\input{generated/token_cost_table.tex}", text)
        self.assertIn(r"\input{generated/closed_loop_training_table.tex}", text)
        self.assertIn(r"\input{generated/neural_generalization_table.tex}", text)
        self.assertIn(r"\input{generated/credit_phase_table.tex}", text)
        self.assertIn("reliability-gated baseline", text)
        self.assertIn("behavior-policy advantage", text)
        self.assertIn(r"A^{\pi_b}(x,a)", text)
        self.assertIn("REINFORCE remains an unbiased", text)
        self.assertIn("information ceiling", text)
        self.assertIn("reliability or consistency error", text)
        self.assertIn(r"H_{\mathrm{credit}}", text)
        self.assertIn("trajectory-broadcast ceiling", text)
        self.assertIn("near tie", text)
        self.assertIn("not independent causal evidence", text)
        self.assertIn("not a production", text)
        self.assertIn("controlled estimator-fidelity toy", text)
        self.assertIn("Observation access", text)
        self.assertIn("anchor-action contrast", text)
        self.assertIn("exact-gradient audit", text)
        self.assertIn("closed-loop training", text)
        self.assertIn("value-critic generalization", text)
        self.assertIn("Score-weighted broadcast ceiling", text)
        self.assertIn("token-invariant group context", text)
        self.assertIn(r"\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)", text)
        self.assertIn("Benchmark Reporting Protocol", text)
        self.assertIn("Budget matching", text)
        self.assertNotIn(r"\paragraph{Closed-loop toy training.}", text)
        self.assertNotIn(r"\tableofcontents", text)
        self.assertNotIn(r"\author{Limes Labs}", text)
        self.assertNotIn("stale reference", text)
        self.assertNotIn("stale-reference", text)
        self.assertNotIn(r"\section{Raw Seed-Level Rows}", text)
        self.assertNotIn(r"\section{Raw Error and Dispersion Rows}", text)
        self.assertNotIn(r"\begin{verbatim}", text)
        self.assertNotIn("near-term reporting scorecard", text)
        self.assertNotIn("policy drift. The ceiling", text)
        self.assertFalse((ROOT / "paper/generated/raw_seed_table.tex").exists())
        self.assertFalse((ROOT / "paper/generated/raw_error_table.tex").exists())
        self.assertFalse((ROOT / "paper/generated/axis_summary_table.tex").exists())
        self.assertFalse((ROOT / "paper/generated/full_case_table.tex").exists())
        self.assertFalse((ROOT / "paper/generated/credit_phase_full_table.tex").exists())
        self.assertNotIn(r"\usepackage{longtable}", text)
        self.assertNotIn(r"\input{generated/axis_summary_table.tex}", text)
        self.assertNotIn(r"\input{generated/full_case_table.tex}", text)
        self.assertNotIn(r"\input{generated/credit_phase_full_table.tex}", text)
        self.assertNotIn(r"\mathcal{r}_t + \gamma V_\phi", text)
        self.assertNotIn(r"A^\star(s_t,a_t)", text)
        self.assertNotIn("token-level causal structure", text)
        self.assertNotIn("Public draft", text)
        self.assertNotIn("Limes Labs Research Workstream", text)
        self.assertNotIn("Which Method Is Better?", text)
        self.assertNotIn("Cost Accounting", text)
        self.assertNotIn("public headline", text)

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

        generated = sorted(path.name for path in (ROOT / "paper/generated").glob("*.tex"))
        self.assertEqual(
            generated,
            sorted(
                [
                    "anchor_coverage_table.tex",
                    "closed_loop_training_table.tex",
                    "credit_phase_table.tex",
                    "deep_matrix_table.tex",
                    "length_imbalance_table.tex",
                    "neural_generalization_table.tex",
                    "policy_baseline_table.tex",
                    "policy_gradient_table.tex",
                    "policy_implied_table.tex",
                    "result_macros.tex",
                    "token_cost_table.tex",
                    "variance_credit_table.tex",
                ]
            ),
        )
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

        neural = json.loads(NEURAL.read_text())
        neural_table = NEURAL_TABLE.read_text()
        self.assertIn("Neural TD", neural_table)
        self.assertIn("Eval keys unseen in tabular train", neural_table)
        self.assertGreaterEqual(
            neural["aggregate"]["sample_counts"]["heldout_exact_state_fraction"],
            0.99,
        )
        self.assertGreater(
            neural["aggregate"]["estimators"]["neural_critic_td"]["pearson_correlation"],
            neural["aggregate"]["estimators"]["group_relative"]["pearson_correlation"] + 0.30,
        )

        phase = json.loads(PHASE.read_text())
        phase_table = PHASE_TABLE.read_text()
        self.assertIn(r"$H_{\mathrm{credit}}$", phase_table)
        self.assertIn("Crossover", phase_table)
        self.assertGreaterEqual(phase["summary"]["cell_count"], 48)
        self.assertGreaterEqual(phase["summary"]["critic_clear_cells"], 1)
        self.assertGreaterEqual(phase["summary"]["group_clear_cells"], 1)
        self.assertLessEqual(phase["summary"]["min_credit_heterogeneity"], 0.05)
        self.assertGreaterEqual(phase["summary"]["max_credit_heterogeneity"], 0.80)
        for row in phase["aggregate_rows"]:
            self.assertLessEqual(
                abs(row["group_correlation"]),
                row["broadcast_ceiling_correlation"] + 1e-9,
            )

        policy_gradient = json.loads(POLICY_GRADIENT.read_text())
        gradient_table = POLICY_GRADIENT_TABLE.read_text()
        policy_implied_table = POLICY_IMPLIED_TABLE.read_text()
        baseline_table = POLICY_BASELINE_TABLE.read_text()
        self.assertIn("Learned value TD", gradient_table)
        self.assertIn("Oracle-value TD", gradient_table)
        self.assertIn("VIMPO", policy_implied_table)
        self.assertIn("VIMPO-style", policy_implied_table)
        self.assertIn("diagnostic", policy_implied_table)
        self.assertIn("not complete", policy_implied_table)
        self.assertIn("algorithm performance", policy_implied_table)
        self.assertIn("BRPO-style", gradient_table)
        self.assertIn(r"$\pi=\pi_{\rm ref}$", policy_implied_table)
        pg_metrics = {
            entry["method"]: entry["metrics"]
            for entry in policy_gradient["estimators"]
        }
        self.assertLess(
            pg_metrics["oracle_value_td"]["variance_trace"],
            pg_metrics["reinforce_return"]["variance_trace"],
        )
        self.assertLess(
            policy_gradient["exact_gradient"]["finite_difference_relative_error"],
            1e-8,
        )
        self.assertEqual(policy_gradient["config"]["replications"], 12)
        vimpo_metrics = {
            entry["method"]: entry
            for entry in policy_gradient["policy_implied_signals"]
        }
        self.assertEqual(
            vimpo_metrics["vimpo_actor_equal_ref"]["metrics"]["mean_gradient_norm"],
            0.0,
        )
        self.assertGreater(
            vimpo_metrics["vimpo_actor_fixed_ref_far"]["reference_kl"],
            vimpo_metrics["vimpo_actor_fixed_ref_near"]["reference_kl"],
        )
        self.assertIn("Score-wtd. resid.", baseline_table)
        self.assertLess(
            policy_gradient["position_diagnostics"]["overall"]["critic_value"][
                "residual_variance_ratio"
            ],
            policy_gradient["position_diagnostics"]["overall"]["group_mean"][
                "residual_variance_ratio"
            ],
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
            "yue2025vapo",
            "yang2026groupbiased",
            "li2026ecpo",
            "zhang2026creditassignment",
            "kang2026vimpo",
            "qi2025anytimereasoner",
            "luo2026dopsd",
            "fu2025areal",
            "khan2026straggler",
        ]:
            self.assertRegex(bib, rf"@\w+\{{{re.escape(key)},")

    def test_latex_pdf_artifact_exists_when_committed(self) -> None:
        self.assertTrue(LATEX_PDF.exists())
        self.assertGreater(LATEX_PDF.stat().st_size, 100_000)
        self.assertEqual(LATEX_PDF.read_bytes()[:4], b"%PDF")

        manifest = json.loads(LATEX_MANIFEST.read_text())
        output = manifest["outputs"]["public/trajectory_rewards_are_not_token_credit.pdf"]
        self.assertEqual(output["bytes"], LATEX_PDF.stat().st_size)
        self.assertEqual(manifest["checks"]["pdf_header"], "%PDF")
        self.assertTrue(manifest["checks"]["pdf_generated"])
        self.assertGreaterEqual(manifest["checks"]["page_count"], 30)
        self.assertTrue(manifest["checks"]["page_count_ok"])
        self.assertIn("paper/main.tex", manifest["inputs"])
        self.assertIn("results/deep_matrix_20seed.json", manifest["inputs"])
        self.assertIn("paper/generated/variance_credit_table.tex", manifest["inputs"])
        self.assertIn("results/variance_credit_grid_seed17.json", manifest["inputs"])
        self.assertIn("paper/generated/anchor_coverage_table.tex", manifest["inputs"])
        self.assertIn("results/anchor_coverage_audit_seedset.json", manifest["inputs"])
        self.assertIn("paper/generated/length_imbalance_table.tex", manifest["inputs"])
        self.assertIn("paper/generated/token_cost_table.tex", manifest["inputs"])
        self.assertIn("paper/generated/closed_loop_training_table.tex", manifest["inputs"])
        self.assertIn("paper/generated/neural_generalization_table.tex", manifest["inputs"])
        self.assertIn("paper/generated/credit_phase_table.tex", manifest["inputs"])
        self.assertIn("paper/generated/policy_gradient_table.tex", manifest["inputs"])
        self.assertIn("paper/generated/policy_implied_table.tex", manifest["inputs"])
        self.assertIn("paper/generated/policy_baseline_table.tex", manifest["inputs"])
        self.assertNotIn("paper/generated/axis_summary_table.tex", manifest["inputs"])
        self.assertNotIn("paper/generated/full_case_table.tex", manifest["inputs"])
        self.assertNotIn("paper/generated/credit_phase_full_table.tex", manifest["inputs"])
        self.assertNotIn("paper/generated/raw_seed_table.tex", manifest["inputs"])
        self.assertNotIn("paper/generated/raw_error_table.tex", manifest["inputs"])
        self.assertIn("results/length_imbalance_audit_seedset.json", manifest["inputs"])
        self.assertIn("results/token_cost_sensitivity_20seed.json", manifest["inputs"])
        self.assertIn("results/closed_loop_credit_training_10seed.json", manifest["inputs"])
        self.assertIn("results/closed_loop_credit_training_low_coverage_10seed.json", manifest["inputs"])
        self.assertIn("results/neural_credit_generalization_seedset.json", manifest["inputs"])
        self.assertIn("results/credit_phase_diagram_seedset.json", manifest["inputs"])
        self.assertIn("results/policy_gradient_fidelity_seed13.json", manifest["inputs"])


if __name__ == "__main__":
    unittest.main()
