import csv
import hashlib
import json
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PublicArtifactTests(unittest.TestCase):
    def test_deep_matrix_has_expected_shape_and_counterexample(self) -> None:
        matrix = json.loads((ROOT / "results/deep_matrix_20seed.json").read_text())

        self.assertEqual(matrix["seed_count"], 20)
        self.assertEqual(matrix["case_count"], 18)
        self.assertEqual(matrix["overall"]["critic_wins_by_mean_correlation"], 17)
        self.assertEqual(matrix["overall"]["group_wins_by_mean_correlation"], 1)
        self.assertEqual(matrix["overall"]["clear_critic_cases_by_ci95"], 16)
        self.assertEqual(matrix["overall"]["near_tie_cases_by_ci95"], 1)
        self.assertEqual(matrix["overall"]["clear_group_cases_by_ci95"], 1)

        cases = {case["case_name"]: case for case in matrix["cases"]}
        near_tie = cases["critic_budget_002_full"]
        self.assertEqual(near_tie["evidence_by_ci95"], "near_tie")

        counterexample = cases["blind_undercovered_counterexample"]
        self.assertEqual(counterexample["winner_by_mean_correlation"], "group")
        self.assertEqual(counterexample["evidence_by_ci95"], "group_clear")
        self.assertLess(counterexample["mean_critic_minus_group_correlation"], 0.0)

        with (ROOT / "results/deep_matrix_20seed.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 20 * 18)

    def test_manifest_hashes_match_public_outputs(self) -> None:
        manifest = json.loads((ROOT / "public/artifact_manifest.json").read_text())
        outputs = manifest["outputs"]

        required = {
            "public/ppo_grpo_opd_long_horizon.pdf",
            "public/ppo_grpo_opd_long_horizon.docx",
            "public/figures/deep_matrix_delta.png",
            "public/figures/deep_matrix_coverage.png",
        }
        self.assertTrue(required.issubset(outputs))

        for relative_path, metadata in outputs.items():
            path = ROOT / relative_path
            self.assertTrue(path.exists(), relative_path)
            self.assertEqual(path.stat().st_size, metadata["bytes"], relative_path)
            self.assertEqual(sha256_file(path), metadata["sha256"], relative_path)

    def test_pdf_and_docx_have_expected_structure(self) -> None:
        pdf = ROOT / "public/ppo_grpo_opd_long_horizon.pdf"
        docx = ROOT / "public/ppo_grpo_opd_long_horizon.docx"
        manifest = json.loads((ROOT / "public/artifact_manifest.json").read_text())
        render_checks = manifest["render_checks"]

        self.assertGreater(pdf.stat().st_size, 100_000)
        self.assertEqual(pdf.read_bytes()[:4], b"%PDF")
        self.assertEqual(render_checks["pdf"]["page_count"], 4)
        self.assertEqual(render_checks["pdf"]["text_check"]["missing_phrases"], [])

        with zipfile.ZipFile(docx) as archive:
            names = set(archive.namelist())
        self.assertIn("word/document.xml", names)
        self.assertIn("word/media/image1.png", names)
        self.assertIn("word/media/image2.png", names)
        self.assertGreaterEqual(render_checks["docx"]["table_count"], 2)
        self.assertGreaterEqual(render_checks["docx"]["inline_shape_count"], 2)
        self.assertEqual(render_checks["docx"]["text_check"]["missing_phrases"], [])
        self.assertEqual(
            render_checks["docx"]["visual_render"],
            "not_run_soffice_unavailable",
        )


if __name__ == "__main__":
    unittest.main()
