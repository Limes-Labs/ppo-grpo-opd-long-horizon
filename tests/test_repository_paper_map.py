import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryPaperMapTests(unittest.TestCase):
    def test_single_canonical_paper_is_unambiguous(self) -> None:
        readme = (ROOT / "README.md").read_text()

        self.assertFalse(
            (ROOT / "PAPER.md").exists(),
            "PAPER.md reads like a second manuscript; keep paper/main.tex canonical.",
        )
        self.assertIn("## Canonical Paper", readme)
        self.assertIn("paper/main.tex", readme)
        self.assertIn("public/trajectory_rewards_are_not_token_credit.pdf", readme)

        public_pdfs = sorted(path.name for path in (ROOT / "public").glob("*.pdf"))
        self.assertEqual(public_pdfs, ["trajectory_rewards_are_not_token_credit.pdf"])
        self.assertFalse((ROOT / "public/ppo_grpo_opd_long_horizon.pdf").exists())
        self.assertFalse((ROOT / "public/ppo_grpo_opd_long_horizon_latex.pdf").exists())
        self.assertFalse((ROOT / "public/ppo_grpo_opd_long_horizon.docx").exists())
        self.assertFalse((ROOT / "public/artifact_manifest.json").exists())

        paper_manifest = json.loads((ROOT / "public/paper_manifest.json").read_text())
        self.assertEqual(
            list(paper_manifest["outputs"]),
            ["public/trajectory_rewards_are_not_token_credit.pdf"],
        )


if __name__ == "__main__":
    unittest.main()
