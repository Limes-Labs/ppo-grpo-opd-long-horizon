#!/usr/bin/env python3
"""Write a manifest for the LaTeX paper artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_value(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True).strip()
    except Exception:
        return "unknown"


def file_payload(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def pdf_page_count(path: Path) -> int:
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo:
        output = subprocess.check_output([pdfinfo, str(path)], text=True)
        for line in output.splitlines():
            if line.startswith("Pages:"):
                return int(line.split(":", 1)[1].strip())

    try:
        from pypdf import PdfReader  # type: ignore

        return len(PdfReader(str(path)).pages)
    except Exception:
        data = path.read_bytes()
        return len(re.findall(rb"/Type\s*/Page\b", data))


def build_manifest(repo: Path, output_pdf: Path, min_pages: int) -> dict[str, Any]:
    inputs = [
        Path("paper/main.tex"),
        Path("paper/references.bib"),
        Path("paper/generated/result_macros.tex"),
        Path("paper/generated/deep_matrix_table.tex"),
        Path("paper/generated/axis_summary_table.tex"),
        Path("paper/generated/full_case_table.tex"),
        Path("paper/generated/raw_seed_table.tex"),
        Path("paper/generated/raw_error_table.tex"),
        Path("paper/generated/variance_credit_table.tex"),
        Path("paper/generated/anchor_coverage_table.tex"),
        Path("paper/generated/length_imbalance_table.tex"),
        Path("paper/generated/token_cost_table.tex"),
        Path("paper/generated/closed_loop_training_table.tex"),
        Path("paper/generated/neural_generalization_table.tex"),
        Path("results/deep_matrix_20seed.json"),
        Path("results/variance_credit_grid_seed17.json"),
        Path("results/anchor_coverage_audit_seedset.json"),
        Path("results/length_imbalance_audit_seedset.json"),
        Path("results/token_cost_sensitivity_20seed.json"),
        Path("results/closed_loop_credit_training_10seed.json"),
        Path("results/closed_loop_credit_training_low_coverage_10seed.json"),
        Path("results/neural_credit_generalization_seedset.json"),
        Path("public/figures/deep_matrix_delta.png"),
        Path("public/figures/deep_matrix_coverage.png"),
    ]
    page_count = pdf_page_count(repo / output_pdf)
    return {
        "artifact_scope": "full LaTeX paper artifact",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/build_latex_paper.sh",
        "tex_engine": command_value(["tectonic", "--version"], repo),
        "git_commit": command_value(["git", "rev-parse", "HEAD"], repo),
        "git_context_note": (
            "The manifest records content hashes and the source commit visible "
            "at generation time. It intentionally omits dirty-worktree status "
            "because generated artifacts are committed together with the "
            "manifest; rebuild from a clean checkout to refresh provenance."
        ),
        "inputs": {
            str(path): file_payload(repo / path)
            for path in inputs
            if (repo / path).exists()
        },
        "outputs": {
            str(output_pdf): file_payload(repo / output_pdf),
        },
        "checks": {
            "pdf_header": (repo / output_pdf).read_bytes()[:4].decode("ascii", "replace"),
            "pdf_generated": (repo / output_pdf).exists()
            and (repo / output_pdf).stat().st_size > 100_000,
            "page_count": page_count,
            "min_pages": min_pages,
            "page_count_ok": page_count >= min_pages,
            "result_source": "results/deep_matrix_20seed.json",
            "variance_credit_source": "results/variance_credit_grid_seed17.json",
            "anchor_coverage_source": "results/anchor_coverage_audit_seedset.json",
            "length_imbalance_source": "results/length_imbalance_audit_seedset.json",
            "token_cost_source": "results/token_cost_sensitivity_20seed.json",
            "closed_loop_source": "results/closed_loop_credit_training_10seed.json",
            "closed_loop_low_coverage_source": "results/closed_loop_credit_training_low_coverage_10seed.json",
            "neural_generalization_source": "results/neural_credit_generalization_seedset.json",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("public/trajectory_rewards_are_not_token_credit.pdf"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("public/paper_manifest.json"),
    )
    parser.add_argument("--min-pages", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path.cwd()
    manifest = build_manifest(repo, args.output_pdf, args.min_pages)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.manifest}")
    if not manifest["checks"]["page_count_ok"]:
        raise SystemExit(
            "LaTeX PDF has "
            f"{manifest['checks']['page_count']} pages; expected at least {args.min_pages}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
