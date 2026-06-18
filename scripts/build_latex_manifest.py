#!/usr/bin/env python3
"""Write a manifest for the LaTeX paper artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def build_manifest(repo: Path, output_pdf: Path) -> dict[str, Any]:
    inputs = [
        Path("paper/main.tex"),
        Path("paper/references.bib"),
        Path("paper/generated/result_macros.tex"),
        Path("paper/generated/deep_matrix_table.tex"),
        Path("results/deep_matrix_20seed.json"),
        Path("public/figures/deep_matrix_delta.png"),
        Path("public/figures/deep_matrix_coverage.png"),
    ]
    return {
        "artifact_scope": "full LaTeX paper artifact",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/build_latex_paper.sh",
        "tex_engine": command_value(["tectonic", "--version"], repo),
        "git_commit": command_value(["git", "rev-parse", "HEAD"], repo),
        "git_status_short": command_value(["git", "status", "--short"], repo),
        "git_context_note": (
            "The manifest records the worktree at generation time. When the "
            "LaTeX source, PDF, and manifest are committed together, the final "
            "commit will necessarily differ; rebuild from a clean checkout to "
            "refresh it."
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
            "result_source": "results/deep_matrix_20seed.json",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("public/ppo_grpo_opd_long_horizon_latex.pdf"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("public/latex_artifact_manifest.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path.cwd()
    manifest = build_manifest(repo, args.output_pdf)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
