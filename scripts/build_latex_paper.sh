#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v tectonic >/dev/null 2>&1; then
  echo "tectonic is required to build the LaTeX paper" >&2
  echo "Install Tectonic or compile paper/main.tex with an equivalent LaTeX engine." >&2
  exit 127
fi

mkdir -p public
mkdir -p paper/build

python3 scripts/build_latex_inputs.py

tectonic \
  --keep-logs \
  --outdir paper/build \
  paper/main.tex

cp paper/build/main.pdf public/ppo_grpo_opd_long_horizon_latex.pdf
python3 scripts/build_latex_manifest.py

echo "wrote public/ppo_grpo_opd_long_horizon_latex.pdf"
