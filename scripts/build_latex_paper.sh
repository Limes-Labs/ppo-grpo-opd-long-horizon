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
mkdir -p .cache .tectonic-cache

export XDG_CACHE_HOME="$ROOT_DIR/.cache"
export TECTONIC_CACHE_DIR="$ROOT_DIR/.tectonic-cache"

python3 scripts/build_latex_inputs.py

tectonic \
  --keep-logs \
  --outdir paper/build \
  paper/main.tex

cp paper/build/main.pdf public/trajectory_rewards_are_not_token_credit.pdf
python3 scripts/build_latex_manifest.py

echo "wrote public/trajectory_rewards_are_not_token_credit.pdf"
