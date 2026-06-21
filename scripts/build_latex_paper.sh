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

PYTHON_BIN="${PYTHON:-python3}"
FIGURE_PYTHON="${FIGURE_PYTHON:-$PYTHON_BIN}"
if ! "$FIGURE_PYTHON" - <<'PY' >/dev/null 2>&1
from PIL import Image
PY
then
  BUNDLED_PYTHON="$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
  if [ -x "$BUNDLED_PYTHON" ] && "$BUNDLED_PYTHON" - <<'PY' >/dev/null 2>&1
from PIL import Image
PY
  then
    FIGURE_PYTHON="$BUNDLED_PYTHON"
  else
    echo "Pillow is required to regenerate public PNG figures." >&2
    echo "Install pillow or set FIGURE_PYTHON to a Python with Pillow." >&2
    exit 127
  fi
fi

export XDG_CACHE_HOME="$ROOT_DIR/.cache"
export TECTONIC_CACHE_DIR="$ROOT_DIR/.tectonic-cache"

"$FIGURE_PYTHON" scripts/render_public_figures.py
"$PYTHON_BIN" scripts/build_latex_inputs.py

tectonic \
  --keep-logs \
  --outdir paper/build \
  paper/main.tex

cp paper/build/main.pdf public/trajectory_rewards_are_not_token_credit.pdf
"$PYTHON_BIN" scripts/build_latex_manifest.py

echo "wrote public/trajectory_rewards_are_not_token_credit.pdf"
