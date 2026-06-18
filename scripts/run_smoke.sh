#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 -m unittest discover -s tests

python3 -m experiments.toy_credit_assignment \
  --seed 11 \
  --train-groups 120 \
  --eval-groups 40 \
  --group-size 6 \
  --max-steps 10 \
  --output runs/toy_credit_assignment_smoke.json

python3 - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("runs/toy_credit_assignment_smoke.json").read_text())
group = payload["metrics"]["group_relative"]
critic = payload["metrics"]["critic_value_model"]
if critic["pearson_correlation"] <= group["pearson_correlation"]:
    raise SystemExit("critic estimator did not beat group-relative correlation")
if critic["calibrated_mse"] >= group["calibrated_mse"]:
    raise SystemExit("critic estimator did not beat group-relative calibrated MSE")
print("smoke artifact: runs/toy_credit_assignment_smoke.json")
PY

