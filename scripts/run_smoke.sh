#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python3}"
SMOKE_DIR="${SMOKE_DIR:-$(mktemp -d)}"

"$PYTHON_BIN" -m unittest discover -s tests

"$PYTHON_BIN" -m experiments.toy_credit_assignment \
  --seed 11 \
  --train-groups 120 \
  --eval-groups 40 \
  --group-size 6 \
  --max-steps 10 \
  --output "$SMOKE_DIR/toy_credit_assignment_smoke.json"

"$PYTHON_BIN" -m experiments.scenario_sweep \
  --seed 11 \
  --output-json "$SMOKE_DIR/toy_sweep_smoke.json" \
  --output-md "$SMOKE_DIR/toy_sweep_smoke.md"

"$PYTHON_BIN" -m experiments.variance_credit_grid \
  --seed 17 \
  --train-groups 60 \
  --eval-groups 16 \
  --group-size 5 \
  --max-steps 10 \
  --branches-per-state 12 \
  --output-json "$SMOKE_DIR/variance_credit_grid_smoke.json" \
  --output-md "$SMOKE_DIR/variance_credit_grid_smoke.md"

"$PYTHON_BIN" - "$SMOKE_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

smoke_dir = Path(sys.argv[1])
payload = json.loads((smoke_dir / "toy_credit_assignment_smoke.json").read_text())
group = payload["metrics"]["group_relative"]
critic = payload["metrics"]["critic_value_model"]
if critic["pearson_correlation"] <= group["pearson_correlation"]:
    raise SystemExit("critic estimator did not beat group-relative correlation")
if critic["calibrated_mse"] >= group["calibrated_mse"]:
    raise SystemExit("critic estimator did not beat group-relative calibrated MSE")

required_counts = {
    "train_trajectories",
    "eval_trajectories",
    "eval_tokens",
    "wait_token_fraction",
    "success_rate",
    "zero_variance_group_fraction",
    "critic_exact_state_rate",
}
missing = required_counts - set(payload["sample_counts"])
if missing:
    raise SystemExit(f"missing sample count keys: {sorted(missing)}")

def assert_finite(value, path):
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise SystemExit(f"non-finite value at {path}")

assert_finite(payload["metrics"], "metrics")

sweep = json.loads((smoke_dir / "toy_sweep_smoke.json").read_text())
if sweep["case_count"] < 5:
    raise SystemExit("scenario sweep did not run enough cases")
if sweep["summary"]["critic_wins_by_correlation"] < 1:
    raise SystemExit("scenario sweep did not include a critic-favorable case")
if sweep["summary"]["group_wins_by_correlation"] < 1:
    raise SystemExit("scenario sweep did not include a group-favorable counterexample")

grid = json.loads((smoke_dir / "variance_credit_grid_smoke.json").read_text())
estimators = {entry["name"]: entry for entry in grid["estimators"]}
if estimators["global_baseline"]["metrics"]["estimate_second_moment"] >= estimators["reinforce_return"]["metrics"]["estimate_second_moment"]:
    raise SystemExit("global baseline did not reduce the trajectory-level second moment")
if estimators["critic_td"]["metrics"]["pearson_correlation"] <= estimators["sibling_group_norm"]["metrics"]["pearson_correlation"]:
    raise SystemExit("critic TD did not beat sibling group correlation in variance/credit grid")
if abs(estimators["sibling_group_norm"]["metrics"]["within_trajectory_variance"]) > 1e-12:
    raise SystemExit("sibling group broadcast unexpectedly had step-level variation")
if estimators["critic_td"]["metrics"]["within_trajectory_variance"] <= 0.0:
    raise SystemExit("critic TD did not create step-level variation")

print(f"smoke artifacts: {smoke_dir}")
PY
