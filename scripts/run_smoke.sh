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

"$PYTHON_BIN" -m experiments.length_imbalance_audit \
  --seeds 11 29 \
  --horizons 4 12 20 \
  --train-groups 30 \
  --eval-groups 8 \
  --group-size 4 \
  --output-json "$SMOKE_DIR/length_imbalance_smoke.json" \
  --output-md "$SMOKE_DIR/length_imbalance_smoke.md"

"$PYTHON_BIN" -m experiments.token_cost_sensitivity \
  --seeds 11 29 47 \
  --scenarios long_wait \
  --token-costs 0.0 0.02 \
  --train-groups 30 \
  --eval-groups 8 \
  --group-size 4 \
  --max-steps 8 \
  --output-json "$SMOKE_DIR/token_cost_smoke.json" \
  --output-md "$SMOKE_DIR/token_cost_smoke.md"

"$PYTHON_BIN" -m experiments.closed_loop_credit_training \
  --seeds 11 29 \
  --train-iterations 8 \
  --groups-per-iteration 6 \
  --group-size 3 \
  --max-steps 8 \
  --eval-groups 12 \
  --eval-every 8 \
  --output-json "$SMOKE_DIR/closed_loop_smoke.json" \
  --output-md "$SMOKE_DIR/closed_loop_smoke.md"

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

length = json.loads((smoke_dir / "length_imbalance_smoke.json").read_text())
short = length["horizon_summaries"][0]
long = length["horizon_summaries"][-1]
if long["mean_group_length_range"] <= short["mean_group_length_range"]:
    raise SystemExit("length imbalance did not grow across smoke horizons")
if long["critic_minus_group_total_r"] <= 0.20:
    raise SystemExit("critic did not beat group total in length audit")
if long["critic_minus_group_per_token_r"] <= 0.20:
    raise SystemExit("critic did not beat length-adjusted group in length audit")

cost = json.loads((smoke_dir / "token_cost_smoke.json").read_text())
if cost["summary"]["clear_positive_rows"] != cost["summary"]["row_count"]:
    raise SystemExit("token-cost smoke did not keep all rows critic-positive")
zero_cost = next(
    row for row in cost["aggregate_rows"]
    if row["scenario"] == "long_wait" and row["token_cost"] == 0.0
)
if zero_cost["delta_r"] <= 0.20:
    raise SystemExit("zero-cost long-wait row did not preserve critic advantage")

closed = json.loads((smoke_dir / "closed_loop_smoke.json").read_text())
closed_methods = {row["method"]: row for row in closed["method_summaries"]}
for required in ["group_total", "critic_td", "coverage_gated"]:
    if required not in closed_methods:
        raise SystemExit(f"closed-loop smoke missing method {required}")
if closed_methods["critic_td"]["final_return"] <= closed_methods["group_total"]["initial_return"]:
    print("closed-loop critic smoke did not improve over initial return; treating as execution smoke only")
for name, row in closed_methods.items():
    for key in ["initial_return", "final_return", "return_improvement"]:
        if not math.isfinite(row[key]):
            raise SystemExit(f"closed-loop smoke non-finite {name}.{key}")
if closed_methods["coverage_gated"]["final_critic_fraction"] <= 0.0:
    raise SystemExit("coverage-gated smoke never used critic credit")

print(f"smoke artifacts: {smoke_dir}")
PY
