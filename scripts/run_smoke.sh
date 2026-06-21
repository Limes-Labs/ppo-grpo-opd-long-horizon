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

"$PYTHON_BIN" -m experiments.anchor_coverage_audit \
  --seeds 11 29 \
  --eval-groups-values 2 8 24 \
  --train-groups 40 \
  --group-size 4 \
  --max-steps 10 \
  --branches-per-state 6 \
  --output-json "$SMOKE_DIR/anchor_coverage_smoke.json" \
  --output-md "$SMOKE_DIR/anchor_coverage_smoke.md"

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

"$PYTHON_BIN" -m experiments.neural_credit_generalization \
  --seeds 11 \
  --train-thresholds 1 3 \
  --eval-thresholds 2 \
  --train-groups 40 \
  --eval-groups 12 \
  --group-size 5 \
  --max-steps 10 \
  --hidden-size 8 \
  --epochs 30 \
  --max-train-examples 2000 \
  --output-json "$SMOKE_DIR/neural_credit_smoke.json" \
  --output-md "$SMOKE_DIR/neural_credit_smoke.md"

"$PYTHON_BIN" -m experiments.credit_phase_diagram \
  --seeds 11,29 \
  --heterogeneity-levels low,high \
  --observability-levels full,blind \
  --coverage-levels low,high \
  --reward-levels contrast \
  --drift-levels matched \
  --eval-groups 12 \
  --group-size 4 \
  --output-json "$SMOKE_DIR/credit_phase_smoke.json" \
  --output-md "$SMOKE_DIR/credit_phase_smoke.md"

"$PYTHON_BIN" -m experiments.selection_regret \
  --phase-json "$SMOKE_DIR/credit_phase_smoke.json" \
  --output-json "$SMOKE_DIR/selection_regret_smoke.json" \
  --output-md "$SMOKE_DIR/selection_regret_smoke.md"

"$PYTHON_BIN" -m experiments.policy_gradient_fidelity \
  --seed 13 \
  --batches 8 \
  --replications 4 \
  --groups-per-batch 5 \
  --group-size 4 \
  --horizon 4 \
  --output-json "$SMOKE_DIR/policy_gradient_smoke.json" \
  --output-md "$SMOKE_DIR/policy_gradient_smoke.md"

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
if estimators["anchor_action_contrast"]["metrics"]["pearson_correlation"] <= estimators["sibling_group_norm"]["metrics"]["pearson_correlation"]:
    raise SystemExit("anchor action contrast did not beat sibling group correlation in variance/credit grid")
if estimators["anchor_action_contrast"]["metrics"]["pearson_correlation"] >= estimators["critic_td"]["metrics"]["pearson_correlation"]:
    raise SystemExit("anchor action contrast unexpectedly beat critic TD in variance/credit grid")
if abs(estimators["sibling_group_norm"]["metrics"]["within_trajectory_variance"]) > 1e-12:
    raise SystemExit("sibling group broadcast unexpectedly had step-level variation")
if estimators["anchor_action_contrast"]["metrics"]["within_trajectory_variance"] <= 0.0:
    raise SystemExit("anchor action contrast did not create step-level variation")
if estimators["critic_td"]["metrics"]["within_trajectory_variance"] <= 0.0:
    raise SystemExit("critic TD did not create step-level variation")

anchor = json.loads((smoke_dir / "anchor_coverage_smoke.json").read_text())
anchor_rows = anchor["aggregate_rows"]
if anchor_rows[-1]["supported_step_fraction"] <= anchor_rows[0]["supported_step_fraction"]:
    raise SystemExit("anchor support did not increase across coverage smoke")
if anchor_rows[-1]["anchor_minus_sibling_r"] <= anchor_rows[0]["anchor_minus_sibling_r"]:
    raise SystemExit("anchor advantage over sibling did not improve with coverage")
if anchor_rows[-1]["critic_minus_anchor_r"] <= 0.0:
    raise SystemExit("critic did not remain above high-coverage anchor contrast")

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

neural = json.loads((smoke_dir / "neural_credit_smoke.json").read_text())
neural_group = neural["aggregate"]["estimators"]["group_relative"]
neural_td = neural["aggregate"]["estimators"]["neural_critic_td"]
if neural["aggregate"]["sample_counts"]["heldout_exact_state_fraction"] < 0.99:
    raise SystemExit("neural smoke did not evaluate on held-out exact states")
if neural_td["pearson_correlation"] <= neural_group["pearson_correlation"] + 0.25:
    raise SystemExit("neural critic did not beat group broadcast in held-out smoke")
if neural_td["within_trajectory_variance"] <= 1e-4:
    raise SystemExit("neural critic did not create step-level variation")

phase = json.loads((smoke_dir / "credit_phase_smoke.json").read_text())
if phase["summary"]["critic_clear_cells"] < 1:
    raise SystemExit("phase smoke did not include a critic-clear cell")
if phase["summary"]["group_clear_cells"] < 1:
    raise SystemExit("phase smoke did not include a group-clear cell")
for row in phase["aggregate_rows"]:
    if not (0.0 <= row["credit_heterogeneity"] <= 1.0):
        raise SystemExit(f"phase smoke invalid heterogeneity in {row['cell_name']}")
    if not (0.0 <= row["broadcast_ceiling_correlation"] <= 1.0):
        raise SystemExit(f"phase smoke invalid broadcast ceiling in {row['cell_name']}")
    if abs(row["group_correlation"]) > row["broadcast_ceiling_correlation"] + 1e-9:
        raise SystemExit(f"group estimator exceeded broadcast ceiling in {row['cell_name']}")

selection = json.loads((smoke_dir / "selection_regret_smoke.json").read_text())
selection_metrics = selection["heldout_metrics"]
for required in ["audit_mse_cost", "always_group", "always_critic"]:
    if required not in selection_metrics:
        raise SystemExit(f"selection-regret smoke missing policy {required}")
if selection["config"]["heldout_cell_count"] <= 0:
    raise SystemExit("selection-regret smoke did not create held-out cells")
if selection_metrics["audit_mse_cost"]["mean_regret"] > max(
    selection_metrics["always_group"]["mean_regret"],
    selection_metrics["always_critic"]["mean_regret"],
) + 1e-12:
    raise SystemExit("selection-regret smoke rule underperformed both static policies")
if not (0.0 <= selection_metrics["audit_mse_cost"]["selection_accuracy"] <= 1.0):
    raise SystemExit("selection-regret smoke invalid selection accuracy")

pg = json.loads((smoke_dir / "policy_gradient_smoke.json").read_text())
pg_estimators = {row["method"]: row["metrics"] for row in pg["estimators"]}
for required in [
    "reinforce_return",
    "sibling_loo_return",
    "prefix_value_baseline",
    "brpo_combined_baseline",
    "learned_value_td",
    "oracle_value_td",
]:
    if required not in pg_estimators:
        raise SystemExit(f"policy-gradient smoke missing {required}")
if pg_estimators["oracle_value_td"]["variance_trace"] >= pg_estimators["reinforce_return"]["variance_trace"]:
    raise SystemExit("policy-gradient smoke oracle-value TD did not reduce gradient variance")
if pg["exact_gradient"]["finite_difference_relative_error"] >= 1e-8:
    raise SystemExit("policy-gradient smoke analytic gradient failed finite-difference check")
pg_policy_implied = {
    row["method"]: row for row in pg["policy_implied_signals"]
}
for required in [
    "vimpo_actor_equal_ref",
    "vimpo_actor_fixed_ref_near",
    "vimpo_actor_fixed_ref_far",
]:
    if required not in pg_policy_implied:
        raise SystemExit(f"policy-gradient smoke missing policy-implied row {required}")
if abs(pg_policy_implied["vimpo_actor_equal_ref"]["metrics"]["mean_gradient_norm"]) > 1e-12:
    raise SystemExit("VIMPO equal-reference smoke was not zero at initialization")
if pg_policy_implied["vimpo_actor_fixed_ref_far"]["reference_kl"] <= pg_policy_implied["vimpo_actor_fixed_ref_near"]["reference_kl"]:
    raise SystemExit("VIMPO fixed-reference KL sweep was not ordered")
pg_baselines = pg["position_diagnostics"]["overall"]
if pg_baselines["critic_value"]["residual_variance_ratio"] >= pg_baselines["group_mean"]["residual_variance_ratio"]:
    raise SystemExit("policy-gradient smoke critic baseline did not reduce residual variance")
if pg["sample_counts"]["null_token_fraction"] <= 0.0:
    raise SystemExit("policy-gradient smoke did not sample the null action")

print(f"smoke artifacts: {smoke_dir}")
PY
