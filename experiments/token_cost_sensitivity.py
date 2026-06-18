"""Token-cost sensitivity audit for the toy credit-assignment experiment.

The main toy uses a small per-token cost. A skeptical reader could ask whether
critic TD wins because it sees the explicit length penalty rather than because
it recovers temporal credit. This audit reruns the baseline and long-wait
settings at several token costs, including zero.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from experiments.deep_matrix import DEFAULT_SEEDS
from experiments.toy_credit_assignment import SCENARIOS, resolve_scenario, run_experiment


DEFAULT_SCENARIOS = ["baseline", "long_wait"]
DEFAULT_TOKEN_COSTS = [0.0, 0.01, 0.02, 0.05]


def ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return 1.96 * stdev(values) / (len(values) ** 0.5)


def run_one(
    *,
    seed: int,
    scenario_name: str,
    token_cost: float,
    train_groups: int,
    eval_groups: int,
    group_size: int,
    max_steps: int,
) -> dict[str, Any]:
    base = resolve_scenario(scenario_name)
    scenario = replace(
        base,
        name=f"{base.name}_cost_{token_cost:g}",
        token_cost=token_cost,
    )
    result = run_experiment(
        seed=seed,
        scenario=scenario,
        train_groups=train_groups,
        eval_groups=eval_groups,
        group_size=group_size,
        max_steps=max_steps,
    )
    group = result["metrics"]["group_relative"]
    critic = result["metrics"]["critic_value_model"]
    return {
        "seed": seed,
        "scenario": scenario_name,
        "token_cost": token_cost,
        "group_r": group["pearson_correlation"],
        "critic_r": critic["pearson_correlation"],
        "delta_r": critic["pearson_correlation"] - group["pearson_correlation"],
        "group_mse": group["calibrated_mse"],
        "critic_mse": critic["calibrated_mse"],
        "group_wait_leak": group["wait_to_active_abs_ratio"],
        "critic_wait_leak": critic["wait_to_active_abs_ratio"],
        "group_within_var": group["within_trajectory_variance"],
        "critic_within_var": critic["within_trajectory_variance"],
        "success_rate": result["sample_counts"]["success_rate"],
        "wait_token_fraction": result["sample_counts"]["wait_token_fraction"],
        "zero_variance_group_fraction": result["sample_counts"][
            "zero_variance_group_fraction"
        ],
        "critic_exact_state_rate": result["sample_counts"]["critic_exact_state_rate"],
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [row["delta_r"] for row in rows]
    return {
        "scenario": rows[0]["scenario"],
        "token_cost": rows[0]["token_cost"],
        "seed_count": len(rows),
        "group_r": fmean(row["group_r"] for row in rows),
        "critic_r": fmean(row["critic_r"] for row in rows),
        "delta_r": fmean(deltas),
        "delta_ci95": ci95(deltas),
        "group_mse": fmean(row["group_mse"] for row in rows),
        "critic_mse": fmean(row["critic_mse"] for row in rows),
        "group_wait_leak": fmean(row["group_wait_leak"] for row in rows),
        "critic_wait_leak": fmean(row["critic_wait_leak"] for row in rows),
        "group_within_var": fmean(row["group_within_var"] for row in rows),
        "critic_within_var": fmean(row["critic_within_var"] for row in rows),
        "success_rate": fmean(row["success_rate"] for row in rows),
        "wait_token_fraction": fmean(row["wait_token_fraction"] for row in rows),
        "zero_variance_group_fraction": fmean(
            row["zero_variance_group_fraction"] for row in rows
        ),
        "critic_exact_state_rate": fmean(row["critic_exact_state_rate"] for row in rows),
    }


def run_sensitivity(
    *,
    seeds: list[int] | None = None,
    scenarios: list[str] | None = None,
    token_costs: list[float] | None = None,
    train_groups: int = 140,
    eval_groups: int = 32,
    group_size: int = 6,
    max_steps: int = 12,
) -> dict[str, Any]:
    seeds = list(DEFAULT_SEEDS if seeds is None else seeds)
    scenarios = list(DEFAULT_SCENARIOS if scenarios is None else scenarios)
    token_costs = list(DEFAULT_TOKEN_COSTS if token_costs is None else token_costs)
    if not seeds or not scenarios or not token_costs:
        raise ValueError("expected at least one seed, scenario, and token cost")
    if any(cost < 0.0 for cost in token_costs):
        raise ValueError("token costs must be non-negative")
    for scenario in scenarios:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}")

    seed_results: list[dict[str, Any]] = []
    for scenario in scenarios:
        for token_cost in token_costs:
            for seed in seeds:
                seed_results.append(
                    run_one(
                        seed=seed,
                        scenario_name=scenario,
                        token_cost=token_cost,
                        train_groups=train_groups,
                        eval_groups=eval_groups,
                        group_size=group_size,
                        max_steps=max_steps,
                    )
                )

    aggregate_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for token_cost in token_costs:
            rows = [
                row
                for row in seed_results
                if row["scenario"] == scenario and row["token_cost"] == token_cost
            ]
            aggregate_rows.append(aggregate(rows))

    clear_positive = sum(
        1 for row in aggregate_rows if row["delta_r"] - row["delta_ci95"] > 0
    )
    long_wait_zero_cost = next(
        (
            row["delta_r"]
            for row in aggregate_rows
            if row["scenario"] == "long_wait" and row["token_cost"] == 0.0
        ),
        None,
    )
    return {
        "config": {
            "seeds": seeds,
            "scenarios": scenarios,
            "token_costs": token_costs,
            "train_groups": train_groups,
            "eval_groups": eval_groups,
            "group_size": group_size,
            "max_steps": max_steps,
        },
        "aggregate_rows": aggregate_rows,
        "seed_results": seed_results,
        "summary": {
            "clear_positive_rows": clear_positive,
            "row_count": len(aggregate_rows),
            "min_delta_minus_ci95": min(
                row["delta_r"] - row["delta_ci95"] for row in aggregate_rows
            ),
            "long_wait_zero_cost_delta": long_wait_zero_cost,
        },
    }


def fmt(value: float) -> str:
    return f"{value:.3f}"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Token-Cost Sensitivity",
        "",
        "This table checks whether critic TD still improves oracle-credit",
        "alignment when the explicit per-token cost is removed.",
        "",
        "| Scenario | Cost | Group r | Critic r | Delta | 95% CI | Group wait | Critic wait |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["aggregate_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["scenario"],
                    f"{row['token_cost']:.2f}",
                    fmt(row["group_r"]),
                    fmt(row["critic_r"]),
                    fmt(row["delta_r"]),
                    fmt(row["delta_ci95"]),
                    fmt(row["group_wait_leak"]),
                    fmt(row["critic_wait_leak"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Summary:",
            f"- Clear positive rows: {result['summary']['clear_positive_rows']} / {result['summary']['row_count']}",
            f"- Minimum delta minus CI95: {fmt(result['summary']['min_delta_minus_ci95'])}",
            "- Long-wait zero-cost delta: "
            + (
                fmt(result["summary"]["long_wait_zero_cost_delta"])
                if result["summary"]["long_wait_zero_cost_delta"] is not None
                else "not run"
            ),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--scenarios", choices=sorted(SCENARIOS), nargs="*", default=None)
    parser.add_argument("--token-costs", type=float, nargs="*", default=None)
    parser.add_argument("--train-groups", type=int, default=140)
    parser.add_argument("--eval-groups", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/token_cost_sensitivity_20seed.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/token_cost_sensitivity_20seed.md"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_sensitivity(
        seeds=args.seeds,
        scenarios=args.scenarios,
        token_costs=args.token_costs,
        train_groups=args.train_groups,
        eval_groups=args.eval_groups,
        group_size=args.group_size,
        max_steps=args.max_steps,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    write_markdown(result, args.output_md)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    print(
        "clear_positive: "
        f"{result['summary']['clear_positive_rows']}/{result['summary']['row_count']}"
    )
    print(
        "long_wait_zero_cost_delta: "
        + (
            f"{result['summary']['long_wait_zero_cost_delta']:.3f}"
            if result["summary"]["long_wait_zero_cost_delta"] is not None
            else "not run"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
