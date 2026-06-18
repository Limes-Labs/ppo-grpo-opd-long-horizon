"""Length-imbalance audit for trajectory-level credit estimators.

This experiment asks a narrower question than the deep matrix:

When sibling rollouts for the same prompt have increasingly different lengths,
does a trajectory-level group scalar recover token-level credit, or does it
mainly remain a rollout-ranking signal?

The audit compares three critic-free group scalars with the same critic TD
estimator used elsewhere:

1. group total return: z-score total return inside the prompt group;
2. group terminal reward: z-score only the terminal verifier result;
3. group return per token: z-score total return divided by trajectory length.

The length-adjusted scalar is included as a conservative baseline. It can
reduce one length confound, but it is still broadcast to every token in a
trajectory and therefore cannot assign different credit to wait/help/harm steps.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

from experiments.deep_matrix import DEFAULT_SEEDS
from experiments.toy_credit_assignment import (
    SCENARIOS,
    StepRecord,
    TabularCritic,
    Trajectory,
    add_critic_advantages,
    add_group_relative_advantages,
    estimator_metrics,
    flatten,
    generate_groups,
    pearson,
    resolve_scenario,
)


DEFAULT_HORIZONS = [4, 8, 12, 16, 20]


def zscore(values: list[float]) -> list[float]:
    mean_value = fmean(values)
    variance = fmean((value - mean_value) ** 2 for value in values)
    stddev = math.sqrt(variance)
    if stddev <= 1e-12:
        return [0.0 for _ in values]
    return [(value - mean_value) / stddev for value in values]


def build_group_scalar(
    groups: list[list[Trajectory]],
    value_fn: Callable[[Trajectory], float],
) -> dict[int, float]:
    estimates: dict[int, float] = {}
    for group in groups:
        normalized = zscore([value_fn(trajectory) for trajectory in group])
        for trajectory, value in zip(group, normalized):
            estimates[trajectory.trajectory_id] = value
    return estimates


def length_diagnostics(groups: list[list[Trajectory]]) -> dict[str, float]:
    ranges: list[float] = []
    stddevs: list[float] = []
    length_return_corrs: list[float] = []
    all_lengths: list[float] = []
    all_returns: list[float] = []

    for group in groups:
        lengths = [float(trajectory.length) for trajectory in group]
        returns = [trajectory.total_return for trajectory in group]
        mean_length = fmean(lengths)
        ranges.append(max(lengths) - min(lengths))
        stddevs.append(
            math.sqrt(fmean((length - mean_length) ** 2 for length in lengths))
        )
        length_return_corrs.append(pearson(lengths, returns))
        all_lengths.extend(lengths)
        all_returns.extend(returns)

    return {
        "mean_group_length_range": fmean(ranges) if ranges else 0.0,
        "mean_group_length_stddev": fmean(stddevs) if stddevs else 0.0,
        "mean_group_length_return_correlation": (
            fmean(length_return_corrs) if length_return_corrs else 0.0
        ),
        "global_length_return_correlation": pearson(all_lengths, all_returns),
    }


def metric_entry(
    name: str,
    label: str,
    trajectories: list[Trajectory],
    getter: Callable[[StepRecord], float],
) -> dict[str, Any]:
    return {
        "name": name,
        "label": label,
        "metrics": estimator_metrics(trajectories, getter),
    }


def run_one(
    *,
    seed: int,
    max_steps: int,
    scenario_name: str,
    train_groups: int,
    eval_groups: int,
    group_size: int,
) -> dict[str, Any]:
    scenario = resolve_scenario(scenario_name)
    if scenario.min_steps > max_steps:
        raise ValueError("scenario min_steps must be <= max_steps")

    train_rng = random.Random(seed)
    train_groups_data = generate_groups(
        train_rng,
        group_count=train_groups,
        group_size=group_size,
        max_steps=max_steps,
        scenario=scenario,
    )
    train_trajectories = flatten(train_groups_data)
    critic = TabularCritic(train_trajectories, observation=scenario.critic_observation)

    eval_rng = random.Random(seed + 1)
    eval_groups_data = generate_groups(
        eval_rng,
        group_count=eval_groups,
        group_size=group_size,
        max_steps=max_steps,
        scenario=scenario,
        trajectory_offset=len(train_trajectories),
    )
    add_group_relative_advantages(eval_groups_data)
    eval_trajectories = flatten(eval_groups_data)
    add_critic_advantages(eval_trajectories, critic)

    terminal_group = build_group_scalar(
        eval_groups_data,
        lambda trajectory: trajectory.terminal_reward,
    )
    per_token_group = build_group_scalar(
        eval_groups_data,
        lambda trajectory: trajectory.total_return / trajectory.length,
    )

    steps = [step for trajectory in eval_trajectories for step in trajectory.steps]
    diagnostics = length_diagnostics(eval_groups_data)
    diagnostics.update(
        {
            "wait_token_fraction": (
                sum(1 for step in steps if step.action == "wait") / len(steps)
                if steps
                else 0.0
            ),
            "success_rate": (
                fmean(trajectory.terminal_reward for trajectory in eval_trajectories)
                if eval_trajectories
                else 0.0
            ),
            "critic_exact_state_rate": (
                sum(
                    1
                    for step in steps
                    if critic.has_exact_state(
                        step.threshold,
                        step.start_score,
                        step.remaining_before,
                    )
                )
                / len(steps)
                if steps
                else 0.0
            ),
        }
    )

    estimators = [
        metric_entry(
            "group_total_return",
            "Group total return",
            eval_trajectories,
            lambda step: step.group_advantage,
        ),
        metric_entry(
            "group_terminal_reward",
            "Group terminal reward",
            eval_trajectories,
            lambda step: terminal_group[step.trajectory_id],
        ),
        metric_entry(
            "group_return_per_token",
            "Group return per token",
            eval_trajectories,
            lambda step: per_token_group[step.trajectory_id],
        ),
        metric_entry(
            "critic_td",
            "Critic TD",
            eval_trajectories,
            lambda step: step.critic_advantage,
        ),
    ]

    return {
        "seed": seed,
        "max_steps": max_steps,
        "diagnostics": diagnostics,
        "estimators": estimators,
    }


def mean_metric(rows: list[dict[str, Any]], estimator_name: str, metric: str) -> float:
    return fmean(
        next(
            estimator
            for estimator in row["estimators"]
            if estimator["name"] == estimator_name
        )["metrics"][metric]
        for row in rows
    )


def aggregate_horizon(max_steps: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    group_r = mean_metric(rows, "group_total_return", "pearson_correlation")
    terminal_r = mean_metric(rows, "group_terminal_reward", "pearson_correlation")
    per_token_r = mean_metric(rows, "group_return_per_token", "pearson_correlation")
    critic_r = mean_metric(rows, "critic_td", "pearson_correlation")
    return {
        "max_steps": max_steps,
        "seed_count": len(rows),
        "mean_group_length_range": fmean(
            row["diagnostics"]["mean_group_length_range"] for row in rows
        ),
        "mean_group_length_stddev": fmean(
            row["diagnostics"]["mean_group_length_stddev"] for row in rows
        ),
        "mean_wait_token_fraction": fmean(
            row["diagnostics"]["wait_token_fraction"] for row in rows
        ),
        "mean_success_rate": fmean(row["diagnostics"]["success_rate"] for row in rows),
        "group_total_r": group_r,
        "group_terminal_r": terminal_r,
        "group_per_token_r": per_token_r,
        "critic_r": critic_r,
        "critic_minus_group_total_r": critic_r - group_r,
        "critic_minus_group_per_token_r": critic_r - per_token_r,
        "group_total_wait_leak": mean_metric(
            rows,
            "group_total_return",
            "wait_to_active_abs_ratio",
        ),
        "group_per_token_wait_leak": mean_metric(
            rows,
            "group_return_per_token",
            "wait_to_active_abs_ratio",
        ),
        "critic_wait_leak": mean_metric(rows, "critic_td", "wait_to_active_abs_ratio"),
        "group_total_within_var": mean_metric(
            rows,
            "group_total_return",
            "within_trajectory_variance",
        ),
        "group_per_token_within_var": mean_metric(
            rows,
            "group_return_per_token",
            "within_trajectory_variance",
        ),
        "critic_within_var": mean_metric(rows, "critic_td", "within_trajectory_variance"),
    }


def run_audit(
    *,
    seeds: list[int] | None = None,
    horizons: list[int] | None = None,
    scenario_name: str = "long_wait",
    train_groups: int = 140,
    eval_groups: int = 36,
    group_size: int = 6,
) -> dict[str, Any]:
    seeds = list(DEFAULT_SEEDS if seeds is None else seeds)
    horizons = list(DEFAULT_HORIZONS if horizons is None else horizons)
    if not seeds or not horizons:
        raise ValueError("expected at least one seed and one horizon")
    if train_groups <= 0 or eval_groups <= 0 or group_size <= 1:
        raise ValueError("expected train/eval groups > 0 and group_size > 1")

    scenario = resolve_scenario(scenario_name)
    if any(horizon < scenario.min_steps for horizon in horizons):
        raise ValueError("all horizons must be >= scenario min_steps")

    seed_results: list[dict[str, Any]] = []
    for max_steps in horizons:
        for seed in seeds:
            seed_results.append(
                run_one(
                    seed=seed,
                    max_steps=max_steps,
                    scenario_name=scenario_name,
                    train_groups=train_groups,
                    eval_groups=eval_groups,
                    group_size=group_size,
                )
            )

    by_horizon = {
        max_steps: [row for row in seed_results if row["max_steps"] == max_steps]
        for max_steps in horizons
    }
    horizon_summaries = [
        aggregate_horizon(max_steps, by_horizon[max_steps]) for max_steps in horizons
    ]
    first = horizon_summaries[0]
    last = horizon_summaries[-1]
    return {
        "config": {
            "seeds": seeds,
            "horizons": horizons,
            "scenario_name": scenario.name,
            "scenario_description": scenario.description,
            "train_groups": train_groups,
            "eval_groups": eval_groups,
            "group_size": group_size,
        },
        "horizon_summaries": horizon_summaries,
        "seed_results": seed_results,
        "summary": {
            "critic_wins_vs_group_total": sum(
                1
                for row in horizon_summaries
                if row["critic_minus_group_total_r"] > 0
            ),
            "critic_wins_vs_group_per_token": sum(
                1
                for row in horizon_summaries
                if row["critic_minus_group_per_token_r"] > 0
            ),
            "delta_growth_short_to_long": (
                last["critic_minus_group_total_r"]
                - first["critic_minus_group_total_r"]
            ),
            "length_range_growth_short_to_long": (
                last["mean_group_length_range"] - first["mean_group_length_range"]
            ),
            "longest_horizon_critic_minus_group_total_r": last[
                "critic_minus_group_total_r"
            ],
            "longest_horizon_critic_minus_group_per_token_r": last[
                "critic_minus_group_per_token_r"
            ],
        },
    }


def fmt(value: float) -> str:
    return f"{value:.3f}"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Length Imbalance Audit",
        "",
        "This table audits whether simple length correction rescues a",
        "trajectory-level group scalar as rollouts become longer and more",
        "imbalanced. `Per-token r` is group-normalized total return divided by",
        "trajectory length; it is still broadcast to every token.",
        "",
        "| Max steps | Len range | Wait frac | Group r | Per-token r | Critic r | Critic - group | Critic - per-token |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["horizon_summaries"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["max_steps"]),
                    fmt(row["mean_group_length_range"]),
                    fmt(row["mean_wait_token_fraction"]),
                    fmt(row["group_total_r"]),
                    fmt(row["group_per_token_r"]),
                    fmt(row["critic_r"]),
                    fmt(row["critic_minus_group_total_r"]),
                    fmt(row["critic_minus_group_per_token_r"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Summary:",
            f"- Critic wins vs group total: {result['summary']['critic_wins_vs_group_total']} / {len(result['horizon_summaries'])}",
            f"- Critic wins vs group per-token: {result['summary']['critic_wins_vs_group_per_token']} / {len(result['horizon_summaries'])}",
            f"- Delta growth, shortest to longest horizon: {fmt(result['summary']['delta_growth_short_to_long'])}",
            f"- Length-range growth, shortest to longest horizon: {fmt(result['summary']['length_range_growth_short_to_long'])}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--horizons", type=int, nargs="*", default=None)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="long_wait")
    parser.add_argument("--train-groups", type=int, default=140)
    parser.add_argument("--eval-groups", type=int, default=36)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/length_imbalance_audit_seedset.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/length_imbalance_audit_seedset.md"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_audit(
        seeds=args.seeds,
        horizons=args.horizons,
        scenario_name=args.scenario,
        train_groups=args.train_groups,
        eval_groups=args.eval_groups,
        group_size=args.group_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    write_markdown(result, args.output_md)

    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    print(
        "wins: "
        f"critic_vs_group={result['summary']['critic_wins_vs_group_total']} "
        f"critic_vs_per_token={result['summary']['critic_wins_vs_group_per_token']}"
    )
    print(
        "longest_delta: "
        f"group={result['summary']['longest_horizon_critic_minus_group_total_r']:.3f} "
        f"per_token={result['summary']['longest_horizon_critic_minus_group_per_token_r']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
