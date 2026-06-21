"""Phase-diagram diagnostic for trajectory broadcast versus critic credit.

The deep matrix shows that critic-style TD advantages usually align better with
the exact behavior-policy advantage in this toy. This module asks a more useful
boundary question: when should a trajectory-constant estimator be expected to
hit an information ceiling, and when is the critic too unreliable to exploit
that ceiling?

It remains a finite-MDP, CPU-only diagnostic. The target stored in
``StepRecord.oracle_advantage`` is the exact behavior-policy advantage under
the toy rollout policy, not an optimal-control advantage.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import replace
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Callable, Iterable

from experiments.toy_credit_assignment import (
    SCENARIOS,
    Scenario,
    StepRecord,
    TabularCritic,
    Trajectory,
    add_critic_advantages,
    add_group_relative_advantages,
    estimator_metrics,
    flatten,
    generate_groups,
    observation_schema,
    critic_is_privileged,
    resolve_scenario,
)


DEFAULT_SEEDS = [11, 29, 47, 83, 131]

HETEROGENEITY_REGIMES: dict[str, dict[str, Any]] = {
    "h005": {
        "target": 0.05,
        "base": "short_dense",
        "min_steps": 1,
        "max_steps": 1,
        "wait_bias_after_success": -0.45,
        "help_bias_when_behind": 0.35,
        "threshold_cycle": (1,),
    },
    "h015": {
        "target": 0.15,
        "base": "short_dense",
        "min_steps": 1,
        "max_steps": 2,
        "wait_bias_after_success": 0.00,
        "help_bias_when_behind": 0.20,
        "threshold_cycle": (1,),
    },
    "h030": {
        "target": 0.30,
        "base": "short_dense",
        "min_steps": 1,
        "max_steps": 2,
        "wait_bias_after_success": 0.00,
        "help_bias_when_behind": -0.10,
        "threshold_cycle": (1,),
    },
    "h050": {
        "target": 0.50,
        "base": "short_dense",
        "min_steps": 2,
        "max_steps": 2,
        "wait_bias_after_success": 0.20,
        "help_bias_when_behind": 0.10,
        "threshold_cycle": (1, 2),
    },
    "h070": {
        "target": 0.70,
        "base": "short_dense",
        "min_steps": 1,
        "max_steps": 7,
        "wait_bias_after_success": 0.40,
        "help_bias_when_behind": -0.10,
        "threshold_cycle": (1, 2, 3),
    },
    "h090": {
        "target": 0.90,
        "base": "long_wait",
        "min_steps": 3,
        "max_steps": 12,
        "wait_bias_after_success": 0.45,
        "help_bias_when_behind": -0.08,
        "threshold_cycle": (2, 3, 4),
    },
}


def variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_value = fmean(values)
    return fmean((value - mean_value) ** 2 for value in values)


def mean_ci95(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean_value = fmean(values)
    if len(values) == 1:
        return mean_value, 0.0
    return mean_value, 1.96 * stdev(values) / math.sqrt(len(values))


def evidence_from_delta(delta: float, ci95: float, practical_delta: float = 0.03) -> str:
    if delta - ci95 > practical_delta:
        return "critic_clear"
    if delta + ci95 < -practical_delta:
        return "group_clear"
    return "near_tie"


def credit_heterogeneity(
    trajectories: Iterable[Trajectory],
    getter: Callable[[StepRecord], float] = lambda step: step.oracle_advantage,
) -> dict[str, float]:
    """Compute the trajectory-broadcast ceiling for step advantages.

    Tokens are micro-averaged: sampling first picks a token uniformly from all
    tokens in the evaluation batch. The within-trajectory variance term is
    therefore weighted by trajectory length.
    """

    trajectories = list(trajectories)
    steps = [step for trajectory in trajectories for step in trajectory.steps]
    values = [getter(step) for step in steps]
    total_var = variance(values)
    token_count = len(values)

    if token_count == 0:
        return {
            "trajectory_count": float(len(trajectories)),
            "token_count": 0.0,
            "total_variance": 0.0,
            "within_trajectory_variance": 0.0,
            "between_trajectory_variance": 0.0,
            "credit_heterogeneity": 0.0,
            "broadcast_ceiling_correlation": 0.0,
        }

    weighted_within = 0.0
    for trajectory in trajectories:
        trajectory_values = [getter(step) for step in trajectory.steps]
        if not trajectory_values:
            continue
        weighted_within += (len(trajectory_values) / token_count) * variance(
            trajectory_values
        )

    if total_var <= 1e-15:
        heterogeneity = 0.0
        ceiling = 1.0
        between = 0.0
    else:
        heterogeneity = max(0.0, min(1.0, weighted_within / total_var))
        between = max(0.0, total_var - weighted_within)
        ceiling = math.sqrt(max(0.0, 1.0 - heterogeneity))

    return {
        "trajectory_count": float(len(trajectories)),
        "token_count": float(token_count),
        "total_variance": total_var,
        "within_trajectory_variance": weighted_within,
        "between_trajectory_variance": between,
        "credit_heterogeneity": heterogeneity,
        "broadcast_ceiling_correlation": ceiling,
    }


def scenario_for_cell(
    *,
    heterogeneity: str,
    observability: str,
    reward: str,
    drift: str,
    phase: str,
) -> tuple[Scenario, int]:
    if heterogeneity in HETEROGENEITY_REGIMES:
        spec = HETEROGENEITY_REGIMES[heterogeneity]
        base = resolve_scenario(str(spec["base"]))
        scenario = replace(
            base,
            name=f"phase_{heterogeneity}",
            description=(
                "Calibrated heterogeneity-regime scenario for the broadcast "
                f"ceiling audit (target H_credit {spec['target']})."
            ),
            wait_bias_after_success=float(spec["wait_bias_after_success"]),
            help_bias_when_behind=float(spec["help_bias_when_behind"]),
            threshold_cycle=tuple(spec["threshold_cycle"]),
            min_steps=int(spec["min_steps"]),
        )
        max_steps = int(spec["max_steps"])
    elif heterogeneity == "low":
        base = resolve_scenario("short_dense")
        scenario = replace(
            base,
            name="phase_low",
            description="Short, low-padding traces for the broadcast ceiling audit.",
            wait_bias_after_success=-0.22,
            help_bias_when_behind=0.12,
            threshold_cycle=(1, 1, 2),
            min_steps=2,
        )
        max_steps = 5
    elif heterogeneity == "high":
        base = resolve_scenario("long_wait")
        scenario = replace(
            base,
            name="phase_high",
            description="Long wait-heavy traces for the broadcast ceiling audit.",
            wait_bias_after_success=0.30,
            threshold_cycle=(1, 2, 3),
            min_steps=3,
        )
        max_steps = 14
    else:
        raise ValueError(f"unknown heterogeneity level {heterogeneity!r}")

    if reward == "contrast":
        scenario = replace(scenario, threshold_cycle=scenario.threshold_cycle)
    elif reward == "sparse":
        scenario = replace(
            scenario,
            name=f"{scenario.name}_sparse",
            threshold_cycle=(3, 4, 5),
            help_bias_when_behind=scenario.help_bias_when_behind - 0.08,
        )
    else:
        raise ValueError(f"unknown reward level {reward!r}")

    if observability not in {"full", "non_privileged", "coarse", "blind"}:
        raise ValueError(f"unknown observability level {observability!r}")
    scenario = replace(scenario, critic_observation=observability)

    if drift == "matched":
        return scenario, max_steps
    if drift != "stale":
        raise ValueError(f"unknown drift level {drift!r}")

    if phase == "train":
        train = replace(
            scenario,
            name=f"{scenario.name}_train_stale",
            wait_bias_after_success=max(-0.24, scenario.wait_bias_after_success - 0.28),
            help_bias_when_behind=scenario.help_bias_when_behind + 0.10,
            harm_bias=scenario.harm_bias - 0.04,
        )
        return train, max_steps
    return scenario, max_steps


def coverage_to_train_trajectories(level: str) -> int:
    if level == "low":
        return 5
    if level == "high":
        return 600
    raise ValueError(f"unknown coverage level {level!r}")


def classify_mechanism(credit_h: float, critic_r: float) -> str:
    high_credit = credit_h >= 0.45
    reliable_critic = critic_r >= 0.60
    if not high_credit and not reliable_critic:
        return "group_or_global"
    if not high_credit and reliable_critic:
        return "either_by_cost"
    if high_credit and reliable_critic:
        return "critic_or_sampled_value"
    return "process_structural_or_hybrid"


def run_cell(
    *,
    seed: int,
    heterogeneity: str,
    observability: str,
    coverage: str,
    reward: str,
    drift: str,
    eval_groups: int,
    group_size: int,
) -> dict[str, Any]:
    train_scenario, train_max_steps = scenario_for_cell(
        heterogeneity=heterogeneity,
        observability=observability,
        reward=reward,
        drift=drift,
        phase="train",
    )
    eval_scenario, eval_max_steps = scenario_for_cell(
        heterogeneity=heterogeneity,
        observability=observability,
        reward=reward,
        drift=drift,
        phase="eval",
    )
    max_steps = max(train_max_steps, eval_max_steps)
    train_trajectories_budget = coverage_to_train_trajectories(coverage)

    train_rng = random.Random(seed)
    train_groups_data = generate_groups(
        train_rng,
        group_count=train_trajectories_budget,
        group_size=1,
        max_steps=max_steps,
        scenario=train_scenario,
    )
    train_trajectories = flatten(train_groups_data)
    critic = TabularCritic(
        train_trajectories,
        observation=eval_scenario.critic_observation,
    )

    eval_rng = random.Random(seed + 1)
    eval_groups_data = generate_groups(
        eval_rng,
        group_count=eval_groups,
        group_size=group_size,
        max_steps=eval_max_steps,
        scenario=eval_scenario,
        trajectory_offset=len(train_trajectories),
    )
    add_group_relative_advantages(eval_groups_data)
    eval_trajectories = flatten(eval_groups_data)
    add_critic_advantages(eval_trajectories, critic)

    steps = [step for trajectory in eval_trajectories for step in trajectory.steps]
    group = estimator_metrics(eval_trajectories, lambda step: step.group_advantage)
    critic_metrics = estimator_metrics(
        eval_trajectories,
        lambda step: step.critic_advantage,
    )
    heterogeneity_metrics = credit_heterogeneity(eval_trajectories)
    hit_rate = (
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
    )
    return {
        "seed": seed,
        "train_trajectories": len(train_trajectories),
        "eval_tokens": len(steps),
        "actor_observation_schema": observation_schema("actor"),
        "critic_observation_schema": observation_schema(eval_scenario.critic_observation),
        "critic_is_privileged": critic_is_privileged(eval_scenario.critic_observation),
        "group_correlation": group["pearson_correlation"],
        "critic_correlation": critic_metrics["pearson_correlation"],
        "critic_minus_group_correlation": (
            critic_metrics["pearson_correlation"] - group["pearson_correlation"]
        ),
        "group_calibrated_mse": group["calibrated_mse"],
        "critic_calibrated_mse": critic_metrics["calibrated_mse"],
        "critic_exact_state_rate": hit_rate,
        "wait_token_fraction": (
            sum(1 for step in steps if step.action == "wait") / len(steps)
            if steps
            else 0.0
        ),
        "success_rate": (
            sum(trajectory.terminal_reward for trajectory in eval_trajectories)
            / len(eval_trajectories)
            if eval_trajectories
            else 0.0
        ),
        **heterogeneity_metrics,
    }


def aggregate_cell(
    *,
    heterogeneity: str,
    observability: str,
    coverage: str,
    reward: str,
    drift: str,
    seed_results: list[dict[str, Any]],
) -> dict[str, Any]:
    deltas = [row["critic_minus_group_correlation"] for row in seed_results]
    delta, ci95 = mean_ci95(deltas)
    group_r = fmean(row["group_correlation"] for row in seed_results)
    critic_r = fmean(row["critic_correlation"] for row in seed_results)
    credit_h = fmean(row["credit_heterogeneity"] for row in seed_results)
    ceiling = fmean(row["broadcast_ceiling_correlation"] for row in seed_results)
    evidence = evidence_from_delta(delta, ci95)
    return {
        "cell_name": "_".join([heterogeneity, observability, coverage, reward, drift]),
        "heterogeneity": heterogeneity,
        "target_credit_heterogeneity": HETEROGENEITY_REGIMES.get(
            heterogeneity,
            {"target": None},
        )["target"],
        "observability": observability,
        "coverage": coverage,
        "reward": reward,
        "drift": drift,
        "seed_results": seed_results,
        "credit_heterogeneity": credit_h,
        "broadcast_ceiling_correlation": ceiling,
        "group_correlation": group_r,
        "critic_correlation": critic_r,
        "critic_minus_group_correlation": delta,
        "ci95_critic_minus_group_correlation": ci95,
        "critic_exact_state_rate": fmean(
            row["critic_exact_state_rate"] for row in seed_results
        ),
        "critic_is_privileged": any(row["critic_is_privileged"] for row in seed_results),
        "train_trajectories": fmean(row["train_trajectories"] for row in seed_results),
        "wait_token_fraction": fmean(row["wait_token_fraction"] for row in seed_results),
        "success_rate": fmean(row["success_rate"] for row in seed_results),
        "evidence_by_ci95": evidence,
        "recommended_mechanism": classify_mechanism(credit_h, critic_r),
    }


def run_phase_diagram(
    *,
    seeds: list[int] | None = None,
    heterogeneity_levels: list[str] | None = None,
    observability_levels: list[str] | None = None,
    coverage_levels: list[str] | None = None,
    reward_levels: list[str] | None = None,
    drift_levels: list[str] | None = None,
    eval_groups: int = 24,
    group_size: int = 5,
) -> dict[str, Any]:
    seeds = seeds or DEFAULT_SEEDS
    heterogeneity_levels = heterogeneity_levels or ["h005", "h015", "h030", "h050", "h070", "h090"]
    observability_levels = observability_levels or ["non_privileged", "blind"]
    coverage_levels = coverage_levels or ["low", "high"]
    reward_levels = reward_levels or ["contrast", "sparse"]
    drift_levels = drift_levels or ["matched"]

    aggregate_rows: list[dict[str, Any]] = []
    for heterogeneity in heterogeneity_levels:
        for observability in observability_levels:
            for coverage in coverage_levels:
                for reward in reward_levels:
                    for drift in drift_levels:
                        seed_results = [
                            run_cell(
                                seed=seed,
                                heterogeneity=heterogeneity,
                                observability=observability,
                                coverage=coverage,
                                reward=reward,
                                drift=drift,
                                eval_groups=eval_groups,
                                group_size=group_size,
                            )
                            for seed in seeds
                        ]
                        aggregate_rows.append(
                            aggregate_cell(
                                heterogeneity=heterogeneity,
                                observability=observability,
                                coverage=coverage,
                                reward=reward,
                                drift=drift,
                                seed_results=seed_results,
                            )
                        )

    critic_clear = sum(
        1 for row in aggregate_rows if row["evidence_by_ci95"] == "critic_clear"
    )
    group_clear = sum(
        1 for row in aggregate_rows if row["evidence_by_ci95"] == "group_clear"
    )
    near_tie = sum(1 for row in aggregate_rows if row["evidence_by_ci95"] == "near_tie")
    return {
        "config": {
            "seeds": seeds,
            "heterogeneity_levels": heterogeneity_levels,
            "observability_levels": observability_levels,
            "coverage_levels": coverage_levels,
            "reward_levels": reward_levels,
            "drift_levels": drift_levels,
            "eval_groups": eval_groups,
            "group_size": group_size,
        },
        "summary": {
            "seed_count": len(seeds),
            "cell_count": len(aggregate_rows),
            "critic_clear_cells": critic_clear,
            "group_clear_cells": group_clear,
            "near_tie_cells": near_tie,
            "mean_credit_heterogeneity": fmean(
                row["credit_heterogeneity"] for row in aggregate_rows
            ),
            "mean_broadcast_ceiling_correlation": fmean(
                row["broadcast_ceiling_correlation"] for row in aggregate_rows
            ),
            "mean_critic_minus_group_correlation": fmean(
                row["critic_minus_group_correlation"] for row in aggregate_rows
            ),
            "min_credit_heterogeneity": min(
                row["credit_heterogeneity"] for row in aggregate_rows
            ),
            "max_credit_heterogeneity": max(
                row["credit_heterogeneity"] for row in aggregate_rows
            ),
        },
        "aggregate_rows": aggregate_rows,
    }


def fmt(value: float) -> str:
    return f"{value:.3f}"


def build_markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# Broadcast ceiling phase diagram",
        "",
        "This diagnostic estimates two boundary quantities in the finite toy MDP:",
        "within-trajectory credit heterogeneity and held-out critic reliability.",
        "The group estimator is trajectory-constant, so its absolute correlation",
        "with exact behavior-policy advantage should not exceed the broadcast",
        "ceiling implied by the heterogeneity index.",
        "",
        "## Summary",
        "",
        f"- Seeds: {result['summary']['seed_count']}",
        f"- Cells: {result['summary']['cell_count']}",
        f"- H_credit range: {fmt(result['summary']['min_credit_heterogeneity'])} to {fmt(result['summary']['max_credit_heterogeneity'])}",
        f"- Clear critic cells: {result['summary']['critic_clear_cells']}",
        f"- Clear group cells: {result['summary']['group_clear_cells']}",
        f"- Near ties: {result['summary']['near_tie_cells']}",
        "",
        "| Cell | Target H | H_credit | Ceiling | Group r | Critic r | Delta r | CI | Read | Mechanism |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in result["aggregate_rows"]:
        lines.append(
            "| {cell} | {target} | {h} | {ceil} | {group} | {critic} | {delta} | +/- {ci} | {read} | {mech} |".format(
                cell=row["cell_name"],
                target=(
                    "na"
                    if row["target_credit_heterogeneity"] is None
                    else fmt(row["target_credit_heterogeneity"])
                ),
                h=fmt(row["credit_heterogeneity"]),
                ceil=fmt(row["broadcast_ceiling_correlation"]),
                group=fmt(row["group_correlation"]),
                critic=fmt(row["critic_correlation"]),
                delta=fmt(row["critic_minus_group_correlation"]),
                ci=fmt(row["ci95_critic_minus_group_correlation"]),
                read=row["evidence_by_ci95"],
                mech=row["recommended_mechanism"],
            )
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- High credit heterogeneity lowers the best possible token-level fidelity",
            "  of any trajectory-constant estimator.",
            "- A critic crosses that ceiling only when its held-out TD signal is",
            "  reliable under the current observation and coverage regime.",
            "- When heterogeneity is high and critic reliability is low, the diagnostic",
            "  points to process rewards, structural anchors, or adaptive hybrids.",
        ]
    )
    return "\n".join(lines) + "\n"


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def int_comma_list(value: str) -> list[int]:
    return [int(item) for item in comma_list(value)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--heterogeneity-levels", default="h005,h015,h030,h050,h070,h090")
    parser.add_argument("--observability-levels", default="non_privileged,blind")
    parser.add_argument("--coverage-levels", default="low,high")
    parser.add_argument("--reward-levels", default="contrast,sparse")
    parser.add_argument("--drift-levels", default="matched")
    parser.add_argument("--eval-groups", type=int, default=24)
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/credit_phase_diagram_seedset.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/credit_phase_diagram_seedset.md"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_phase_diagram(
        seeds=int_comma_list(args.seeds),
        heterogeneity_levels=comma_list(args.heterogeneity_levels),
        observability_levels=comma_list(args.observability_levels),
        coverage_levels=comma_list(args.coverage_levels),
        reward_levels=comma_list(args.reward_levels),
        drift_levels=comma_list(args.drift_levels),
        eval_groups=args.eval_groups,
        group_size=args.group_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    args.output_md.write_text(build_markdown_report(result))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    print(
        "phase cells: "
        f"critic_clear={result['summary']['critic_clear_cells']} "
        f"group_clear={result['summary']['group_clear_cells']} "
        f"near_tie={result['summary']['near_tie_cells']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
