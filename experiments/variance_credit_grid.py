"""Variance-reduction versus credit-assignment estimator grid.

This experiment reuses the finite toy dynamics from ``toy_credit_assignment``
but compares a broader set of advantage estimators. The point is to separate two
questions that critic-based PPO often answers at once:

1. What baseline or control variate reduces policy-gradient variance?
2. Does the estimator assign different credit to different steps in a rollout?

The output is still an estimator-fidelity benchmark, not a closed-loop
PPO/GRPO training run.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

from experiments.toy_credit_assignment import (
    DEFAULT_SCENARIO,
    SCENARIOS,
    Scenario,
    StepRecord,
    TabularCritic,
    Trajectory,
    add_critic_advantages,
    add_group_relative_advantages,
    estimator_metrics,
    exact_value,
    flatten,
    generate_groups,
    next_score,
    resolve_scenario,
    sample_action,
    terminal_reward,
)


def step_key(step: StepRecord) -> tuple[int, int]:
    return step.trajectory_id, step.remaining_before


def state_seed(
    seed: int,
    threshold: int,
    score: int,
    remaining: int,
    salt: str,
) -> str:
    return f"{seed}:{threshold}:{score}:{remaining}:{salt}"


def sampled_state_value(
    *,
    seed: int,
    threshold: int,
    score: int,
    remaining: int,
    scenario: Scenario,
    branches: int,
) -> float:
    """Monte Carlo value estimate from a state with deterministic state seeding."""

    if remaining == 0:
        return terminal_reward(threshold, score)

    returns: list[float] = []
    rng = random.Random(state_seed(seed, threshold, score, remaining, "mc"))
    for _ in range(branches):
        branch_score = score
        for horizon in range(remaining, 0, -1):
            action = sample_action(
                rng,
                threshold=threshold,
                score=branch_score,
                remaining=horizon,
                scenario=scenario,
            )
            branch_score = next_score(branch_score, action)
        returns.append(terminal_reward(threshold, branch_score) - scenario.token_cost * remaining)
    return fmean(returns)


def build_sampled_mc_advantages(
    steps: list[StepRecord],
    *,
    seed: int,
    scenario: Scenario,
    branches_per_state: int,
) -> tuple[dict[tuple[int, int], float], dict[str, int]]:
    state_cache: dict[tuple[int, int, int], float] = {}
    estimates: dict[tuple[int, int], float] = {}

    def value(threshold: int, score: int, remaining: int) -> float:
        key = (threshold, score, remaining)
        if key not in state_cache:
            state_cache[key] = sampled_state_value(
                seed=seed,
                threshold=threshold,
                score=score,
                remaining=remaining,
                scenario=scenario,
                branches=branches_per_state,
            )
        return state_cache[key]

    for step in steps:
        estimates[step_key(step)] = (
            step.step_reward
            + value(step.threshold, step.next_score, step.remaining_after)
            - value(step.threshold, step.start_score, step.remaining_before)
        )

    return estimates, {
        "unique_states_sampled": len(state_cache),
        "branches_per_state": branches_per_state,
        "sampled_value_rollouts": len(state_cache) * branches_per_state,
    }


def build_leave_one_out_advantages(
    groups: list[list[Trajectory]],
) -> dict[int, float]:
    advantages: dict[int, float] = {}
    for group in groups:
        returns = [trajectory.total_return for trajectory in group]
        total = sum(returns)
        for trajectory, value in zip(group, returns):
            if len(group) <= 1:
                advantages[trajectory.trajectory_id] = 0.0
                continue
            baseline = (total - value) / (len(group) - 1)
            advantages[trajectory.trajectory_id] = value - baseline
    return advantages


def anchor_key(step: StepRecord) -> tuple[int, int, int]:
    return step.threshold, step.start_score, step.remaining_before


def build_anchor_action_contrast(
    steps: list[StepRecord],
) -> tuple[dict[tuple[int, int], float], dict[str, float]]:
    """Critic-free step credit from repeated state-action anchors.

    For a state that appears in multiple rollouts, estimate an action value by
    leave-one-out return-to-go among sibling visits to the same state and action.
    The baseline is the leave-one-out state return. Unsupported anchors return
    zero rather than borrowing a value model.
    """

    state_sums: dict[tuple[int, int, int], float] = {}
    state_counts: dict[tuple[int, int, int], int] = {}
    action_sums: dict[tuple[tuple[int, int, int], str], float] = {}
    action_counts: dict[tuple[tuple[int, int, int], str], int] = {}

    for step in steps:
        state = anchor_key(step)
        action = (state, step.action)
        state_sums[state] = state_sums.get(state, 0.0) + step.return_to_go
        state_counts[state] = state_counts.get(state, 0) + 1
        action_sums[action] = action_sums.get(action, 0.0) + step.return_to_go
        action_counts[action] = action_counts.get(action, 0) + 1

    estimates: dict[tuple[int, int], float] = {}
    supported = 0
    for step in steps:
        state = anchor_key(step)
        action = (state, step.action)
        state_count = state_counts[state]
        action_count = action_counts[action]
        if state_count > 1 and action_count > 1:
            state_value = (state_sums[state] - step.return_to_go) / (state_count - 1)
            action_value = (
                action_sums[action] - step.return_to_go
            ) / (action_count - 1)
            estimates[step_key(step)] = action_value - state_value
            supported += 1
        else:
            estimates[step_key(step)] = 0.0

    return estimates, {
        "unique_anchor_states": float(len(state_counts)),
        "unique_anchor_actions": float(len(action_counts)),
        "supported_step_fraction": supported / len(steps) if steps else 0.0,
    }


def estimate_variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_value = fmean(values)
    return fmean((value - mean_value) ** 2 for value in values)


def summarize_estimator(
    *,
    name: str,
    label: str,
    variance_reduction: str,
    credit_assignment: str,
    trajectories: list[Trajectory],
    getter: Callable[[StepRecord], float],
    cost_note: str,
) -> dict[str, Any]:
    steps = [step for trajectory in trajectories for step in trajectory.steps]
    estimates = [getter(step) for step in steps]
    metrics = estimator_metrics(trajectories, getter)
    metrics.update(
        {
            "estimate_mean": fmean(estimates) if estimates else 0.0,
            "estimate_variance": estimate_variance(estimates),
            "estimate_second_moment": fmean(value * value for value in estimates)
            if estimates
            else 0.0,
        }
    )
    return {
        "name": name,
        "label": label,
        "variance_reduction": variance_reduction,
        "credit_assignment": credit_assignment,
        "cost_note": cost_note,
        "metrics": metrics,
    }


def run_grid(
    *,
    seed: int = 17,
    scenario_name: str = "long_wait",
    train_groups: int = 160,
    eval_groups: int = 48,
    group_size: int = 6,
    max_steps: int = 14,
    branches_per_state: int = 32,
) -> dict[str, Any]:
    if train_groups <= 0 or eval_groups <= 0 or group_size <= 1 or max_steps < 2:
        raise ValueError("expected train/eval groups > 0, group_size > 1, max_steps >= 2")
    if branches_per_state <= 0:
        raise ValueError("branches_per_state must be positive")

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
    train_return_mean = fmean(trajectory.total_return for trajectory in train_trajectories)
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

    trajectory_by_id = {
        trajectory.trajectory_id: trajectory for trajectory in eval_trajectories
    }
    loo_advantages = build_leave_one_out_advantages(eval_groups_data)
    steps = [step for trajectory in eval_trajectories for step in trajectory.steps]
    sampled_mc, sampled_mc_counts = build_sampled_mc_advantages(
        steps,
        seed=seed + 10_000,
        scenario=scenario,
        branches_per_state=branches_per_state,
    )
    anchor_action, anchor_action_counts = build_anchor_action_contrast(steps)

    estimators = [
        summarize_estimator(
            name="reinforce_return",
            label="REINFORCE return",
            variance_reduction="none",
            credit_assignment="trajectory",
            trajectories=eval_trajectories,
            getter=lambda step: trajectory_by_id[step.trajectory_id].total_return,
            cost_note="single rollout, no baseline",
        ),
        summarize_estimator(
            name="global_baseline",
            label="Global baseline",
            variance_reduction="global/running",
            credit_assignment="trajectory",
            trajectories=eval_trajectories,
            getter=lambda step: trajectory_by_id[step.trajectory_id].total_return
            - train_return_mean,
            cost_note="single rollout plus global return tracker",
        ),
        summarize_estimator(
            name="sibling_group_norm",
            label="Sibling group norm",
            variance_reduction="sibling group",
            credit_assignment="trajectory",
            trajectories=eval_trajectories,
            getter=lambda step: step.group_advantage,
            cost_note="multiple comparable rollouts per prompt",
        ),
        summarize_estimator(
            name="leave_one_out",
            label="Leave-one-out",
            variance_reduction="sibling group",
            credit_assignment="trajectory",
            trajectories=eval_trajectories,
            getter=lambda step: loo_advantages[step.trajectory_id],
            cost_note="multiple comparable rollouts per prompt",
        ),
        summarize_estimator(
            name="dense_progress_reward",
            label="Dense progress reward",
            variance_reduction="external dense signal",
            credit_assignment="step",
            trajectories=eval_trajectories,
            getter=lambda step: float(step.next_score - step.start_score),
            cost_note="requires process signal or progress verifier",
        ),
        summarize_estimator(
            name="anchor_action_contrast",
            label="Anchor action contrast",
            variance_reduction="anchor state group",
            credit_assignment="step",
            trajectories=eval_trajectories,
            getter=lambda step: anchor_action[step_key(step)],
            cost_note="requires repeated state-action anchors",
        ),
        summarize_estimator(
            name="critic_td",
            label="Learned critic TD",
            variance_reduction="learned critic V(s)",
            credit_assignment="step",
            trajectories=eval_trajectories,
            getter=lambda step: step.critic_advantage,
            cost_note="requires fitted value model",
        ),
        summarize_estimator(
            name="sampled_mc_td",
            label="Sampled MC value",
            variance_reduction="sampled value",
            credit_assignment="step",
            trajectories=eval_trajectories,
            getter=lambda step: sampled_mc[step_key(step)],
            cost_note="requires branch rollouts from states",
        ),
        summarize_estimator(
            name="oracle_advantage",
            label="Oracle advantage ceiling",
            variance_reduction="oracle value",
            credit_assignment="step",
            trajectories=eval_trajectories,
            getter=lambda step: step.oracle_advantage,
            cost_note="not available in real training",
        ),
    ]

    non_oracle = [entry for entry in estimators if entry["name"] != "oracle_advantage"]
    best_non_oracle = max(
        non_oracle,
        key=lambda entry: entry["metrics"]["pearson_correlation"],
    )
    best_trajectory = max(
        [entry for entry in non_oracle if entry["credit_assignment"] == "trajectory"],
        key=lambda entry: entry["metrics"]["pearson_correlation"],
    )
    best_step = max(
        [entry for entry in non_oracle if entry["credit_assignment"] == "step"],
        key=lambda entry: entry["metrics"]["pearson_correlation"],
    )

    return {
        "config": {
            "seed": seed,
            "scenario_name": scenario.name,
            "scenario_description": scenario.description,
            "critic_observation": scenario.critic_observation,
            "train_groups": train_groups,
            "eval_groups": eval_groups,
            "group_size": group_size,
            "max_steps": max_steps,
            "branches_per_state": branches_per_state,
            "token_cost": scenario.token_cost,
        },
        "sample_counts": {
            "train_trajectories": len(train_trajectories),
            "eval_trajectories": len(eval_trajectories),
            "eval_tokens": len(steps),
            "train_return_mean": train_return_mean,
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
            "sampled_mc": sampled_mc_counts,
            "anchor_action": anchor_action_counts,
        },
        "estimators": estimators,
        "summary": {
            "best_non_oracle_by_correlation": best_non_oracle["name"],
            "best_trajectory_by_correlation": best_trajectory["name"],
            "best_step_by_correlation": best_step["name"],
            "step_minus_trajectory_best_correlation": (
                best_step["metrics"]["pearson_correlation"]
                - best_trajectory["metrics"]["pearson_correlation"]
            ),
            "global_minus_reinforce_second_moment": (
                next(
                    entry["metrics"]["estimate_second_moment"]
                    for entry in estimators
                    if entry["name"] == "global_baseline"
                )
                - next(
                    entry["metrics"]["estimate_second_moment"]
                    for entry in estimators
                    if entry["name"] == "reinforce_return"
                )
            ),
        },
    }


def fmt(value: float) -> str:
    return f"{value:.3f}"


def build_markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# Variance Reduction vs Credit Assignment Grid",
        "",
        "This CPU-only toy run separates variance reduction from credit",
        "assignment. It compares trajectory-level baselines with step-level",
        "estimators against the known oracle advantage. It is estimator-fidelity",
        "evidence, not a closed-loop training result.",
        "",
        "## Configuration",
        "",
    ]
    for key, value in result["config"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Estimator | Variance reduction | Credit | r | MSE | Sign | Wait leak | Within-traj var | Second moment |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for estimator in result["estimators"]:
        metrics = estimator["metrics"]
        lines.append(
            "| {label} | {variance} | {credit} | {corr} | {mse} | {sign} | "
            "{leak} | {within} | {moment} |".format(
                label=estimator["label"],
                variance=estimator["variance_reduction"],
                credit=estimator["credit_assignment"],
                corr=fmt(metrics["pearson_correlation"]),
                mse=f"{metrics['calibrated_mse']:.5f}",
                sign=fmt(metrics["sign_accuracy"]),
                leak=fmt(metrics["wait_to_active_abs_ratio"]),
                within=fmt(metrics["within_trajectory_variance"]),
                moment=fmt(metrics["estimate_second_moment"]),
            )
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- Global baselines can reduce the second moment of a trajectory-level",
            "  policy-gradient signal without creating step-level credit.",
            "- Sibling groups can be memory-light, but in this long-wait setting their",
            "  scalar advantages still have zero within-trajectory variation.",
            "- Anchor-action contrast is a critic-free structural batch estimator",
            "  over repeated exact toy states; unsupported anchors fall back to zero.",
            "- Step-level estimators create intra-trajectory variation and reduce",
            "  wait-token leakage when they have useful state or process signal.",
            "- The oracle row is a ceiling, included only to calibrate estimator fidelity.",
            "",
            "## Summary",
            "",
            f"- Best non-oracle estimator: `{result['summary']['best_non_oracle_by_correlation']}`.",
            f"- Best step-level estimator: `{result['summary']['best_step_by_correlation']}`.",
            f"- Best trajectory-level estimator: `{result['summary']['best_trajectory_by_correlation']}`.",
            "- Step best minus trajectory best correlation: "
            f"{fmt(result['summary']['step_minus_trajectory_best_correlation'])}.",
            "- Global baseline minus REINFORCE second moment: "
            f"{fmt(result['summary']['global_minus_reinforce_second_moment'])}.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="long_wait")
    parser.add_argument("--train-groups", type=int, default=160)
    parser.add_argument("--eval-groups", type=int, default=48)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=14)
    parser.add_argument("--branches-per-state", type=int, default=32)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/variance_credit_grid_seed17.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/variance_credit_grid_seed17.md"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_grid(
        seed=args.seed,
        scenario_name=args.scenario,
        train_groups=args.train_groups,
        eval_groups=args.eval_groups,
        group_size=args.group_size,
        max_steps=args.max_steps,
        branches_per_state=args.branches_per_state,
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
        "best: "
        f"non_oracle={result['summary']['best_non_oracle_by_correlation']} "
        f"step={result['summary']['best_step_by_correlation']} "
        f"trajectory={result['summary']['best_trajectory_by_correlation']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
