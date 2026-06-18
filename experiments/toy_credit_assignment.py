"""Toy comparison of group-relative and critic-style advantages.

The environment is intentionally tiny. Each trajectory has a hidden score, a
variable remaining horizon, and three possible token/action types:

- ``help`` increases the score.
- ``harm`` decreases the score.
- ``wait`` spends a token without changing the score.

A terminal verifier gives success when the final score reaches the prompt's
threshold. The known dynamics let us compute an oracle state-action advantage.
The experiment then compares:

1. A GRPO-style group-relative response advantage, broadcast to every token in a
   sampled trajectory.
2. A value-model-style estimator, learned from sampled rollouts with a tabular
   critic and used in a one-step TD advantage.

This is not an RL training benchmark. It isolates one credit-assignment
mechanism that becomes important in variable-length long-horizon traces.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Callable, Iterable

TOKEN_COST = 0.02
ACTIONS = ("help", "harm", "wait")


@dataclass
class StepRecord:
    trajectory_id: int
    prompt_id: int
    threshold: int
    action: str
    start_score: int
    next_score: int
    remaining_before: int
    remaining_after: int
    step_reward: float
    terminal_reward: float
    total_return: float
    return_to_go: float
    oracle_advantage: float
    group_advantage: float = 0.0
    critic_advantage: float = 0.0


@dataclass
class Trajectory:
    trajectory_id: int
    prompt_id: int
    threshold: int
    length: int
    final_score: int
    terminal_reward: float
    total_return: float
    steps: list[StepRecord]


class TabularCritic:
    """A small value estimator trained from sampled returns-to-go."""

    def __init__(self, trajectories: Iterable[Trajectory]):
        by_state: dict[tuple[int, int, int], list[float]] = defaultdict(list)
        by_threshold_remaining: dict[tuple[int, int], list[float]] = defaultdict(list)
        by_remaining: dict[int, list[float]] = defaultdict(list)
        all_values: list[float] = []

        for trajectory in trajectories:
            terminal_key = state_key(
                trajectory.threshold,
                trajectory.final_score,
                0,
            )
            by_state[terminal_key].append(trajectory.terminal_reward)
            by_threshold_remaining[(trajectory.threshold, 0)].append(
                trajectory.terminal_reward
            )
            by_remaining[0].append(trajectory.terminal_reward)
            all_values.append(trajectory.terminal_reward)

            for step in trajectory.steps:
                key = state_key(step.threshold, step.start_score, step.remaining_before)
                by_state[key].append(step.return_to_go)
                by_threshold_remaining[
                    (step.threshold, step.remaining_before)
                ].append(step.return_to_go)
                by_remaining[step.remaining_before].append(step.return_to_go)
                all_values.append(step.return_to_go)

        self.by_state = {key: fmean(values) for key, values in by_state.items()}
        self.by_threshold_remaining = {
            key: fmean(values) for key, values in by_threshold_remaining.items()
        }
        self.by_remaining = {key: fmean(values) for key, values in by_remaining.items()}
        self.global_mean = fmean(all_values) if all_values else 0.0

    def value(self, threshold: int, score: int, remaining: int) -> float:
        key = state_key(threshold, score, remaining)
        if key in self.by_state:
            return self.by_state[key]

        threshold_key = (threshold, remaining)
        if threshold_key in self.by_threshold_remaining:
            return self.by_threshold_remaining[threshold_key]

        if remaining in self.by_remaining:
            return self.by_remaining[remaining]

        return self.global_mean


def state_key(threshold: int, score: int, remaining: int) -> tuple[int, int, int]:
    return threshold, score, remaining


def action_probabilities(threshold: int, score: int, remaining: int) -> dict[str, float]:
    """Behavior policy used to sample toy trajectories.

    Once a trajectory is likely to succeed, the policy often emits ``wait``
    tokens. That creates the variable-length padding behavior where
    response-level advantages can assign credit to uninformative tokens.
    """

    gap = threshold - score
    if gap >= 2:
        probs = {"help": 0.58, "harm": 0.16, "wait": 0.26}
    elif gap == 1:
        probs = {"help": 0.46, "harm": 0.18, "wait": 0.36}
    elif gap == 0:
        probs = {"help": 0.25, "harm": 0.18, "wait": 0.57}
    else:
        probs = {"help": 0.18, "harm": 0.14, "wait": 0.68}

    if remaining <= 2 and gap > 0:
        probs["help"] += 0.08
        probs["wait"] -= 0.08

    return probs


def sample_action(rng: random.Random, threshold: int, score: int, remaining: int) -> str:
    roll = rng.random()
    cumulative = 0.0
    for action, probability in action_probabilities(threshold, score, remaining).items():
        cumulative += probability
        if roll <= cumulative:
            return action
    return "wait"


def next_score(score: int, action: str) -> int:
    if action == "help":
        return score + 1
    if action == "harm":
        return score - 1
    return score


def terminal_reward(threshold: int, score: int) -> float:
    return 1.0 if score >= threshold else 0.0


def exact_value(threshold: int, score: int, remaining: int) -> float:
    """Expected future return under the behavior policy from a known state."""

    cache: dict[tuple[int, int, int], float] = {}

    def solve(state_threshold: int, state_score: int, state_remaining: int) -> float:
        key = (state_threshold, state_score, state_remaining)
        if key in cache:
            return cache[key]
        if state_remaining == 0:
            value = terminal_reward(state_threshold, state_score)
        else:
            value = 0.0
            probs = action_probabilities(state_threshold, state_score, state_remaining)
            for action, probability in probs.items():
                score_after = next_score(state_score, action)
                value += probability * (
                    -TOKEN_COST
                    + solve(state_threshold, score_after, state_remaining - 1)
                )
        cache[key] = value
        return value

    return solve(threshold, score, remaining)


def simulate_trajectory(
    rng: random.Random,
    trajectory_id: int,
    prompt_id: int,
    threshold: int,
    max_steps: int,
) -> Trajectory:
    length = rng.randint(2, max_steps)
    score = 0
    raw_steps: list[tuple[str, int, int, int, int, float]] = []

    for token_index in range(length):
        remaining_before = length - token_index
        remaining_after = remaining_before - 1
        start_score = score
        action = sample_action(rng, threshold, score, remaining_before)
        score = next_score(score, action)
        oracle_advantage = (
            -TOKEN_COST
            + exact_value(threshold, score, remaining_after)
            - exact_value(threshold, start_score, remaining_before)
        )
        raw_steps.append(
            (
                action,
                start_score,
                score,
                remaining_before,
                remaining_after,
                oracle_advantage,
            )
        )

    final_reward = terminal_reward(threshold, score)
    total_return = final_reward - TOKEN_COST * length
    steps: list[StepRecord] = []
    for token_index, raw_step in enumerate(raw_steps):
        action, start_score, after_score, remaining_before, remaining_after, oracle = raw_step
        return_to_go = final_reward - TOKEN_COST * (length - token_index)
        steps.append(
            StepRecord(
                trajectory_id=trajectory_id,
                prompt_id=prompt_id,
                threshold=threshold,
                action=action,
                start_score=start_score,
                next_score=after_score,
                remaining_before=remaining_before,
                remaining_after=remaining_after,
                step_reward=-TOKEN_COST,
                terminal_reward=final_reward,
                total_return=total_return,
                return_to_go=return_to_go,
                oracle_advantage=oracle,
            )
        )

    return Trajectory(
        trajectory_id=trajectory_id,
        prompt_id=prompt_id,
        threshold=threshold,
        length=length,
        final_score=score,
        terminal_reward=final_reward,
        total_return=total_return,
        steps=steps,
    )


def generate_groups(
    rng: random.Random,
    *,
    group_count: int,
    group_size: int,
    max_steps: int,
    trajectory_offset: int = 0,
) -> list[list[Trajectory]]:
    groups: list[list[Trajectory]] = []
    next_trajectory_id = trajectory_offset
    for prompt_id in range(group_count):
        threshold = 1 + (prompt_id % 3)
        group: list[Trajectory] = []
        for _ in range(group_size):
            group.append(
                simulate_trajectory(
                    rng=rng,
                    trajectory_id=next_trajectory_id,
                    prompt_id=prompt_id,
                    threshold=threshold,
                    max_steps=max_steps,
                )
            )
            next_trajectory_id += 1
        groups.append(group)
    return groups


def flatten(groups: Iterable[Iterable[Trajectory]]) -> list[Trajectory]:
    return [trajectory for group in groups for trajectory in group]


def add_group_relative_advantages(groups: list[list[Trajectory]]) -> None:
    for group in groups:
        returns = [trajectory.total_return for trajectory in group]
        mean_return = fmean(returns)
        variance = fmean((value - mean_return) ** 2 for value in returns)
        stddev = math.sqrt(variance)
        for trajectory in group:
            advantage = 0.0
            if stddev > 1e-12:
                advantage = (trajectory.total_return - mean_return) / stddev
            for step in trajectory.steps:
                step.group_advantage = advantage


def add_critic_advantages(
    trajectories: Iterable[Trajectory],
    critic: TabularCritic,
) -> None:
    for trajectory in trajectories:
        for step in trajectory.steps:
            current_value = critic.value(
                step.threshold,
                step.start_score,
                step.remaining_before,
            )
            next_value = critic.value(
                step.threshold,
                step.next_score,
                step.remaining_after,
            )
            step.critic_advantage = step.step_reward + next_value - current_value


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or not xs:
        return 0.0
    mean_x = fmean(xs)
    mean_y = fmean(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    var_x = sum(value * value for value in centered_x)
    var_y = sum(value * value for value in centered_y)
    if var_x <= 1e-15 or var_y <= 1e-15:
        return 0.0
    cov = sum(x * y for x, y in zip(centered_x, centered_y))
    return cov / math.sqrt(var_x * var_y)


def calibrated_mse(estimates: list[float], targets: list[float]) -> float:
    """MSE after fitting one affine calibration, so scale does not dominate."""

    if not estimates:
        return 0.0
    mean_estimate = fmean(estimates)
    mean_target = fmean(targets)
    centered_estimates = [value - mean_estimate for value in estimates]
    centered_targets = [value - mean_target for value in targets]
    var_estimate = sum(value * value for value in centered_estimates)
    if var_estimate <= 1e-15:
        slope = 0.0
    else:
        slope = (
            sum(x * y for x, y in zip(centered_estimates, centered_targets))
            / var_estimate
        )
    intercept = mean_target - slope * mean_estimate
    return fmean((slope * estimate + intercept - target) ** 2 for estimate, target in zip(estimates, targets))


def sign_accuracy(estimates: list[float], targets: list[float]) -> float:
    if not estimates:
        return 0.0
    matches = 0
    considered = 0
    for estimate, target in zip(estimates, targets):
        if abs(target) <= 1e-9:
            continue
        considered += 1
        if math.copysign(1.0, estimate) == math.copysign(1.0, target):
            matches += 1
    return matches / considered if considered else 0.0


def mean_abs(values: Iterable[float]) -> float:
    values = list(values)
    return fmean(abs(value) for value in values) if values else 0.0


def within_trajectory_variance(
    trajectories: Iterable[Trajectory],
    getter: Callable[[StepRecord], float],
) -> float:
    variances: list[float] = []
    for trajectory in trajectories:
        values = [getter(step) for step in trajectory.steps]
        if len(values) <= 1:
            continue
        mean_value = fmean(values)
        variances.append(fmean((value - mean_value) ** 2 for value in values))
    return fmean(variances) if variances else 0.0


def estimator_metrics(
    trajectories: list[Trajectory],
    getter: Callable[[StepRecord], float],
) -> dict[str, float]:
    steps = [step for trajectory in trajectories for step in trajectory.steps]
    estimates = [getter(step) for step in steps]
    targets = [step.oracle_advantage for step in steps]
    wait_estimates = [getter(step) for step in steps if step.action == "wait"]
    active_estimates = [getter(step) for step in steps if step.action != "wait"]
    active_abs = mean_abs(active_estimates)
    return {
        "pearson_correlation": pearson(estimates, targets),
        "calibrated_mse": calibrated_mse(estimates, targets),
        "raw_mse": fmean((estimate - target) ** 2 for estimate, target in zip(estimates, targets)),
        "sign_accuracy": sign_accuracy(estimates, targets),
        "mean_abs_wait_tokens": mean_abs(wait_estimates),
        "mean_abs_active_tokens": active_abs,
        "wait_to_active_abs_ratio": mean_abs(wait_estimates) / active_abs if active_abs > 0 else 0.0,
        "within_trajectory_variance": within_trajectory_variance(
            trajectories,
            getter,
        ),
    }


def run_experiment(
    *,
    seed: int = 7,
    train_groups: int = 240,
    eval_groups: int = 80,
    group_size: int = 6,
    max_steps: int = 12,
) -> dict[str, object]:
    if train_groups <= 0 or eval_groups <= 0 or group_size <= 1 or max_steps < 2:
        raise ValueError("expected train/eval groups > 0, group_size > 1, max_steps >= 2")

    train_rng = random.Random(seed)
    train_groups_data = generate_groups(
        train_rng,
        group_count=train_groups,
        group_size=group_size,
        max_steps=max_steps,
    )
    train_trajectories = flatten(train_groups_data)
    critic = TabularCritic(train_trajectories)

    eval_rng = random.Random(seed + 1)
    eval_groups_data = generate_groups(
        eval_rng,
        group_count=eval_groups,
        group_size=group_size,
        max_steps=max_steps,
        trajectory_offset=len(train_trajectories),
    )
    add_group_relative_advantages(eval_groups_data)
    eval_trajectories = flatten(eval_groups_data)
    add_critic_advantages(eval_trajectories, critic)

    steps = [step for trajectory in eval_trajectories for step in trajectory.steps]
    group_metrics = estimator_metrics(
        eval_trajectories,
        lambda step: step.group_advantage,
    )
    critic_metrics = estimator_metrics(
        eval_trajectories,
        lambda step: step.critic_advantage,
    )
    oracle_values = [step.oracle_advantage for step in steps]

    return {
        "config": {
            "seed": seed,
            "train_groups": train_groups,
            "eval_groups": eval_groups,
            "group_size": group_size,
            "max_steps": max_steps,
            "token_cost": TOKEN_COST,
        },
        "sample_counts": {
            "train_trajectories": len(train_trajectories),
            "eval_trajectories": len(eval_trajectories),
            "eval_tokens": len(steps),
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
        },
        "oracle": {
            "mean_advantage": fmean(oracle_values) if oracle_values else 0.0,
            "advantage_variance": (
                fmean((value - fmean(oracle_values)) ** 2 for value in oracle_values)
                if oracle_values
                else 0.0
            ),
        },
        "metrics": {
            "group_relative": group_metrics,
            "critic_value_model": critic_metrics,
            "comparison": {
                "critic_minus_group_correlation": (
                    critic_metrics["pearson_correlation"]
                    - group_metrics["pearson_correlation"]
                ),
                "critic_minus_group_calibrated_mse": (
                    critic_metrics["calibrated_mse"] - group_metrics["calibrated_mse"]
                ),
                "critic_minus_group_wait_leakage": (
                    critic_metrics["wait_to_active_abs_ratio"]
                    - group_metrics["wait_to_active_abs_ratio"]
                ),
            },
        },
        "example_eval_steps": [asdict(step) for step in steps[:8]],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-groups", type=int, default=240)
    parser.add_argument("--eval-groups", type=int, default=80)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("runs/toy_credit_assignment.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_experiment(
        seed=args.seed,
        train_groups=args.train_groups,
        eval_groups=args.eval_groups,
        group_size=args.group_size,
        max_steps=args.max_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    group = result["metrics"]["group_relative"]
    critic = result["metrics"]["critic_value_model"]
    print(f"wrote {args.output}")
    print(
        "pearson: "
        f"group={group['pearson_correlation']:.3f} "
        f"critic={critic['pearson_correlation']:.3f}"
    )
    print(
        "calibrated_mse: "
        f"group={group['calibrated_mse']:.5f} "
        f"critic={critic['calibrated_mse']:.5f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

