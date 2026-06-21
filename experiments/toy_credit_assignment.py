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
from functools import lru_cache
from pathlib import Path
from statistics import fmean
from typing import Callable, Iterable

TOKEN_COST = 0.02
ACTIONS = ("help", "harm", "wait")


OBSERVATION_SCHEMAS: dict[str, dict[str, bool | str]] = {
    "actor": {
        "prompt_threshold": True,
        "progress": True,
        "remaining_horizon": True,
        "previous_actions": False,
        "description": "The toy behavior policy observes threshold, progress, and remaining horizon.",
    },
    "non_privileged": {
        "prompt_threshold": True,
        "progress": True,
        "remaining_horizon": True,
        "previous_actions": False,
        "description": "Critic observation equal to the actor observation z_t=o_t.",
    },
    "full": {
        "prompt_threshold": True,
        "progress": True,
        "remaining_horizon": True,
        "previous_actions": False,
        "description": "Full toy state used by the behavior policy; not privileged in this toy.",
    },
    "coarse": {
        "prompt_threshold": True,
        "progress": "sign",
        "remaining_horizon": True,
        "previous_actions": False,
        "description": "Threshold and remaining horizon plus only the sign of progress.",
    },
    "blind": {
        "prompt_threshold": True,
        "progress": False,
        "remaining_horizon": True,
        "previous_actions": False,
        "description": "Threshold and remaining horizon, with progress hidden.",
    },
}


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    token_cost: float = TOKEN_COST
    wait_bias_after_success: float = 0.0
    help_bias_when_behind: float = 0.0
    harm_bias: float = 0.0
    critic_observation: str = "full"
    threshold_cycle: tuple[int, ...] = (1, 2, 3)
    min_steps: int = 2


DEFAULT_SCENARIO = Scenario(
    name="baseline",
    description="Original mixed-horizon toy setting with a full-state critic.",
)

SCENARIOS: dict[str, Scenario] = {
    DEFAULT_SCENARIO.name: DEFAULT_SCENARIO,
    "short_dense": Scenario(
        name="short_dense",
        description="Shorter traces with fewer wait tokens; response-level groups are less wasteful.",
        wait_bias_after_success=-0.18,
        help_bias_when_behind=0.08,
        threshold_cycle=(1, 1, 2),
    ),
    "long_wait": Scenario(
        name="long_wait",
        description="Long successful traces often contain uninformative wait tokens.",
        wait_bias_after_success=0.24,
        threshold_cycle=(1, 2, 3),
    ),
    "sparse_hard": Scenario(
        name="sparse_hard",
        description="Harder thresholds produce sparse successes and weaker group reward dispersion.",
        wait_bias_after_success=0.10,
        help_bias_when_behind=-0.06,
        threshold_cycle=(2, 3, 4),
    ),
    "coarse_critic": Scenario(
        name="coarse_critic",
        description="The critic only observes score sign, not exact progress.",
        wait_bias_after_success=0.18,
        critic_observation="coarse",
    ),
    "blind_critic": Scenario(
        name="blind_critic",
        description="The critic cannot observe progress score, so terminal group information can dominate.",
        wait_bias_after_success=0.16,
        critic_observation="blind",
    ),
}


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

    def __init__(self, trajectories: Iterable[Trajectory], observation: str = "full"):
        self.observation = observation
        by_state: dict[tuple[int, int, int], list[float]] = defaultdict(list)
        by_threshold_remaining: dict[tuple[int, int], list[float]] = defaultdict(list)
        by_remaining: dict[int, list[float]] = defaultdict(list)
        all_values: list[float] = []

        for trajectory in trajectories:
            terminal_key = state_key(
                trajectory.threshold,
                trajectory.final_score,
                0,
                self.observation,
            )
            by_state[terminal_key].append(trajectory.terminal_reward)
            by_threshold_remaining[(trajectory.threshold, 0)].append(
                trajectory.terminal_reward
            )
            by_remaining[0].append(trajectory.terminal_reward)
            all_values.append(trajectory.terminal_reward)

            for step in trajectory.steps:
                key = state_key(
                    step.threshold,
                    step.start_score,
                    step.remaining_before,
                    self.observation,
                )
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
        if remaining == 0:
            return terminal_reward(threshold, score)

        key = state_key(threshold, score, remaining, self.observation)
        if key in self.by_state:
            return self.by_state[key]

        threshold_key = (threshold, remaining)
        if threshold_key in self.by_threshold_remaining:
            return self.by_threshold_remaining[threshold_key]

        if remaining in self.by_remaining:
            return self.by_remaining[remaining]

        return self.global_mean

    def has_exact_state(self, threshold: int, score: int, remaining: int) -> bool:
        if remaining == 0:
            return True
        key = state_key(threshold, score, remaining, self.observation)
        return key in self.by_state


def state_key(
    threshold: int,
    score: int,
    remaining: int,
    observation: str = "full",
) -> tuple[int, int, int]:
    if observation == "blind":
        observed_score = 0
    elif observation == "coarse":
        observed_score = 1 if score > 0 else -1 if score < 0 else 0
    elif observation in {"full", "actor", "non_privileged"}:
        observed_score = score
    else:
        raise ValueError(f"unknown observation mode {observation!r}")
    return threshold, observed_score, remaining


def observation_schema(observation: str) -> dict[str, bool | str]:
    if observation not in OBSERVATION_SCHEMAS:
        raise ValueError(f"unknown observation mode {observation!r}")
    return dict(OBSERVATION_SCHEMAS[observation])


def critic_is_privileged(observation: str) -> bool:
    actor = OBSERVATION_SCHEMAS["actor"]
    critic = OBSERVATION_SCHEMAS[observation]
    progress_order = {False: 0, "sign": 1, True: 2}
    for key in ["prompt_threshold", "remaining_horizon", "previous_actions"]:
        if bool(critic[key]) and not bool(actor[key]):
            return True
    return progress_order[critic["progress"]] > progress_order[actor["progress"]]


def normalize_probabilities(probs: dict[str, float]) -> dict[str, float]:
    clipped = {action: max(0.02, probability) for action, probability in probs.items()}
    total = sum(clipped.values())
    return {action: probability / total for action, probability in clipped.items()}


def action_probabilities(
    threshold: int,
    score: int,
    remaining: int,
    scenario: Scenario = DEFAULT_SCENARIO,
) -> dict[str, float]:
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

    if gap > 0:
        probs["help"] += scenario.help_bias_when_behind
        probs["wait"] -= scenario.help_bias_when_behind
    else:
        probs["wait"] += scenario.wait_bias_after_success
        probs["help"] -= scenario.wait_bias_after_success * 0.7
        probs["harm"] -= scenario.wait_bias_after_success * 0.3

    probs["harm"] += scenario.harm_bias
    probs["wait"] -= scenario.harm_bias

    return normalize_probabilities(probs)


def sample_action(
    rng: random.Random,
    threshold: int,
    score: int,
    remaining: int,
    scenario: Scenario,
) -> str:
    roll = rng.random()
    cumulative = 0.0
    for action, probability in action_probabilities(
        threshold,
        score,
        remaining,
        scenario,
    ).items():
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


@lru_cache(maxsize=None)
def exact_value(
    threshold: int,
    score: int,
    remaining: int,
    scenario: Scenario = DEFAULT_SCENARIO,
) -> float:
    """Expected future return under the behavior policy from a known state."""

    if remaining == 0:
        return terminal_reward(threshold, score)

    value = 0.0
    probs = action_probabilities(threshold, score, remaining, scenario)
    for action, probability in probs.items():
        score_after = next_score(score, action)
        value += probability * (
            -scenario.token_cost
            + exact_value(threshold, score_after, remaining - 1, scenario)
        )
    return value


def simulate_trajectory(
    rng: random.Random,
    trajectory_id: int,
    prompt_id: int,
    threshold: int,
    max_steps: int,
    scenario: Scenario = DEFAULT_SCENARIO,
) -> Trajectory:
    length = rng.randint(scenario.min_steps, max_steps)
    score = 0
    raw_steps: list[tuple[str, int, int, int, int, float]] = []

    for token_index in range(length):
        remaining_before = length - token_index
        remaining_after = remaining_before - 1
        start_score = score
        action = sample_action(rng, threshold, score, remaining_before, scenario)
        score = next_score(score, action)
        oracle_advantage = (
            -scenario.token_cost
            + exact_value(threshold, score, remaining_after, scenario)
            - exact_value(threshold, start_score, remaining_before, scenario)
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
    total_return = final_reward - scenario.token_cost * length
    steps: list[StepRecord] = []
    for token_index, raw_step in enumerate(raw_steps):
        action, start_score, after_score, remaining_before, remaining_after, oracle = raw_step
        return_to_go = final_reward - scenario.token_cost * (length - token_index)
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
                step_reward=-scenario.token_cost,
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
    scenario: Scenario = DEFAULT_SCENARIO,
    trajectory_offset: int = 0,
) -> list[list[Trajectory]]:
    groups: list[list[Trajectory]] = []
    next_trajectory_id = trajectory_offset
    for prompt_id in range(group_count):
        threshold = scenario.threshold_cycle[prompt_id % len(scenario.threshold_cycle)]
        group: list[Trajectory] = []
        for _ in range(group_size):
            group.append(
                simulate_trajectory(
                    rng=rng,
                    trajectory_id=next_trajectory_id,
                    prompt_id=prompt_id,
                    threshold=threshold,
                    max_steps=max_steps,
                    scenario=scenario,
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


def group_diagnostics(groups: list[list[Trajectory]]) -> dict[str, float]:
    stddevs: list[float] = []
    zero_variance_groups = 0
    for group in groups:
        returns = [trajectory.total_return for trajectory in group]
        mean_return = fmean(returns)
        variance = fmean((value - mean_return) ** 2 for value in returns)
        stddev = math.sqrt(variance)
        stddevs.append(stddev)
        if stddev <= 1e-12:
            zero_variance_groups += 1

    return {
        "zero_variance_group_fraction": (
            zero_variance_groups / len(groups) if groups else 0.0
        ),
        "mean_group_return_stddev": fmean(stddevs) if stddevs else 0.0,
    }


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
    if len(xs) != len(ys):
        raise ValueError("expected equal-length inputs")
    if not xs:
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

    if len(estimates) != len(targets):
        raise ValueError("expected equal-length inputs")
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
    if len(estimates) != len(targets):
        raise ValueError("expected equal-length inputs")
    if not estimates:
        return 0.0
    matches = 0
    considered = 0
    for estimate, target in zip(estimates, targets):
        if abs(target) <= 1e-9:
            continue
        considered += 1
        if abs(estimate) <= 1e-9:
            continue
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


def resolve_scenario(
    scenario_name: str = DEFAULT_SCENARIO.name,
    scenario: Scenario | None = None,
) -> Scenario:
    if scenario is not None:
        return scenario
    if scenario_name not in SCENARIOS:
        choices = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"unknown scenario {scenario_name!r}; choices: {choices}")
    return SCENARIOS[scenario_name]


def run_experiment(
    *,
    seed: int = 7,
    scenario_name: str = DEFAULT_SCENARIO.name,
    scenario: Scenario | None = None,
    train_groups: int = 240,
    eval_groups: int = 80,
    group_size: int = 6,
    train_group_size: int | None = None,
    max_steps: int = 12,
) -> dict[str, object]:
    train_group_size = group_size if train_group_size is None else train_group_size
    if (
        train_groups <= 0
        or eval_groups <= 0
        or group_size <= 1
        or train_group_size <= 0
        or max_steps < 2
    ):
        raise ValueError("expected train/eval groups > 0, group_size > 1, max_steps >= 2")
    scenario = resolve_scenario(scenario_name, scenario)
    if scenario.min_steps > max_steps:
        raise ValueError("scenario min_steps must be <= max_steps")

    train_rng = random.Random(seed)
    train_groups_data = generate_groups(
        train_rng,
        group_count=train_groups,
        group_size=train_group_size,
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
    group_stats = group_diagnostics(eval_groups_data)
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
            "scenario_name": scenario.name,
            "scenario_description": scenario.description,
            "critic_observation": scenario.critic_observation,
            "actor_observation_schema": observation_schema("actor"),
            "critic_observation_schema": observation_schema(scenario.critic_observation),
            "critic_is_privileged": critic_is_privileged(scenario.critic_observation),
            "train_groups": train_groups,
            "eval_groups": eval_groups,
            "group_size": group_size,
            "train_group_size": train_group_size,
            "max_steps": max_steps,
            "token_cost": scenario.token_cost,
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
            "zero_variance_group_fraction": group_stats[
                "zero_variance_group_fraction"
            ],
            "mean_group_return_stddev": group_stats["mean_group_return_stddev"],
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
            "nonzero_oracle_advantage_fraction": (
                sum(1 for step in steps if abs(step.oracle_advantage) > 1e-9)
                / len(steps)
                if steps
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
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default=DEFAULT_SCENARIO.name,
    )
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
        scenario_name=args.scenario,
        train_groups=args.train_groups,
        eval_groups=args.eval_groups,
        group_size=args.group_size,
        max_steps=args.max_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )

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
