"""Closed-loop toy policy training for long-horizon credit estimators.

The estimator-fidelity experiments ask whether an advantage estimator matches a
known oracle. This file asks a downstream question under the same toy dynamics:
does the estimator actually improve a policy under a fixed generated-token
budget?

It intentionally remains small and CPU-only. The policy is tabular softmax over
the toy state (threshold, score, remaining horizon). The compared update signals
are:

- ``group_total``: z-score total return inside each prompt group;
- ``group_length_norm``: z-score total return divided by trajectory length;
- ``critic_td``: one-step temporal-difference estimates from a replay critic;
- ``coverage_gated``: a new hybrid that uses critic TD only when a state has
  enough replay coverage, falling back to group credit elsewhere.

Coverage-gated credit is the novel proposal here. It is not claimed as a final
LLM algorithm; it is a testable design pattern: value-based temporal credit
should be used where the value model is locally supported, while group-relative
signals should remain available for blind or undercovered regions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterable

from experiments.deep_matrix import DEFAULT_SEEDS
from experiments.toy_credit_assignment import (
    ACTIONS,
    SCENARIOS,
    Scenario,
    StepRecord,
    Trajectory,
    action_probabilities,
    next_score,
    resolve_scenario,
    state_key,
    terminal_reward,
)


DEFAULT_METHODS = [
    "group_total",
    "group_length_norm",
    "critic_td",
    "coverage_gated",
]


@dataclass(frozen=True)
class TrainingConfig:
    scenario_name: str = "long_wait"
    train_iterations: int = 70
    groups_per_iteration: int = 20
    group_size: int = 5
    max_steps: int = 14
    eval_groups: int = 80
    learning_rate: float = 0.035
    entropy_bonus: float = 0.003
    critic_replay_limit: int = 1800
    gate_min_count: int = 2
    eval_every: int = 10


class TabularPolicy:
    def __init__(self, scenario: Scenario, max_steps: int):
        self.scenario = scenario
        self.max_steps = max_steps
        self.logits: dict[tuple[int, int, int], dict[str, float]] = {}

    def _state(self, threshold: int, score: int, remaining: int) -> tuple[int, int, int]:
        return threshold, score, remaining

    def _ensure(self, threshold: int, score: int, remaining: int) -> dict[str, float]:
        key = self._state(threshold, score, remaining)
        if key not in self.logits:
            probs = action_probabilities(threshold, score, remaining, self.scenario)
            self.logits[key] = {
                action: math.log(max(probability, 1e-6))
                for action, probability in probs.items()
            }
        return self.logits[key]

    def probabilities(
        self,
        threshold: int,
        score: int,
        remaining: int,
    ) -> dict[str, float]:
        logits = self._ensure(threshold, score, remaining)
        max_logit = max(logits.values())
        exp_values = {
            action: math.exp(value - max_logit) for action, value in logits.items()
        }
        total = sum(exp_values.values())
        return {action: value / total for action, value in exp_values.items()}

    def sample_action(
        self,
        rng: random.Random,
        threshold: int,
        score: int,
        remaining: int,
    ) -> str:
        roll = rng.random()
        cumulative = 0.0
        for action, probability in self.probabilities(
            threshold,
            score,
            remaining,
        ).items():
            cumulative += probability
            if roll <= cumulative:
                return action
        return ACTIONS[-1]

    def update(
        self,
        threshold: int,
        score: int,
        remaining: int,
        action: str,
        advantage: float,
        learning_rate: float,
        entropy_bonus: float,
    ) -> None:
        logits = self._ensure(threshold, score, remaining)
        probs = self.probabilities(threshold, score, remaining)
        clipped = max(-3.0, min(3.0, advantage))

        for candidate in ACTIONS:
            grad = (1.0 if candidate == action else 0.0) - probs[candidate]
            entropy_push = -math.log(max(probs[candidate], 1e-12)) - 1.0
            logits[candidate] += learning_rate * (
                clipped * grad + entropy_bonus * entropy_push
            )


class ReplayCritic:
    def __init__(self, trajectories: Iterable[Trajectory], observation: str):
        self.observation = observation
        self.by_state: dict[tuple[int, int, int], float] = {}
        self.counts: dict[tuple[int, int, int], int] = {}
        self.by_threshold_remaining: dict[tuple[int, int], float] = {}
        self.by_remaining: dict[int, float] = {}
        all_values: list[float] = []

        state_values: dict[tuple[int, int, int], list[float]] = defaultdict(list)
        threshold_values: dict[tuple[int, int], list[float]] = defaultdict(list)
        remaining_values: dict[int, list[float]] = defaultdict(list)

        for trajectory in trajectories:
            terminal_key = state_key(
                trajectory.threshold,
                trajectory.final_score,
                0,
                observation,
            )
            state_values[terminal_key].append(trajectory.terminal_reward)
            threshold_values[(trajectory.threshold, 0)].append(trajectory.terminal_reward)
            remaining_values[0].append(trajectory.terminal_reward)
            all_values.append(trajectory.terminal_reward)
            for step in trajectory.steps:
                key = state_key(
                    step.threshold,
                    step.start_score,
                    step.remaining_before,
                    observation,
                )
                state_values[key].append(step.return_to_go)
                threshold_values[(step.threshold, step.remaining_before)].append(
                    step.return_to_go
                )
                remaining_values[step.remaining_before].append(step.return_to_go)
                all_values.append(step.return_to_go)

        self.by_state = {key: fmean(values) for key, values in state_values.items()}
        self.counts = {key: len(values) for key, values in state_values.items()}
        self.by_threshold_remaining = {
            key: fmean(values) for key, values in threshold_values.items()
        }
        self.by_remaining = {
            key: fmean(values) for key, values in remaining_values.items()
        }
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

    def count(self, threshold: int, score: int, remaining: int) -> int:
        if remaining == 0:
            return 10**9
        return self.counts.get(state_key(threshold, score, remaining, self.observation), 0)


def simulate_policy_trajectory(
    rng: random.Random,
    policy: TabularPolicy,
    trajectory_id: int,
    prompt_id: int,
    threshold: int,
    max_steps: int,
    scenario: Scenario,
) -> Trajectory:
    length = rng.randint(scenario.min_steps, max_steps)
    score = 0
    raw_steps: list[tuple[str, int, int, int, int]] = []

    for token_index in range(length):
        remaining_before = length - token_index
        remaining_after = remaining_before - 1
        start_score = score
        action = policy.sample_action(rng, threshold, score, remaining_before)
        score = next_score(score, action)
        raw_steps.append(
            (
                action,
                start_score,
                score,
                remaining_before,
                remaining_after,
            )
        )

    final_reward = terminal_reward(threshold, score)
    total_return = final_reward - scenario.token_cost * length
    steps: list[StepRecord] = []
    for token_index, raw in enumerate(raw_steps):
        action, start_score, after_score, remaining_before, remaining_after = raw
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
                return_to_go=final_reward - scenario.token_cost * (length - token_index),
                oracle_advantage=0.0,
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


def generate_policy_groups(
    rng: random.Random,
    policy: TabularPolicy,
    *,
    group_count: int,
    group_size: int,
    max_steps: int,
    scenario: Scenario,
    trajectory_offset: int,
) -> list[list[Trajectory]]:
    groups: list[list[Trajectory]] = []
    next_trajectory_id = trajectory_offset
    for prompt_id in range(group_count):
        threshold = scenario.threshold_cycle[prompt_id % len(scenario.threshold_cycle)]
        group: list[Trajectory] = []
        for _ in range(group_size):
            group.append(
                simulate_policy_trajectory(
                    rng,
                    policy,
                    next_trajectory_id,
                    prompt_id,
                    threshold,
                    max_steps,
                    scenario,
                )
            )
            next_trajectory_id += 1
        groups.append(group)
    return groups


def flatten(groups: Iterable[Iterable[Trajectory]]) -> list[Trajectory]:
    return [trajectory for group in groups for trajectory in group]


def zscore(values: list[float]) -> list[float]:
    mean_value = fmean(values)
    variance = fmean((value - mean_value) ** 2 for value in values)
    stddev = math.sqrt(variance)
    if stddev <= 1e-12:
        return [0.0 for _ in values]
    return [(value - mean_value) / stddev for value in values]


def broadcast_group_values(
    groups: list[list[Trajectory]],
    value_fn: Callable[[Trajectory], float],
) -> dict[tuple[int, int], float]:
    values: dict[tuple[int, int], float] = {}
    for group in groups:
        normalized = zscore([value_fn(trajectory) for trajectory in group])
        for trajectory, advantage in zip(group, normalized):
            for step in trajectory.steps:
                values[(step.trajectory_id, step.remaining_before)] = advantage
    return values


def normalize_step_values(values: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    raw = list(values.values())
    if not raw:
        return values
    mean_value = fmean(raw)
    variance = fmean((value - mean_value) ** 2 for value in raw)
    stddev = math.sqrt(variance)
    if stddev <= 1e-12:
        return {key: 0.0 for key in values}
    return {key: (value - mean_value) / stddev for key, value in values.items()}


def build_advantages(
    method: str,
    groups: list[list[Trajectory]],
    critic: ReplayCritic,
    config: TrainingConfig,
) -> tuple[dict[tuple[int, int], float], dict[str, float]]:
    group_total = broadcast_group_values(groups, lambda trajectory: trajectory.total_return)

    if method == "group_total":
        return group_total, {"critic_fraction": 0.0}

    if method == "group_length_norm":
        return (
            broadcast_group_values(
                groups,
                lambda trajectory: trajectory.total_return / trajectory.length,
            ),
            {"critic_fraction": 0.0},
        )

    values: dict[tuple[int, int], float] = {}
    critic_used = 0
    total_steps = 0
    for trajectory in flatten(groups):
        for step in trajectory.steps:
            total_steps += 1
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
            critic_advantage = step.step_reward + next_value - current_value
            key = (step.trajectory_id, step.remaining_before)
            if method == "critic_td":
                values[key] = critic_advantage
                if critic.count(step.threshold, step.start_score, step.remaining_before) > 0:
                    critic_used += 1
            elif method == "coverage_gated":
                if (
                    critic.count(step.threshold, step.start_score, step.remaining_before)
                    >= config.gate_min_count
                ):
                    values[key] = critic_advantage
                    critic_used += 1
                else:
                    values[key] = group_total[key]
            else:
                raise ValueError(f"unknown method {method!r}")

    return normalize_step_values(values), {
        "critic_fraction": critic_used / total_steps if total_steps else 0.0,
    }


def apply_update(
    policy: TabularPolicy,
    trajectories: list[Trajectory],
    advantages: dict[tuple[int, int], float],
    config: TrainingConfig,
) -> None:
    for trajectory in trajectories:
        for step in trajectory.steps:
            policy.update(
                step.threshold,
                step.start_score,
                step.remaining_before,
                step.action,
                advantages[(step.trajectory_id, step.remaining_before)],
                config.learning_rate,
                config.entropy_bonus,
            )


def evaluate_policy(
    rng: random.Random,
    policy: TabularPolicy,
    *,
    scenario: Scenario,
    config: TrainingConfig,
    trajectory_offset: int,
) -> dict[str, float]:
    groups = generate_policy_groups(
        rng,
        policy,
        group_count=config.eval_groups,
        group_size=1,
        max_steps=config.max_steps,
        scenario=scenario,
        trajectory_offset=trajectory_offset,
    )
    trajectories = flatten(groups)
    steps = [step for trajectory in trajectories for step in trajectory.steps]
    action_rates = {
        f"{action}_rate": (
            sum(1 for step in steps if step.action == action) / len(steps)
            if steps
            else 0.0
        )
        for action in ACTIONS
    }
    metrics = {
        "mean_return": fmean(t.total_return for t in trajectories),
        "success_rate": fmean(t.terminal_reward for t in trajectories),
        "mean_length": fmean(t.length for t in trajectories),
        "wait_token_fraction": action_rates["wait_rate"],
    }
    metrics.update(action_rates)
    return metrics


def train_method(
    *,
    seed: int,
    method: str,
    config: TrainingConfig,
) -> dict[str, Any]:
    scenario = resolve_scenario(config.scenario_name)
    rng = random.Random(seed)
    eval_rng = random.Random(seed + 100_000)
    policy = TabularPolicy(scenario, config.max_steps)
    replay: deque[Trajectory] = deque(maxlen=config.critic_replay_limit)
    trajectory_offset = 0

    initial_eval = evaluate_policy(
        eval_rng,
        policy,
        scenario=scenario,
        config=config,
        trajectory_offset=1_000_000,
    )
    learning_curve = [
        {
            "iteration": 0,
            "critic_fraction": 0.0,
            "eval": initial_eval,
        }
    ]

    for iteration in range(1, config.train_iterations + 1):
        groups = generate_policy_groups(
            rng,
            policy,
            group_count=config.groups_per_iteration,
            group_size=config.group_size,
            max_steps=config.max_steps,
            scenario=scenario,
            trajectory_offset=trajectory_offset,
        )
        trajectories = flatten(groups)
        trajectory_offset += len(trajectories)
        critic = ReplayCritic(replay, observation=scenario.critic_observation)
        advantages, diagnostics = build_advantages(method, groups, critic, config)
        apply_update(policy, trajectories, advantages, config)
        replay.extend(trajectories)

        if iteration % config.eval_every == 0 or iteration == config.train_iterations:
            learning_curve.append(
                {
                    "iteration": iteration,
                    "critic_fraction": diagnostics["critic_fraction"],
                    "eval": evaluate_policy(
                        eval_rng,
                        policy,
                        scenario=scenario,
                        config=config,
                        trajectory_offset=1_000_000 + iteration * config.eval_groups,
                    ),
                }
            )

    final_eval = learning_curve[-1]["eval"]
    return {
        "seed": seed,
        "method": method,
        "config": config.__dict__,
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "improvement": {
            "mean_return": final_eval["mean_return"] - initial_eval["mean_return"],
            "success_rate": final_eval["success_rate"] - initial_eval["success_rate"],
        },
        "learning_curve": learning_curve,
    }


def mean(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    values = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return fmean(values)


def summarize_method(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "method": method,
        "seed_count": len(rows),
        "initial_return": mean(rows, ("initial_eval", "mean_return")),
        "final_return": mean(rows, ("final_eval", "mean_return")),
        "return_improvement": mean(rows, ("improvement", "mean_return")),
        "initial_success": mean(rows, ("initial_eval", "success_rate")),
        "final_success": mean(rows, ("final_eval", "success_rate")),
        "success_improvement": mean(rows, ("improvement", "success_rate")),
        "final_wait_fraction": mean(rows, ("final_eval", "wait_token_fraction")),
        "final_help_rate": mean(rows, ("final_eval", "help_rate")),
        "final_harm_rate": mean(rows, ("final_eval", "harm_rate")),
        "final_critic_fraction": fmean(
            row["learning_curve"][-1]["critic_fraction"] for row in rows
        ),
    }


def run_closed_loop(
    *,
    seeds: list[int] | None = None,
    methods: list[str] | None = None,
    config: TrainingConfig | None = None,
) -> dict[str, Any]:
    seeds = list(DEFAULT_SEEDS[:10] if seeds is None else seeds)
    methods = list(DEFAULT_METHODS if methods is None else methods)
    config = TrainingConfig() if config is None else config
    if not seeds:
        raise ValueError("expected at least one seed")
    unknown = sorted(set(methods) - set(DEFAULT_METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")

    runs: list[dict[str, Any]] = []
    for method in methods:
        for seed in seeds:
            runs.append(train_method(seed=seed, method=method, config=config))

    by_method = {
        method: [row for row in runs if row["method"] == method] for method in methods
    }
    summaries = [summarize_method(method, by_method[method]) for method in methods]
    best = max(summaries, key=lambda row: row["final_return"])
    group_total = next(row for row in summaries if row["method"] == "group_total")
    coverage = next(row for row in summaries if row["method"] == "coverage_gated")
    return {
        "config": {
            **config.__dict__,
            "seeds": seeds,
            "methods": methods,
        },
        "method_summaries": summaries,
        "runs": runs,
        "summary": {
            "best_by_final_return": best["method"],
            "coverage_minus_group_return": (
                coverage["final_return"] - group_total["final_return"]
            ),
            "coverage_minus_group_success": (
                coverage["final_success"] - group_total["final_success"]
            ),
        },
    }


def fmt(value: float) -> str:
    return f"{value:.3f}"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Closed-Loop Credit Training",
        "",
        "A tabular softmax policy is trained directly in the toy environment under",
        "matched generated-token budgets. Coverage-gated credit is the hybrid",
        "proposal: use critic TD only for states with replay coverage, otherwise",
        "fall back to group-relative credit.",
        "",
        "| Method | Initial return | Final return | Delta return | Final success | Final wait | Critic frac |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["method_summaries"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["method"],
                    fmt(row["initial_return"]),
                    fmt(row["final_return"]),
                    fmt(row["return_improvement"]),
                    fmt(row["final_success"]),
                    fmt(row["final_wait_fraction"]),
                    fmt(row["final_critic_fraction"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Summary:",
            f"- Best by final return: {result['summary']['best_by_final_return']}",
            f"- Coverage-gated minus group return: {fmt(result['summary']['coverage_minus_group_return'])}",
            f"- Coverage-gated minus group success: {fmt(result['summary']['coverage_minus_group_success'])}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--methods", choices=DEFAULT_METHODS, nargs="*", default=None)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="long_wait")
    parser.add_argument("--train-iterations", type=int, default=70)
    parser.add_argument("--groups-per-iteration", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=14)
    parser.add_argument("--eval-groups", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--entropy-bonus", type=float, default=0.003)
    parser.add_argument("--critic-replay-limit", type=int, default=1800)
    parser.add_argument("--gate-min-count", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/closed_loop_credit_training_10seed.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/closed_loop_credit_training_10seed.md"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TrainingConfig(
        scenario_name=args.scenario,
        train_iterations=args.train_iterations,
        groups_per_iteration=args.groups_per_iteration,
        group_size=args.group_size,
        max_steps=args.max_steps,
        eval_groups=args.eval_groups,
        learning_rate=args.learning_rate,
        entropy_bonus=args.entropy_bonus,
        critic_replay_limit=args.critic_replay_limit,
        gate_min_count=args.gate_min_count,
        eval_every=args.eval_every,
    )
    result = run_closed_loop(
        seeds=args.seeds,
        methods=args.methods,
        config=config,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    write_markdown(result, args.output_md)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    print(f"best: {result['summary']['best_by_final_return']}")
    print(
        "coverage_minus_group_return: "
        f"{result['summary']['coverage_minus_group_return']:.3f}"
    )
    print(
        "metrics: "
        f"coverage_minus_group_return={result['summary']['coverage_minus_group_return']:.6f} "
        f"coverage_minus_group_success={result['summary']['coverage_minus_group_success']:.6f} "
        f"best_is_coverage_gated={1.0 if result['summary']['best_by_final_return'] == 'coverage_gated' else 0.0}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
