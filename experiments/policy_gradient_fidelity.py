"""Exact finite-MDP gradient-fidelity audit for token-credit estimators.

The older toy experiments measure whether step-level estimates correlate with a
known oracle advantage. This module asks a stricter policy-optimization
question in a tiny MDP where the expected return and its gradient can be
computed directly. It also introduces a genuine ``null`` action: unlike
``delay``, null changes neither progress nor the effective decision horizon, so
its exact one-step advantage is zero.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from collections import defaultdict
from statistics import fmean, stdev
from typing import Any, Callable, Iterable

ACTIONS = ("help", "harm", "delay", "null")
NON_NULL_ACTIONS = tuple(action for action in ACTIONS if action != "null")


@dataclass(frozen=True)
class MDPConfig:
    threshold: int = 2
    horizon: int = 5
    token_cost: float = 0.02
    beta: float = 0.35
    target_kl: float = 0.003
    score_margin: int = 3


@dataclass(frozen=True)
class Step:
    position: int
    score: int
    remaining: int
    action: str
    next_score: int
    next_remaining: int
    reward: float
    return_to_go: float = 0.0


@dataclass(frozen=True)
class Trajectory:
    steps: tuple[Step, ...]
    terminal_reward: float
    total_return: float


@dataclass(frozen=True)
class Gradient:
    values: tuple[float, ...]

    @property
    def norm(self) -> float:
        return math.sqrt(sum(value * value for value in self.values))

    def dot(self, other: "Gradient") -> float:
        return sum(a * b for a, b in zip(self.values, other.values))

    def minus(self, other: "Gradient") -> "Gradient":
        return Gradient(tuple(a - b for a, b in zip(self.values, other.values)))

    def scaled(self, scale: float) -> "Gradient":
        return Gradient(tuple(scale * value for value in self.values))

    @classmethod
    def zeros(cls, size: int) -> "Gradient":
        return cls(tuple(0.0 for _ in range(size)))


class TabularSoftmaxPolicy:
    """Tabular softmax policy over ``(score, effective_remaining)`` states."""

    def __init__(self, config: MDPConfig):
        self.config = config
        self.logits: dict[tuple[int, int], dict[str, float]] = {}
        self._value_cache: dict[tuple[int, int], float] = {}
        self.materialize()

    def copy(self) -> "TabularSoftmaxPolicy":
        clone = TabularSoftmaxPolicy(self.config)
        clone.logits = {
            state: dict(action_logits) for state, action_logits in self.logits.items()
        }
        clone._value_cache = {}
        return clone

    @cached_property
    def parameter_keys(self) -> tuple[tuple[int, int, str], ...]:
        return tuple(
            (score, remaining, action)
            for remaining in range(1, self.config.horizon + 1)
            for score in self.score_range()
            for action in ACTIONS
        )

    @cached_property
    def parameter_index(self) -> dict[tuple[int, int, str], int]:
        return {key: index for index, key in enumerate(self.parameter_keys)}

    def score_range(self) -> range:
        margin = self.config.horizon + self.config.score_margin
        return range(-margin, self.config.threshold + margin + 1)

    def materialize(self) -> None:
        for remaining in range(1, self.config.horizon + 1):
            for score in self.score_range():
                self._ensure(score, remaining)

    def _initial_probabilities(self, score: int, remaining: int) -> dict[str, float]:
        gap = self.config.threshold - score
        if gap >= 2:
            probs = {"help": 0.56, "harm": 0.15, "delay": 0.23, "null": 0.06}
        elif gap == 1:
            probs = {"help": 0.45, "harm": 0.17, "delay": 0.31, "null": 0.07}
        elif gap == 0:
            probs = {"help": 0.25, "harm": 0.17, "delay": 0.50, "null": 0.08}
        else:
            probs = {"help": 0.18, "harm": 0.14, "delay": 0.58, "null": 0.10}
        if remaining <= 2 and gap > 0:
            probs["help"] += 0.08
            probs["delay"] -= 0.08
        return normalize(probs)

    def _ensure(self, score: int, remaining: int) -> dict[str, float]:
        key = (score, remaining)
        if key not in self.logits:
            probs = self._initial_probabilities(score, remaining)
            self.logits[key] = {
                action: math.log(max(probs[action], 1e-9)) for action in ACTIONS
            }
        return self.logits[key]

    def probabilities(self, score: int, remaining: int) -> dict[str, float]:
        logits = self._ensure(score, remaining)
        max_logit = max(logits.values())
        exp_values = {
            action: math.exp(logit - max_logit) for action, logit in logits.items()
        }
        total = sum(exp_values.values())
        return {action: value / total for action, value in exp_values.items()}

    def sample_action(self, rng: random.Random, score: int, remaining: int) -> str:
        roll = rng.random()
        cumulative = 0.0
        for action, probability in self.probabilities(score, remaining).items():
            cumulative += probability
            if roll <= cumulative:
                return action
        return ACTIONS[-1]

    def shifted_reference(self, *, help_shift: float = -0.45) -> "TabularSoftmaxPolicy":
        """Return a deterministic fixed reference at a nonzero KL distance."""

        reference = self.copy()
        for action_logits in reference.logits.values():
            action_logits["help"] += help_shift
            action_logits["delay"] -= help_shift * 0.55
            action_logits["harm"] -= help_shift * 0.25
            action_logits["null"] -= help_shift * 0.20
        return reference


def normalize(probs: dict[str, float]) -> dict[str, float]:
    clipped = {key: max(1e-6, value) for key, value in probs.items()}
    total = sum(clipped.values())
    return {key: value / total for key, value in clipped.items()}


def terminal_reward(config: MDPConfig, score: int) -> float:
    return 1.0 if score >= config.threshold else 0.0


def transition(config: MDPConfig, score: int, remaining: int, action: str) -> tuple[int, int, float]:
    if action == "null":
        return score, remaining, 0.0
    if action == "help":
        next_score = score + 1
    elif action == "harm":
        next_score = score - 1
    elif action == "delay":
        next_score = score
    else:
        raise ValueError(f"unknown action {action!r}")
    return next_score, remaining - 1, -config.token_cost


def exact_value(policy: TabularSoftmaxPolicy, score: int, remaining: int) -> float:
    def value(inner_score: int, inner_remaining: int) -> float:
        if inner_remaining <= 0:
            return terminal_reward(policy.config, inner_score)
        key = (inner_score, inner_remaining)
        if key in policy._value_cache:
            return policy._value_cache[key]
        probs = policy.probabilities(inner_score, inner_remaining)
        non_null_mass = 1.0 - probs["null"]
        if non_null_mass <= 1e-12:
            raise ValueError("null action probability is too high to solve value")
        weighted = 0.0
        for action in NON_NULL_ACTIONS:
            next_score, next_remaining, reward = transition(
                policy.config,
                inner_score,
                inner_remaining,
                action,
            )
            weighted += probs[action] * (reward + value(next_score, next_remaining))
        policy._value_cache[key] = weighted / non_null_mass
        return policy._value_cache[key]

    return value(score, remaining)


def exact_advantage(
    policy: TabularSoftmaxPolicy,
    score: int,
    remaining: int,
    action: str,
) -> float:
    next_score, next_remaining, reward = transition(policy.config, score, remaining, action)
    return reward + exact_value(policy, next_score, next_remaining) - exact_value(
        policy,
        score,
        remaining,
    )


def exact_start_return(policy: TabularSoftmaxPolicy) -> float:
    return exact_value(policy, score=0, remaining=policy.config.horizon)


def exact_policy_gradient(policy: TabularSoftmaxPolicy) -> Gradient:
    """Compute the exact finite-MDP policy gradient from occupancy measures."""

    values = [0.0 for _ in policy.parameter_keys]
    incoming: dict[tuple[int, int], float] = defaultdict(float)
    incoming[(0, policy.config.horizon)] = 1.0
    index = policy.parameter_index

    for remaining in range(policy.config.horizon, 0, -1):
        for score in policy.score_range():
            incoming_mass = incoming[(score, remaining)]
            if incoming_mass <= 1e-15:
                continue
            probs = policy.probabilities(score, remaining)
            non_null_mass = 1.0 - probs["null"]
            if non_null_mass <= 1e-12:
                raise ValueError("null action probability is too high for occupancy solve")
            expected_visits = incoming_mass / non_null_mass
            for action in ACTIONS:
                advantage = exact_advantage(policy, score, remaining, action)
                weight = expected_visits * probs[action] * advantage
                if abs(weight) <= 1e-15:
                    continue
                for candidate in ACTIONS:
                    score_grad = (1.0 if candidate == action else 0.0) - probs[candidate]
                    values[index[(score, remaining, candidate)]] += weight * score_grad
            for action in NON_NULL_ACTIONS:
                next_score, next_remaining, _ = transition(
                    policy.config,
                    score,
                    remaining,
                    action,
                )
                incoming[(next_score, next_remaining)] += expected_visits * probs[action]

    return Gradient(tuple(values))


def finite_difference_policy_gradient(
    policy: TabularSoftmaxPolicy,
    eps: float = 1e-5,
) -> Gradient:
    """Numerical check for the occupancy-measure gradient."""

    keys = policy.parameter_keys
    values: list[float] = []
    for score, remaining, action in keys:
        plus = policy.copy()
        plus.logits[(score, remaining)][action] += eps
        minus = policy.copy()
        minus.logits[(score, remaining)][action] -= eps
        values.append((exact_start_return(plus) - exact_start_return(minus)) / (2.0 * eps))
    return Gradient(tuple(values))


def grad_log_prob(policy: TabularSoftmaxPolicy, score: int, remaining: int, action: str) -> Gradient:
    probs = policy.probabilities(score, remaining)
    entries = []
    for key_score, key_remaining, key_action in policy.parameter_keys:
        if key_score == score and key_remaining == remaining:
            entries.append((1.0 if key_action == action else 0.0) - probs[key_action])
        else:
            entries.append(0.0)
    return Gradient(tuple(entries))


def add_grad(lhs: list[float], rhs: Gradient, scale: float) -> None:
    for index, value in enumerate(rhs.values):
        lhs[index] += scale * value


def add_score_function_gradient(
    lhs: list[float],
    policy: TabularSoftmaxPolicy,
    score: int,
    remaining: int,
    action: str,
    scale: float,
) -> None:
    probs = policy.probabilities(score, remaining)
    index = policy.parameter_index
    for candidate in ACTIONS:
        grad = (1.0 if candidate == action else 0.0) - probs[candidate]
        lhs[index[(score, remaining, candidate)]] += scale * grad


def simulate_trajectory(
    rng: random.Random,
    policy: TabularSoftmaxPolicy,
    *,
    max_null_multiplier: int = 12,
) -> Trajectory:
    score = 0
    remaining = policy.config.horizon
    steps: list[Step] = []
    null_guard = max_null_multiplier * policy.config.horizon
    emitted = 0
    rewards: list[float] = []
    while remaining > 0:
        emitted += 1
        action = policy.sample_action(rng, score, remaining)
        next_score, next_remaining, reward = transition(policy.config, score, remaining, action)
        steps.append(
            Step(
                position=emitted,
                score=score,
                remaining=remaining,
                action=action,
                next_score=next_score,
                next_remaining=next_remaining,
                reward=reward,
            )
        )
        rewards.append(reward)
        score, remaining = next_score, next_remaining
        if emitted >= null_guard and remaining > 0:
            next_score, next_remaining, reward = transition(
                policy.config,
                score,
                remaining,
                "delay",
            )
            steps.append(
                Step(
                    position=emitted + 1,
                    score=score,
                    remaining=remaining,
                    action="delay",
                    next_score=next_score,
                    next_remaining=next_remaining,
                    reward=reward,
                )
            )
            rewards.append(reward)
            score, remaining = next_score, next_remaining
            emitted = 0

    terminal = terminal_reward(policy.config, score)
    total = terminal + sum(rewards)
    enriched: list[Step] = []
    future = terminal
    for step in reversed(steps):
        future += step.reward
        enriched.append(
            Step(
                position=step.position,
                score=step.score,
                remaining=step.remaining,
                action=step.action,
                next_score=step.next_score,
                next_remaining=step.next_remaining,
                reward=step.reward,
                return_to_go=future,
            )
        )
    enriched.reverse()
    return Trajectory(tuple(enriched), terminal, total)


def sample_batch(
    rng: random.Random,
    policy: TabularSoftmaxPolicy,
    *,
    groups_per_batch: int,
    group_size: int,
) -> list[list[Trajectory]]:
    return [
        [simulate_trajectory(rng, policy) for _ in range(group_size)]
        for _ in range(groups_per_batch)
    ]


def flatten(groups: Iterable[Iterable[Trajectory]]) -> list[Trajectory]:
    return [trajectory for group in groups for trajectory in group]


class LearnedValueTable:
    """Cross-fitted tabular value model for the exact-gradient audit."""

    def __init__(self, trajectories: Iterable[Trajectory]):
        by_state: dict[tuple[int, int], list[float]] = defaultdict(list)
        by_remaining: dict[int, list[float]] = defaultdict(list)
        all_values: list[float] = []

        for trajectory in trajectories:
            terminal_state = trajectory.steps[-1] if trajectory.steps else None
            if terminal_state is not None:
                by_state[(terminal_state.next_score, 0)].append(trajectory.terminal_reward)
                by_remaining[0].append(trajectory.terminal_reward)
                all_values.append(trajectory.terminal_reward)
            for step in trajectory.steps:
                by_state[(step.score, step.remaining)].append(step.return_to_go)
                by_remaining[step.remaining].append(step.return_to_go)
                all_values.append(step.return_to_go)

        self.by_state = {key: fmean(values) for key, values in by_state.items()}
        self.by_remaining = {key: fmean(values) for key, values in by_remaining.items()}
        self.global_mean = fmean(all_values) if all_values else 0.0

    def value(self, config: MDPConfig, score: int, remaining: int) -> float:
        if remaining <= 0:
            return terminal_reward(config, score)
        key = (score, remaining)
        if key in self.by_state:
            return self.by_state[key]
        if remaining in self.by_remaining:
            return self.by_remaining[remaining]
        return self.global_mean

    def has_state(self, score: int, remaining: int) -> bool:
        if remaining <= 0:
            return True
        return (score, remaining) in self.by_state


def vimpo_signal(
    policy: TabularSoftmaxPolicy,
    reference: TabularSoftmaxPolicy,
    score: int,
    remaining: int,
    action: str,
    beta: float | None = None,
    approximate_kl: bool = False,
) -> float:
    beta = policy.config.beta if beta is None else beta
    probs = policy.probabilities(score, remaining)
    ref_probs = reference.probabilities(score, remaining)
    log_ratio = math.log(max(probs[action], 1e-12) / max(ref_probs[action], 1e-12))
    if approximate_kl:
        kl = 0.5 * sum(
            probs[candidate]
            * (
                math.log(max(probs[candidate], 1e-12) / max(ref_probs[candidate], 1e-12))
                ** 2
            )
            for candidate in ACTIONS
        )
    else:
        kl = sum(
            probs[candidate]
            * math.log(max(probs[candidate], 1e-12) / max(ref_probs[candidate], 1e-12))
            for candidate in ACTIONS
        )
    return beta * (log_ratio - kl)


def prefix_budget_baseline(policy: TabularSoftmaxPolicy, remaining: int) -> float:
    """A cheap budget-only baseline, analogous to a coarse prefix verifier."""

    scores = list(policy.score_range())
    return fmean(exact_value(policy, score, remaining) for score in scores)


def score_norm_squared(policy: TabularSoftmaxPolicy, score: int, remaining: int, action: str) -> float:
    probs = policy.probabilities(score, remaining)
    return sum(
        ((1.0 if candidate == action else 0.0) - probs[candidate]) ** 2
        for candidate in ACTIONS
    )


def estimate_gradient_for_batch(
    method: str,
    groups: list[list[Trajectory]],
    policy: TabularSoftmaxPolicy,
    reference: TabularSoftmaxPolicy | None = None,
    learned_critic: LearnedValueTable | None = None,
) -> tuple[Gradient, list[tuple[float, float]], dict[str, float]]:
    gradient_values = [0.0 for _ in policy.parameter_keys]
    estimate_target_pairs: list[tuple[float, float]] = []
    trajectories = flatten(groups)
    trajectory_count = max(1, len(trajectories))

    group_loo: dict[int, float] = {}
    trajectory_index = 0
    for group in groups:
        returns = [trajectory.total_return for trajectory in group]
        for local_index, trajectory in enumerate(group):
            if len(group) <= 1:
                baseline = 0.0
            else:
                sibling_returns = [
                    value for index, value in enumerate(returns) if index != local_index
                ]
                baseline = fmean(sibling_returns)
            group_loo[trajectory_index] = trajectory.total_return - baseline
            trajectory_index += 1

    diagnostics = {
        "terminal_consistency_residual": 0.0,
        "critic_state_hit_rate": 0.0,
        "null_credit_sum": 0.0,
        "null_abs_credit_sum": 0.0,
        "null_credit_count": 0.0,
    }
    residuals: list[float] = []
    critic_hits = 0
    critic_queries = 0
    trajectory_index = 0
    for group in groups:
        for trajectory in group:
            for step in trajectory.steps:
                target = exact_advantage(policy, step.score, step.remaining, step.action)
                if method == "reinforce_return":
                    estimate = trajectory.total_return
                elif method == "sibling_loo_return":
                    estimate = group_loo[trajectory_index]
                elif method == "prefix_value_baseline":
                    estimate = step.return_to_go - prefix_budget_baseline(
                        policy,
                        step.remaining,
                    )
                elif method == "brpo_combined_baseline":
                    estimate = step.return_to_go - 0.5 * (
                        prefix_budget_baseline(policy, step.remaining)
                        + (trajectory.total_return - group_loo[trajectory_index])
                    )
                elif method == "oracle_value_td":
                    estimate = target
                elif method == "learned_value_td":
                    if learned_critic is None:
                        raise ValueError("learned_value_td requires a learned critic")
                    current_value = learned_critic.value(
                        policy.config,
                        step.score,
                        step.remaining,
                    )
                    next_value = learned_critic.value(
                        policy.config,
                        step.next_score,
                        step.next_remaining,
                    )
                    estimate = step.reward + next_value - current_value
                    critic_queries += 1
                    if learned_critic.has_state(step.score, step.remaining):
                        critic_hits += 1
                elif method.startswith("vimpo_actor_"):
                    if reference is None:
                        raise ValueError("VIMPO methods require a reference policy")
                    estimate = vimpo_signal(policy, reference, step.score, step.remaining, step.action)
                    residuals.append(abs(estimate - target))
                else:
                    raise ValueError(f"unknown method {method!r}")
                if step.action == "null":
                    diagnostics["null_credit_sum"] += estimate
                    diagnostics["null_abs_credit_sum"] += abs(estimate)
                    diagnostics["null_credit_count"] += 1.0
                add_score_function_gradient(
                    gradient_values,
                    policy,
                    step.score,
                    step.remaining,
                    step.action,
                    estimate / trajectory_count,
                )
                estimate_target_pairs.append((estimate, target))
            trajectory_index += 1
    if residuals:
        diagnostics["terminal_consistency_residual"] = fmean(residuals)
    if critic_queries:
        diagnostics["critic_state_hit_rate"] = critic_hits / critic_queries
    return Gradient(tuple(gradient_values)), estimate_target_pairs, diagnostics


def baseline_diagnostic_rows(
    groups: list[list[Trajectory]],
    policy: TabularSoftmaxPolicy,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for group in groups:
        returns = [trajectory.total_return for trajectory in group]
        for local_index, trajectory in enumerate(group):
            if len(group) <= 1:
                sibling_baseline = 0.0
            else:
                sibling_baseline = fmean(
                    value for index, value in enumerate(returns) if index != local_index
                )
            for step in trajectory.steps:
                prefix_baseline = prefix_budget_baseline(policy, step.remaining)
                critic_baseline = exact_value(policy, step.score, step.remaining)
                baselines = {
                    "group_mean": sibling_baseline,
                    "prefix_budget": prefix_baseline,
                    "brpo_combined": 0.5 * (sibling_baseline + prefix_baseline),
                    "critic_value": critic_baseline,
                }
                psi2 = score_norm_squared(policy, step.score, step.remaining, step.action)
                for name, baseline in baselines.items():
                    rows.append(
                        {
                            "baseline": name,
                            "position": step.position,
                            "return_to_go": step.return_to_go,
                            "baseline_value": baseline,
                            "score_norm_squared": psi2,
                        }
                    )
    return rows


def variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_value = fmean(values)
    return fmean((value - mean_value) ** 2 for value in values)


def summarize_baseline_rows(rows: list[dict[str, float | int | str]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        grouped.setdefault((str(row["baseline"]), "all"), []).append(row)
        grouped.setdefault(
            (str(row["baseline"]), f"pos_{int(row['position'])}"),
            [],
        ).append(row)

    summaries: list[dict[str, Any]] = []
    for (baseline, position), bucket in sorted(grouped.items()):
        returns = [float(row["return_to_go"]) for row in bucket]
        baselines = [float(row["baseline_value"]) for row in bucket]
        residuals = [
            float(row["return_to_go"]) - float(row["baseline_value"]) for row in bucket
        ]
        psi2 = [float(row["score_norm_squared"]) for row in bucket]
        return_var = variance(returns)
        summaries.append(
            {
                "baseline": baseline,
                "position": position,
                "count": len(bucket),
                "corr_baseline_return": pearson(list(zip(baselines, returns))),
                "residual_variance_ratio": (
                    variance(residuals) / return_var if return_var > 1e-15 else 0.0
                ),
                "score_weighted_residual_second_moment": fmean(
                    residual * residual * weight
                    for residual, weight in zip(residuals, psi2)
                ),
            }
        )
    return {
        "summary_rows": summaries,
        "overall": {
            row["baseline"]: row
            for row in summaries
            if row["position"] == "all"
        },
    }


def mean_gradient(gradients: list[Gradient]) -> Gradient:
    if not gradients:
        raise ValueError("expected at least one gradient")
    size = len(gradients[0].values)
    return Gradient(
        tuple(fmean(gradient.values[index] for gradient in gradients) for index in range(size))
    )


def variance_trace(gradients: list[Gradient], mean: Gradient) -> float:
    if not gradients:
        return 0.0
    return fmean(gradient.minus(mean).norm ** 2 for gradient in gradients)


def ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def pearson(pairs: list[tuple[float, float]]) -> float:
    if not pairs:
        return 0.0
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    mean_x = fmean(xs)
    mean_y = fmean(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    var_x = sum(value * value for value in centered_x)
    var_y = sum(value * value for value in centered_y)
    if var_x <= 1e-15 or var_y <= 1e-15:
        return 0.0
    return sum(x * y for x, y in zip(centered_x, centered_y)) / math.sqrt(var_x * var_y)


def calibrated_mse(pairs: list[tuple[float, float]]) -> float:
    if not pairs:
        return 0.0
    estimates = [pair[0] for pair in pairs]
    targets = [pair[1] for pair in pairs]
    mean_estimate = fmean(estimates)
    mean_target = fmean(targets)
    centered_estimates = [value - mean_estimate for value in estimates]
    centered_targets = [value - mean_target for value in targets]
    var_estimate = sum(value * value for value in centered_estimates)
    slope = 0.0
    if var_estimate > 1e-15:
        slope = sum(x * y for x, y in zip(centered_estimates, centered_targets)) / var_estimate
    intercept = mean_target - slope * mean_estimate
    return fmean(
        (slope * estimate + intercept - target) ** 2
        for estimate, target in zip(estimates, targets)
    )


def policy_kl(
    source: TabularSoftmaxPolicy,
    target: TabularSoftmaxPolicy,
    *,
    state_filter: Callable[[int, int], bool] | None = None,
) -> float:
    values: list[float] = []
    for remaining in range(1, source.config.horizon + 1):
        for score in source.score_range():
            if state_filter and not state_filter(score, remaining):
                continue
            source_probs = source.probabilities(score, remaining)
            target_probs = target.probabilities(score, remaining)
            values.append(
                sum(
                    source_probs[action]
                    * math.log(
                        max(source_probs[action], 1e-12)
                        / max(target_probs[action], 1e-12)
                    )
                    for action in ACTIONS
                )
            )
    return fmean(values) if values else 0.0


def stepped_policy(policy: TabularSoftmaxPolicy, direction: Gradient, scale: float) -> TabularSoftmaxPolicy:
    new_policy = policy.copy()
    for (score, remaining, action), delta in zip(policy.parameter_keys, direction.values):
        new_policy.logits[(score, remaining)][action] += scale * delta
    return new_policy


def matched_kl_improvement(policy: TabularSoftmaxPolicy, direction: Gradient) -> float:
    if direction.norm <= 1e-15:
        return 0.0
    base_return = exact_start_return(policy)
    target_kl = policy.config.target_kl
    high = 1.0
    for _ in range(30):
        if policy_kl(policy, stepped_policy(policy, direction, high)) >= target_kl:
            break
        high *= 2.0
    low = 0.0
    for _ in range(40):
        mid = 0.5 * (low + high)
        if policy_kl(policy, stepped_policy(policy, direction, mid)) <= target_kl:
            low = mid
        else:
            high = mid
    return exact_start_return(stepped_policy(policy, direction, low)) - base_return


def summarize_gradient_metrics(
    *,
    method: str,
    gradients: list[Gradient],
    pairs: list[tuple[float, float]],
    exact_gradient_value: Gradient,
    policy: TabularSoftmaxPolicy,
    diagnostics: list[dict[str, float]],
) -> dict[str, float]:
    mean = mean_gradient(gradients)
    exact_norm = exact_gradient_value.norm
    cosine = 0.0
    if mean.norm > 1e-15 and exact_norm > 1e-15:
        cosine = mean.dot(exact_gradient_value) / (mean.norm * exact_norm)
    bias_norm = mean.minus(exact_gradient_value).norm
    return {
        "mean_gradient_norm": mean.norm,
        "relative_mean_error_norm": bias_norm / exact_norm if exact_norm > 0 else 0.0,
        "relative_bias_norm": bias_norm / exact_norm if exact_norm > 0 else 0.0,
        "variance_trace": variance_trace(gradients, mean),
        "gradient_cosine": cosine,
        "matched_kl_improvement": matched_kl_improvement(policy, mean),
        "batch_matched_kl_improvement_mean": 0.0,
        "batch_matched_kl_improvement_p05": 0.0,
        "batch_negative_update_probability": 0.0,
        "advantage_correlation": pearson(pairs),
        "calibrated_advantage_mse": calibrated_mse(pairs),
        "terminal_consistency_residual": (
            fmean(row["terminal_consistency_residual"] for row in diagnostics)
            if diagnostics
            else 0.0
        ),
        "critic_state_hit_rate": (
            fmean(row["critic_state_hit_rate"] for row in diagnostics)
            if diagnostics
            else 0.0
        ),
        "null_credit_mean": (
            sum(row["null_credit_sum"] for row in diagnostics)
            / sum(row["null_credit_count"] for row in diagnostics)
            if diagnostics and sum(row["null_credit_count"] for row in diagnostics) > 0
            else 0.0
        ),
        "null_abs_credit_mean": (
            sum(row["null_abs_credit_sum"] for row in diagnostics)
            / sum(row["null_credit_count"] for row in diagnostics)
            if diagnostics and sum(row["null_credit_count"] for row in diagnostics) > 0
            else 0.0
        ),
    }


def gradient_metrics(
    *,
    method: str,
    gradients: list[Gradient],
    pairs: list[tuple[float, float]],
    exact_gradient_value: Gradient,
    policy: TabularSoftmaxPolicy,
    diagnostics: list[dict[str, float]],
    replication_summaries: list[dict[str, float]],
    compute_batch_stats: bool = True,
) -> dict[str, float]:
    metrics = summarize_gradient_metrics(
        method=method,
        gradients=gradients,
        pairs=pairs,
        exact_gradient_value=exact_gradient_value,
        policy=policy,
        diagnostics=diagnostics,
    )
    for key in [
        "relative_mean_error_norm",
        "gradient_cosine",
        "matched_kl_improvement",
        "advantage_correlation",
        "calibrated_advantage_mse",
    ]:
        values = [row[key] for row in replication_summaries if key in row]
        metrics[f"{key}_ci95"] = ci95(values)
    mean = mean_gradient(gradients)
    variance_terms = [gradient.minus(mean).norm ** 2 for gradient in gradients]
    metrics["variance_trace_ci95"] = ci95(variance_terms)
    if compute_batch_stats:
        batch_improvements = [
            matched_kl_improvement(policy, gradient) for gradient in gradients
        ]
        metrics["batch_matched_kl_improvement_mean"] = (
            fmean(batch_improvements) if batch_improvements else 0.0
        )
        metrics["batch_matched_kl_improvement_p05"] = quantile(batch_improvements, 0.05)
        metrics["batch_negative_update_probability"] = (
            sum(1 for value in batch_improvements if value < 0.0) / len(batch_improvements)
            if batch_improvements
            else 0.0
        )
    return metrics


def run_policy_gradient_audit(
    *,
    seed: int = 13,
    batches: int = 200,
    groups_per_batch: int = 16,
    group_size: int = 5,
    replications: int = 200,
    config: MDPConfig | None = None,
) -> dict[str, Any]:
    if batches <= 0 or groups_per_batch <= 0 or group_size <= 1 or replications <= 0:
        raise ValueError("expected positive batches/groups and group_size > 1")
    config = MDPConfig() if config is None else config
    policy = TabularSoftmaxPolicy(config)
    exact = exact_policy_gradient(policy)
    finite_diff = finite_difference_policy_gradient(policy)
    equal_reference = policy.copy()
    references = [
        ("vimpo_actor_equal_ref", equal_reference),
        ("vimpo_actor_fixed_ref_near", policy.shifted_reference(help_shift=-0.12)),
        ("vimpo_actor_fixed_ref_mid", policy.shifted_reference(help_shift=-0.30)),
        ("vimpo_actor_fixed_ref_far", policy.shifted_reference(help_shift=-0.55)),
    ]
    estimator_methods = [
        ("reinforce_return", None),
        ("sibling_loo_return", None),
        ("prefix_value_baseline", None),
        ("brpo_combined_baseline", None),
        ("learned_value_td", None),
        ("oracle_value_td", None),
    ]
    methods = estimator_methods + references
    per_method: dict[str, dict[str, Any]] = {
        name: {"gradients": [], "pairs": [], "diagnostics": [], "replications": []}
        for name, _ in methods
    }

    token_counts: list[int] = []
    null_counts = 0
    delay_counts = 0
    baseline_rows: list[dict[str, float | int | str]] = []
    batches_per_replication = max(1, math.ceil(batches / replications))
    total_batches = batches_per_replication * replications

    for replication in range(replications):
        rng = random.Random(seed + 10_007 * replication)
        rep_method: dict[str, dict[str, Any]] = {
            name: {"gradients": [], "pairs": [], "diagnostics": []}
            for name, _ in methods
        }
        for _ in range(batches_per_replication):
            groups = sample_batch(
                rng,
                policy,
                groups_per_batch=groups_per_batch,
                group_size=group_size,
            )
            critic_groups = sample_batch(
                rng,
                policy,
                groups_per_batch=groups_per_batch,
                group_size=group_size,
            )
            learned_critic = LearnedValueTable(flatten(critic_groups))
            trajectories = flatten(groups)
            token_counts.extend(len(trajectory.steps) for trajectory in trajectories)
            null_counts += sum(
                1
                for trajectory in trajectories
                for step in trajectory.steps
                if step.action == "null"
            )
            delay_counts += sum(
                1
                for trajectory in trajectories
                for step in trajectory.steps
                if step.action == "delay"
            )
            baseline_rows.extend(baseline_diagnostic_rows(groups, policy))
            for method, reference in methods:
                gradient, pairs, diagnostics = estimate_gradient_for_batch(
                    method,
                    groups,
                    policy,
                    reference,
                    learned_critic,
                )
                per_method[method]["gradients"].append(gradient)
                per_method[method]["pairs"].extend(pairs)
                per_method[method]["diagnostics"].append(diagnostics)
                rep_method[method]["gradients"].append(gradient)
                rep_method[method]["pairs"].extend(pairs)
                rep_method[method]["diagnostics"].append(diagnostics)

        for method, _ in methods:
            per_method[method]["replications"].append(
                summarize_gradient_metrics(
                    method=method,
                    gradients=rep_method[method]["gradients"],
                    pairs=rep_method[method]["pairs"],
                    exact_gradient_value=exact,
                    policy=policy,
                    diagnostics=rep_method[method]["diagnostics"],
                )
            )

    def build_rows(
        selected_methods: list[tuple[str, TabularSoftmaxPolicy | None]],
        *,
        compute_batch_stats: bool,
    ) -> list[dict[str, Any]]:
        rows = []
        for method, reference in selected_methods:
            row: dict[str, Any] = {
                "method": method,
                "metrics": gradient_metrics(
                    method=method,
                    gradients=per_method[method]["gradients"],
                    pairs=per_method[method]["pairs"],
                    exact_gradient_value=exact,
                    policy=policy,
                    diagnostics=per_method[method]["diagnostics"],
                    replication_summaries=per_method[method]["replications"],
                    compute_batch_stats=compute_batch_stats,
                ),
            }
            if reference is not None:
                row["reference_kl"] = policy_kl(policy, reference)
            rows.append(row)
        return rows

    estimators = build_rows(estimator_methods, compute_batch_stats=True)
    policy_implied_signals = build_rows(references, compute_batch_stats=False)

    finite_diff_error = finite_diff.minus(exact).norm
    exact_norm = exact.norm

    return {
        "config": {
            **config.__dict__,
            "actions": list(ACTIONS),
            "seed": seed,
            "batches": total_batches,
            "requested_batches": batches,
            "replications": replications,
            "batches_per_replication": batches_per_replication,
            "groups_per_batch": groups_per_batch,
            "group_size": group_size,
        },
        "exact_gradient": {
            "norm": exact.norm,
            "base_return": exact_start_return(policy),
            "target_kl": config.target_kl,
            "finite_difference_error_norm": finite_diff_error,
            "finite_difference_relative_error": (
                finite_diff_error / exact_norm if exact_norm > 0 else 0.0
            ),
        },
        "sample_counts": {
            "trajectories": total_batches * groups_per_batch * group_size,
            "mean_emitted_tokens": fmean(token_counts) if token_counts else 0.0,
            "null_token_fraction": null_counts / sum(token_counts) if token_counts else 0.0,
            "delay_token_fraction": delay_counts / sum(token_counts) if token_counts else 0.0,
        },
        "reference_diagnostics": {
            "equal_reference_kl": policy_kl(policy, equal_reference),
            **{
                f"{method}_kl": policy_kl(policy, reference)
                for method, reference in references
            },
        },
        "position_diagnostics": summarize_baseline_rows(baseline_rows),
        "estimators": estimators,
        "policy_implied_signals": policy_implied_signals,
    }


def fmt(value: float) -> str:
    return f"{value:.3f}"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Policy-gradient fidelity audit",
        "",
        "Finite MDP with exact value and occupancy-measure policy gradient.",
        "",
        "| Method | Rel. mean err. | 95% CI | Var trace | Cosine | Mean batch dJ | P5 batch dJ | Neg. batch | Null abs A | Adv. r | Cal. MSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["estimators"]:
        metrics = row["metrics"]
        lines.append(
            " | ".join(
                [
                    f"| `{row['method']}`",
                    fmt(metrics["relative_mean_error_norm"]),
                    fmt(metrics["relative_mean_error_norm_ci95"]),
                    fmt(metrics["variance_trace"]),
                    fmt(metrics["gradient_cosine"]),
                    fmt(metrics["batch_matched_kl_improvement_mean"]),
                    fmt(metrics["batch_matched_kl_improvement_p05"]),
                    fmt(metrics["batch_negative_update_probability"]),
                    fmt(metrics["null_abs_credit_mean"]),
                    fmt(metrics["advantage_correlation"]),
                    fmt(metrics["calibrated_advantage_mse"]) + " |",
                ]
            )
        )
    lines.extend(["", "## Policy-implied actor coefficients", ""])
    for row in result["policy_implied_signals"]:
        metrics = row["metrics"]
        lines.append(
            f"- `{row['method']}`: ref KL {fmt(row['reference_kl'])}, "
            f"cos {fmt(metrics['gradient_cosine'])}, "
            f"mean err {fmt(metrics['relative_mean_error_norm'])}"
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            f"- Exact gradient norm: {fmt(result['exact_gradient']['norm'])}",
            f"- Finite-difference relative check error: {fmt(result['exact_gradient']['finite_difference_relative_error'])}",
            f"- Base return: {fmt(result['exact_gradient']['base_return'])}",
            f"- Mean emitted tokens: {fmt(result['sample_counts']['mean_emitted_tokens'])}",
            f"- Null token fraction: {fmt(result['sample_counts']['null_token_fraction'])}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--groups-per-batch", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--token-cost", type=float, default=0.02)
    parser.add_argument("--beta", type=float, default=0.35)
    parser.add_argument("--target-kl", type=float, default=0.003)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_policy_gradient_audit(
        seed=args.seed,
        batches=args.batches,
        groups_per_batch=args.groups_per_batch,
        group_size=args.group_size,
        replications=args.replications,
        config=MDPConfig(
            threshold=args.threshold,
            horizon=args.horizon,
            token_cost=args.token_cost,
            beta=args.beta,
            target_kl=args.target_kl,
        ),
    )
    if args.output_json:
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_md:
        write_markdown(result, args.output_md)


if __name__ == "__main__":
    main()
