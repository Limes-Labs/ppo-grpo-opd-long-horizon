"""Closed-loop autoregressive sequence-policy audit.

The tabular closed-loop audit updates a separate softmax table for every exact
toy state. This file removes that assumption while staying CPU-only and
dependency-free: the policy is a shared one-hidden-layer MLP that conditions on
prompt and prefix-derived sequence state, then samples the next action token.

Compared methods:

- ``group_broadcast``: z-scored terminal trajectory return, broadcast to every
  generated token in the rollout group.
- ``neural_value_td``: a tiny learned value critic trained on current/replay
  returns-to-go, used as a one-step TD advantage.

This is still a synthetic mechanism experiment, not an LLM benchmark. Its job
is narrower: check whether the paper's tabular closed-loop result survives a
shared-parameter autoregressive policy where credit affects learned sequence
behavior rather than exact-state table entries.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable

from experiments.closed_loop_credit_training import (
    broadcast_group_values,
    evaluate_policy,
    flatten,
    generate_policy_groups,
    normalize_step_values,
)
from experiments.deep_matrix import DEFAULT_SEEDS
from experiments.neural_credit_generalization import TinyValueNetwork, value_examples
from experiments.toy_credit_assignment import (
    ACTIONS,
    SCENARIOS,
    Scenario,
    Trajectory,
    resolve_scenario,
)


DEFAULT_METHODS = ["group_broadcast", "neural_value_td"]

T_CRITICAL_975 = {
    1: 12.706204736432095,
    2: 4.302652729749464,
    3: 3.182446305284263,
    4: 2.7764451051977987,
    5: 2.570581835636314,
    6: 2.4469118511449692,
    7: 2.3646242510102993,
    8: 2.306004135204166,
    9: 2.2621571627409915,
    10: 2.2281388519649385,
    11: 2.200985160091638,
    12: 2.178812829663418,
    13: 2.160368656461013,
    14: 2.1447866879169273,
    15: 2.131449545559323,
    16: 2.1199052992210112,
    17: 2.1098155778331806,
    18: 2.10092204024096,
    19: 2.093024054408263,
    20: 2.0859634472658364,
    21: 2.079613844727662,
    22: 2.0738730679040147,
    23: 2.0686576104190406,
    24: 2.0638985616280205,
    25: 2.059538552753294,
    26: 2.055529438642871,
    27: 2.0518305164802833,
    28: 2.048407141795244,
    29: 2.045229642132703,
    30: 2.042272456301238,
}


@dataclass(frozen=True)
class SequencePolicyConfig:
    scenario_name: str = "long_wait"
    train_iterations: int = 12
    groups_per_iteration: int = 8
    group_size: int = 3
    max_steps: int = 8
    eval_groups: int = 60
    policy_hidden_size: int = 6
    critic_hidden_size: int = 6
    learning_rate: float = 0.02
    entropy_bonus: float = 0.002
    critic_learning_rate: float = 0.018
    critic_epochs: int = 3
    critic_replay_limit: int = 1200
    max_critic_examples: int = 300
    eval_every: int = 12


class AutoregressiveMLPPolicy:
    """A tiny sequence policy over action tokens with manual gradients."""

    def __init__(
        self,
        *,
        scenario: Scenario,
        max_steps: int,
        hidden_size: int,
        rng: random.Random,
    ):
        self.scenario = scenario
        self.max_steps = max_steps
        self.hidden_size = hidden_size
        self.feature_count = 8
        self.w1 = [
            [rng.uniform(-0.08, 0.08) for _ in range(self.feature_count)]
            for _ in range(hidden_size)
        ]
        self.b1 = [0.0 for _ in range(hidden_size)]
        self.w2 = [
            [rng.uniform(-0.08, 0.08) for _ in range(hidden_size)]
            for _ in ACTIONS
        ]
        self.b2 = [0.15, -0.15, 0.0]

    def features(self, threshold: int, score: int, remaining: int) -> list[float]:
        scale = max(1.0, float(max(self.scenario.threshold_cycle)))
        elapsed = self.max_steps - remaining
        gap = threshold - score
        margin = score - threshold
        return [
            1.0,
            threshold / scale,
            score / (scale + self.max_steps),
            remaining / max(1.0, self.max_steps),
            elapsed / max(1.0, self.max_steps),
            gap / (scale + self.max_steps),
            margin / (scale + self.max_steps),
            1.0 if score >= threshold else 0.0,
        ]

    def forward(self, threshold: int, score: int, remaining: int) -> tuple[list[float], list[float]]:
        features = self.features(threshold, score, remaining)
        hidden_raw = [
            self.b1[index] + sum(weight * value for weight, value in zip(row, features))
            for index, row in enumerate(self.w1)
        ]
        hidden = [math.tanh(value) for value in hidden_raw]
        logits = [
            self.b2[action_index]
            + sum(weight * value for weight, value in zip(row, hidden))
            for action_index, row in enumerate(self.w2)
        ]
        return hidden, logits

    def probabilities(
        self,
        threshold: int,
        score: int,
        remaining: int,
    ) -> dict[str, float]:
        _, logits = self.forward(threshold, score, remaining)
        max_logit = max(logits)
        exp_values = [math.exp(value - max_logit) for value in logits]
        total = sum(exp_values)
        return {
            action: exp_values[index] / total for index, action in enumerate(ACTIONS)
        }

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
        features = self.features(threshold, score, remaining)
        hidden, logits = self.forward(threshold, score, remaining)
        max_logit = max(logits)
        exp_values = [math.exp(value - max_logit) for value in logits]
        total = sum(exp_values)
        probs = [value / total for value in exp_values]
        old_w2 = [row[:] for row in self.w2]
        action_index = ACTIONS.index(action)
        clipped = max(-3.0, min(3.0, advantage))

        output_gradients = []
        for index, probability in enumerate(probs):
            score_gradient = (1.0 if index == action_index else 0.0) - probability
            entropy_push = -math.log(max(probability, 1e-12)) - 1.0
            output_gradients.append(
                clipped * score_gradient + entropy_bonus * probability * entropy_push
            )

        for output_index, gradient in enumerate(output_gradients):
            self.b2[output_index] += learning_rate * gradient
            for hidden_index, hidden_value in enumerate(hidden):
                self.w2[output_index][hidden_index] += (
                    learning_rate * gradient * hidden_value
                )

        for hidden_index, hidden_value in enumerate(hidden):
            local = (
                sum(
                    old_w2[output_index][hidden_index] * output_gradients[output_index]
                    for output_index in range(len(ACTIONS))
                )
                * (1.0 - hidden_value * hidden_value)
            )
            self.b1[hidden_index] += learning_rate * local
            for feature_index, feature_value in enumerate(features):
                self.w1[hidden_index][feature_index] += (
                    learning_rate * local * feature_value
                )


def build_sequence_advantages(
    method: str,
    groups: list[list[Trajectory]],
    critic: TinyValueNetwork | None,
) -> tuple[dict[tuple[int, int], float], dict[str, float]]:
    group_values = broadcast_group_values(groups, lambda trajectory: trajectory.total_return)
    if method == "group_broadcast":
        return group_values, {"critic_fit_mse": 0.0, "critic_fraction": 0.0}
    if method != "neural_value_td":
        raise ValueError(f"unknown method {method!r}")
    if critic is None:
        raise ValueError("neural_value_td requires a critic")

    values: dict[tuple[int, int], float] = {}
    for trajectory in flatten(groups):
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
            values[(step.trajectory_id, step.remaining_before)] = (
                step.step_reward + next_value - current_value
            )
    return normalize_step_values(values), {"critic_fraction": 1.0}


def apply_sequence_update(
    policy: AutoregressiveMLPPolicy,
    trajectories: Iterable[Trajectory],
    advantages: dict[tuple[int, int], float],
    config: SequencePolicyConfig,
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


def fit_value_critic(
    *,
    seed: int,
    iteration: int,
    scenario: Scenario,
    trajectories: list[Trajectory],
    config: SequencePolicyConfig,
) -> tuple[TinyValueNetwork, float]:
    critic = TinyValueNetwork(
        scenario=scenario,
        max_steps=config.max_steps,
        hidden_size=config.critic_hidden_size,
        rng=random.Random(seed + 37_000 + iteration),
    )
    critic.epochs = config.critic_epochs
    critic.learning_rate = config.critic_learning_rate
    examples = value_examples(
        trajectories,
        random.Random(seed + 41_000 + iteration),
        config.max_critic_examples,
    )
    return critic, critic.fit(examples)


def train_method(
    *,
    seed: int,
    method: str,
    config: SequencePolicyConfig,
) -> dict[str, Any]:
    scenario = resolve_scenario(config.scenario_name)
    rng = random.Random(seed)
    eval_rng = random.Random(seed + 100_000)
    policy = AutoregressiveMLPPolicy(
        scenario=scenario,
        max_steps=config.max_steps,
        hidden_size=config.policy_hidden_size,
        rng=random.Random(seed + 9_001),
    )
    replay: deque[Trajectory] = deque(maxlen=config.critic_replay_limit)
    trajectory_offset = 0

    initial_eval = evaluate_policy(
        eval_rng,
        policy,
        scenario=scenario,
        config=config,  # type: ignore[arg-type]
        trajectory_offset=1_000_000,
    )
    learning_curve = [
        {
            "iteration": 0,
            "critic_fraction": 0.0,
            "critic_fit_mse": 0.0,
            "eval": initial_eval,
        }
    ]

    for iteration in range(1, config.train_iterations + 1):
        groups = generate_policy_groups(
            rng,
            policy,  # type: ignore[arg-type]
            group_count=config.groups_per_iteration,
            group_size=config.group_size,
            max_steps=config.max_steps,
            scenario=scenario,
            trajectory_offset=trajectory_offset,
        )
        trajectories = flatten(groups)
        trajectory_offset += len(trajectories)

        critic = None
        critic_fit_mse = 0.0
        if method == "neural_value_td":
            critic, critic_fit_mse = fit_value_critic(
                seed=seed,
                iteration=iteration,
                scenario=scenario,
                trajectories=list(replay) + trajectories,
                config=config,
            )

        advantages, diagnostics = build_sequence_advantages(method, groups, critic)
        diagnostics["critic_fit_mse"] = critic_fit_mse
        apply_sequence_update(policy, trajectories, advantages, config)
        replay.extend(trajectories)

        if iteration % config.eval_every == 0 or iteration == config.train_iterations:
            learning_curve.append(
                {
                    "iteration": iteration,
                    "critic_fraction": diagnostics["critic_fraction"],
                    "critic_fit_mse": diagnostics["critic_fit_mse"],
                    "eval": evaluate_policy(
                        eval_rng,
                        policy,
                        scenario=scenario,
                        config=config,  # type: ignore[arg-type]
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
        "final_critic_fit_mse": fmean(
            row["learning_curve"][-1]["critic_fit_mse"] for row in rows
        ),
    }


def t_critical_975(df: int) -> float:
    if df <= 0:
        raise ValueError("df must be positive")
    if df in T_CRITICAL_975:
        return T_CRITICAL_975[df]
    return 1.959963984540054


def paired_final_return_ci(
    runs: list[dict[str, Any]],
    numerator_method: str,
    denominator_method: str,
) -> tuple[int, float, float, float]:
    by_seed: dict[int, dict[str, float]] = {}
    for run in runs:
        seed = int(run["seed"])
        by_seed.setdefault(seed, {})[run["method"]] = float(run["final_eval"]["mean_return"])

    deltas = []
    for seed, methods_by_seed in sorted(by_seed.items()):
        if numerator_method not in methods_by_seed or denominator_method not in methods_by_seed:
            raise ValueError(f"unpaired sequence-policy seed: {seed}")
        deltas.append(methods_by_seed[numerator_method] - methods_by_seed[denominator_method])

    if len(deltas) < 2:
        raise ValueError("paired CI requires at least two paired seeds")
    paired_mean = fmean(deltas)
    half_width = t_critical_975(len(deltas) - 1) * stdev(deltas) / math.sqrt(len(deltas))
    return len(deltas), paired_mean, paired_mean - half_width, paired_mean + half_width


def run_sequence_policy_audit(
    *,
    seeds: list[int] | None = None,
    methods: list[str] | None = None,
    config: SequencePolicyConfig | None = None,
) -> dict[str, Any]:
    seeds = list(DEFAULT_SEEDS[:10] if seeds is None else seeds)
    methods = list(DEFAULT_METHODS if methods is None else methods)
    config = SequencePolicyConfig() if config is None else config
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
    group = next(row for row in summaries if row["method"] == "group_broadcast")
    neural = next(row for row in summaries if row["method"] == "neural_value_td")
    best = max(summaries, key=lambda row: row["final_return"])
    paired_n, paired_delta, paired_low, paired_high = paired_final_return_ci(
        runs,
        numerator_method="neural_value_td",
        denominator_method="group_broadcast",
    )
    return {
        "config": {
            **config.__dict__,
            "seeds": seeds,
            "methods": methods,
            "policy_family": "autoregressive_mlp",
            "critic_family": "one_hidden_layer_value_mlp",
        },
        "method_summaries": summaries,
        "runs": runs,
        "summary": {
            "best_by_final_return": best["method"],
            "paired_seed_count": paired_n,
            "neural_minus_group_return": paired_delta,
            "neural_minus_group_return_ci95": [paired_low, paired_high],
            "neural_minus_group_success": (
                neural["final_success"] - group["final_success"]
            ),
            "neural_minus_group_wait_fraction": (
                neural["final_wait_fraction"] - group["final_wait_fraction"]
            ),
        },
    }


def fmt(value: float) -> str:
    return f"{value:.3f}"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Sequence-Policy Training Audit",
        "",
        "A shared one-hidden-layer autoregressive MLP policy samples action tokens",
        "from prompt and prefix-derived sequence state. The audit compares group",
        "broadcast credit with learned neural value-TD credit under matched rollout",
        "budgets. This is a synthetic sequence-policy mechanism check, not an",
        "LLM-scale transformer benchmark.",
        "",
        "| Method | Init R | Final R | Delta R | Success | Wait frac. | Critic MSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        "group_broadcast": "Group broadcast",
        "neural_value_td": "Neural value TD",
    }
    for row in result["method_summaries"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    labels.get(row["method"], row["method"]),
                    fmt(row["initial_return"]),
                    fmt(row["final_return"]),
                    fmt(row["return_improvement"]),
                    fmt(row["final_success"]),
                    fmt(row["final_wait_fraction"]),
                    fmt(row["final_critic_fit_mse"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Summary:",
            f"- Policy family: {result['config']['policy_family']}",
            f"- {result['summary']['paired_seed_count']} paired seeds",
            f"- Neural minus group final return: {fmt(result['summary']['neural_minus_group_return'])}",
            "- Neural minus group final return 95% paired CI: "
            f"[{fmt(result['summary']['neural_minus_group_return_ci95'][0])}, "
            f"{fmt(result['summary']['neural_minus_group_return_ci95'][1])}]",
            f"- Neural minus group success: {fmt(result['summary']['neural_minus_group_success'])}",
            f"- Neural minus group wait fraction: {fmt(result['summary']['neural_minus_group_wait_fraction'])}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="long_wait")
    parser.add_argument("--methods", nargs="*", choices=DEFAULT_METHODS, default=None)
    parser.add_argument("--train-iterations", type=int, default=12)
    parser.add_argument("--groups-per-iteration", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--eval-groups", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=12)
    parser.add_argument("--policy-hidden-size", type=int, default=6)
    parser.add_argument("--critic-hidden-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--entropy-bonus", type=float, default=0.002)
    parser.add_argument("--critic-learning-rate", type=float, default=0.018)
    parser.add_argument("--critic-epochs", type=int, default=3)
    parser.add_argument("--critic-replay-limit", type=int, default=1200)
    parser.add_argument("--max-critic-examples", type=int, default=300)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/sequence_policy_training_seedset.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/sequence_policy_training_seedset.md"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = SequencePolicyConfig(
        scenario_name=args.scenario,
        train_iterations=args.train_iterations,
        groups_per_iteration=args.groups_per_iteration,
        group_size=args.group_size,
        max_steps=args.max_steps,
        eval_groups=args.eval_groups,
        policy_hidden_size=args.policy_hidden_size,
        critic_hidden_size=args.critic_hidden_size,
        learning_rate=args.learning_rate,
        entropy_bonus=args.entropy_bonus,
        critic_learning_rate=args.critic_learning_rate,
        critic_epochs=args.critic_epochs,
        critic_replay_limit=args.critic_replay_limit,
        max_critic_examples=args.max_critic_examples,
        eval_every=args.eval_every,
    )
    result = run_sequence_policy_audit(
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
    print(
        "neural_minus_group_return="
        f"{result['summary']['neural_minus_group_return']:.6f} "
        "neural_minus_group_success="
        f"{result['summary']['neural_minus_group_success']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
