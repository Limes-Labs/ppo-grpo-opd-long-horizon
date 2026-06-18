"""Tiny neural value-critic generalization audit.

The tabular experiments know the exact toy state. This audit asks a slightly
harder question: if a critic is a tiny neural function approximator, can it
interpolate temporal credit to an unseen threshold value?

Training uses thresholds 1 and 3. Evaluation uses threshold 2 only. Because the
exact state key includes threshold, all evaluation states are held out from a
tabular exact-state perspective; any value signal must come from feature-level
generalization. The implementation is dependency-free and CPU-only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from experiments.deep_matrix import DEFAULT_SEEDS
from experiments.toy_credit_assignment import (
    SCENARIOS,
    Scenario,
    TabularCritic,
    Trajectory,
    add_group_relative_advantages,
    estimator_metrics,
    flatten,
    generate_groups,
    resolve_scenario,
    state_key,
    terminal_reward,
)


@dataclass(frozen=True)
class NeuralGeneralizationConfig:
    scenario_name: str = "long_wait"
    train_thresholds: tuple[int, ...] = (1, 3)
    eval_thresholds: tuple[int, ...] = (2,)
    train_groups: int = 60
    eval_groups: int = 16
    group_size: int = 5
    max_steps: int = 10
    hidden_size: int = 8
    epochs: int = 35
    learning_rate: float = 0.02
    max_train_examples: int = 2500


class TinyValueNetwork:
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
        self.w1 = [[rng.uniform(-0.10, 0.10) for _ in range(7)] for _ in range(hidden_size)]
        self.b1 = [0.0 for _ in range(hidden_size)]
        self.w2 = [rng.uniform(-0.10, 0.10) for _ in range(hidden_size)]
        self.b2 = 0.0

    def features(self, threshold: int, score: int, remaining: int) -> list[float]:
        scale = max(1.0, float(max(self.scenario.threshold_cycle)))
        gap = threshold - score
        margin = score - threshold
        return [
            1.0,
            threshold / scale,
            score / (scale + 2.0),
            remaining / max(1.0, self.max_steps),
            gap / (scale + 2.0),
            margin / (scale + 2.0),
            1.0 if score >= threshold else 0.0,
        ]

    def forward(self, threshold: int, score: int, remaining: int) -> tuple[list[float], float]:
        features = self.features(threshold, score, remaining)
        hidden_raw = [
            self.b1[index] + sum(weight * value for weight, value in zip(row, features))
            for index, row in enumerate(self.w1)
        ]
        hidden = [math.tanh(value) for value in hidden_raw]
        output = self.b2 + sum(weight * value for weight, value in zip(self.w2, hidden))
        return hidden, output

    def value(self, threshold: int, score: int, remaining: int) -> float:
        if remaining == 0:
            return terminal_reward(threshold, score)
        _, output = self.forward(threshold, score, remaining)
        return output

    def fit(self, examples: list[tuple[int, int, int, float]]) -> float:
        if not examples:
            return 0.0
        for _ in range(self.epochs):
            for threshold, score, remaining, target in examples:
                self.update(threshold, score, remaining, target)
        return fmean(
            (self.value(threshold, score, remaining) - target) ** 2
            for threshold, score, remaining, target in examples
        )

    @property
    def epochs(self) -> int:
        return getattr(self, "_epochs", 1)

    @epochs.setter
    def epochs(self, value: int) -> None:
        self._epochs = value

    @property
    def learning_rate(self) -> float:
        return getattr(self, "_learning_rate", 0.02)

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self._learning_rate = value

    def update(self, threshold: int, score: int, remaining: int, target: float) -> None:
        features = self.features(threshold, score, remaining)
        hidden, prediction = self.forward(threshold, score, remaining)
        delta = target - prediction
        old_w2 = self.w2[:]

        self.b2 += self.learning_rate * delta
        for hidden_index, hidden_value in enumerate(hidden):
            self.w2[hidden_index] += self.learning_rate * delta * hidden_value

        for hidden_index, hidden_value in enumerate(hidden):
            local = delta * old_w2[hidden_index] * (1.0 - hidden_value * hidden_value)
            self.b1[hidden_index] += self.learning_rate * local
            for feature_index, feature_value in enumerate(features):
                self.w1[hidden_index][feature_index] += (
                    self.learning_rate * local * feature_value
                )


def scenario_with_thresholds(base: Scenario, thresholds: tuple[int, ...]) -> Scenario:
    return replace(base, threshold_cycle=thresholds)


def value_examples(
    trajectories: list[Trajectory],
    rng: random.Random,
    max_examples: int,
) -> list[tuple[int, int, int, float]]:
    examples = []
    for trajectory in trajectories:
        examples.append(
            (
                trajectory.threshold,
                trajectory.final_score,
                0,
                trajectory.terminal_reward,
            )
        )
        for step in trajectory.steps:
            examples.append(
                (
                    step.threshold,
                    step.start_score,
                    step.remaining_before,
                    step.return_to_go,
                )
            )
    if max_examples > 0 and len(examples) > max_examples:
        examples = rng.sample(examples, max_examples)
    return examples


def add_neural_advantages(
    trajectories: Iterable[Trajectory],
    value_model: TinyValueNetwork,
) -> None:
    for trajectory in trajectories:
        for step in trajectory.steps:
            current_value = value_model.value(
                step.threshold,
                step.start_score,
                step.remaining_before,
            )
            next_value = value_model.value(
                step.threshold,
                step.next_score,
                step.remaining_after,
            )
            step.critic_advantage = step.step_reward + next_value - current_value


def run_one_seed(seed: int, config: NeuralGeneralizationConfig) -> dict[str, Any]:
    base = resolve_scenario(config.scenario_name)
    train_scenario = scenario_with_thresholds(base, config.train_thresholds)
    eval_scenario = scenario_with_thresholds(base, config.eval_thresholds)
    train_rng = random.Random(seed)
    train_groups_data = generate_groups(
        train_rng,
        group_count=config.train_groups,
        group_size=config.group_size,
        max_steps=config.max_steps,
        scenario=train_scenario,
    )
    train_trajectories = flatten(train_groups_data)

    exact_train_critic = TabularCritic(
        train_trajectories,
        observation=train_scenario.critic_observation,
    )
    value_model = TinyValueNetwork(
        scenario=scenario_with_thresholds(
            base,
            tuple(sorted(set(config.train_thresholds + config.eval_thresholds))),
        ),
        max_steps=config.max_steps,
        hidden_size=config.hidden_size,
        rng=random.Random(seed + 7_001),
    )
    value_model.epochs = config.epochs
    value_model.learning_rate = config.learning_rate
    train_mse = value_model.fit(
        value_examples(
            train_trajectories,
            random.Random(seed + 7_002),
            config.max_train_examples,
        )
    )

    eval_rng = random.Random(seed + 1)
    eval_groups_data = generate_groups(
        eval_rng,
        group_count=config.eval_groups,
        group_size=config.group_size,
        max_steps=config.max_steps,
        scenario=eval_scenario,
        trajectory_offset=len(train_trajectories),
    )
    add_group_relative_advantages(eval_groups_data)
    eval_trajectories = flatten(eval_groups_data)
    add_neural_advantages(eval_trajectories, value_model)

    eval_steps = [step for trajectory in eval_trajectories for step in trajectory.steps]
    heldout_count = sum(
        1
        for step in eval_steps
        if not exact_train_critic.has_exact_state(
            step.threshold,
            step.start_score,
            step.remaining_before,
        )
    )

    return {
        "seed": seed,
        "train_value_mse": train_mse,
        "estimators": {
            "group_relative": estimator_metrics(
                eval_trajectories,
                lambda step: step.group_advantage,
            ),
            "neural_critic_td": estimator_metrics(
                eval_trajectories,
                lambda step: step.critic_advantage,
            ),
        },
        "sample_counts": {
            "train_trajectories": len(train_trajectories),
            "eval_trajectories": len(eval_trajectories),
            "eval_tokens": len(eval_steps),
            "heldout_exact_state_fraction": (
                heldout_count / len(eval_steps) if eval_steps else 0.0
            ),
        },
    }


def mean_metric(seed_results: list[dict[str, Any]], estimator: str, metric: str) -> float:
    return fmean(row["estimators"][estimator][metric] for row in seed_results)


def aggregate(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    estimators = {}
    for estimator in ["group_relative", "neural_critic_td"]:
        estimators[estimator] = {
            metric: mean_metric(seed_results, estimator, metric)
            for metric in [
                "pearson_correlation",
                "calibrated_mse",
                "raw_mse",
                "sign_accuracy",
                "mean_abs_wait_tokens",
                "mean_abs_active_tokens",
                "wait_to_active_abs_ratio",
                "within_trajectory_variance",
            ]
        }
    sample_counts = {
        key: fmean(row["sample_counts"][key] for row in seed_results)
        for key in [
            "train_trajectories",
            "eval_trajectories",
            "eval_tokens",
            "heldout_exact_state_fraction",
        ]
    }
    return {
        "estimators": estimators,
        "sample_counts": sample_counts,
        "train_value_mse": fmean(row["train_value_mse"] for row in seed_results),
        "neural_minus_group_correlation": (
            estimators["neural_critic_td"]["pearson_correlation"]
            - estimators["group_relative"]["pearson_correlation"]
        ),
    }


def run_neural_generalization(
    *,
    seeds: list[int] | None = None,
    config: NeuralGeneralizationConfig | None = None,
) -> dict[str, Any]:
    seeds = list(DEFAULT_SEEDS[:5] if seeds is None else seeds)
    config = NeuralGeneralizationConfig() if config is None else config
    if not seeds:
        raise ValueError("expected at least one seed")
    if not config.train_thresholds or not config.eval_thresholds:
        raise ValueError("expected train and eval thresholds")
    seed_results = [run_one_seed(seed, config) for seed in seeds]
    return {
        "config": {
            **config.__dict__,
            "train_thresholds": list(config.train_thresholds),
            "eval_thresholds": list(config.eval_thresholds),
            "seeds": seeds,
        },
        "aggregate": aggregate(seed_results),
        "seed_results": seed_results,
    }


def fmt(value: float) -> str:
    return f"{value:.3f}"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    group = result["aggregate"]["estimators"]["group_relative"]
    neural = result["aggregate"]["estimators"]["neural_critic_td"]
    counts = result["aggregate"]["sample_counts"]
    lines = [
        "# Tiny Neural Generalization Audit",
        "",
        "The neural value critic is trained on thresholds 1 and 3, then evaluated",
        "on held-out threshold 2. Exact tabular state lookup cannot help on the",
        "evaluation threshold; the value model must generalize from features.",
        "",
        "| Estimator | Pearson r | Cal. MSE | Sign | Wait leak | Within var |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            "| Group relative | "
            f"{fmt(group['pearson_correlation'])} | "
            f"{group['calibrated_mse']:.4f} | "
            f"{fmt(group['sign_accuracy'])} | "
            f"{fmt(group['wait_to_active_abs_ratio'])} | "
            f"{fmt(group['within_trajectory_variance'])} |"
        ),
        (
            "| Neural critic TD | "
            f"{fmt(neural['pearson_correlation'])} | "
            f"{neural['calibrated_mse']:.4f} | "
            f"{fmt(neural['sign_accuracy'])} | "
            f"{fmt(neural['wait_to_active_abs_ratio'])} | "
            f"{fmt(neural['within_trajectory_variance'])} |"
        ),
        "",
        "Summary:",
        f"- Held-out exact-state fraction: {fmt(counts['heldout_exact_state_fraction'])}",
        f"- Neural minus group Pearson r: {fmt(result['aggregate']['neural_minus_group_correlation'])}",
        f"- Train value MSE: {result['aggregate']['train_value_mse']:.4f}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="long_wait")
    parser.add_argument("--train-thresholds", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--eval-thresholds", type=int, nargs="+", default=[2])
    parser.add_argument("--train-groups", type=int, default=60)
    parser.add_argument("--eval-groups", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--max-train-examples", type=int, default=2500)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/neural_credit_generalization_seedset.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/neural_credit_generalization_seedset.md"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base = resolve_scenario(args.scenario)
    config = NeuralGeneralizationConfig(
        scenario_name=base.name,
        train_thresholds=tuple(args.train_thresholds),
        eval_thresholds=tuple(args.eval_thresholds),
        train_groups=args.train_groups,
        eval_groups=args.eval_groups,
        group_size=args.group_size,
        max_steps=args.max_steps,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_train_examples=args.max_train_examples,
    )
    result = run_neural_generalization(seeds=args.seeds, config=config)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    write_markdown(result, args.output_md)
    group = result["aggregate"]["estimators"]["group_relative"]
    neural = result["aggregate"]["estimators"]["neural_critic_td"]
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    print(
        f"neural_r={neural['pearson_correlation']:.6f} "
        f"group_r={group['pearson_correlation']:.6f} "
        f"heldout={result['aggregate']['sample_counts']['heldout_exact_state_fraction']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
