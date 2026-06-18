"""Run a small pre-registered sweep over toy credit-assignment regimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from experiments.toy_credit_assignment import run_experiment


SWEEP_CASES: list[dict[str, Any]] = [
    {
        "case_name": "short_dense_full_critic",
        "scenario_name": "short_dense",
        "train_groups": 140,
        "eval_groups": 40,
        "group_size": 6,
        "max_steps": 6,
        "interpretation": "Shorter, denser traces reduce the cost of response-level scalar credit.",
    },
    {
        "case_name": "baseline_full_critic",
        "scenario_name": "baseline",
        "train_groups": 140,
        "eval_groups": 40,
        "group_size": 6,
        "max_steps": 10,
        "interpretation": "Baseline mixed horizons with a fully observed tabular value model.",
    },
    {
        "case_name": "long_wait_full_critic",
        "scenario_name": "long_wait",
        "train_groups": 180,
        "eval_groups": 48,
        "group_size": 6,
        "max_steps": 14,
        "interpretation": "Long wait-heavy traces test whether response-level rewards praise no-op tokens.",
    },
    {
        "case_name": "sparse_hard_full_critic",
        "scenario_name": "sparse_hard",
        "train_groups": 220,
        "eval_groups": 48,
        "group_size": 8,
        "max_steps": 14,
        "interpretation": "Sparse success tests whether either estimator has enough signal.",
    },
    {
        "case_name": "coarse_critic_partial_state",
        "scenario_name": "coarse_critic",
        "train_groups": 160,
        "eval_groups": 40,
        "group_size": 6,
        "max_steps": 12,
        "interpretation": "A partially observed critic checks whether value information remains useful.",
    },
    {
        "case_name": "blind_critic_counterexample",
        "scenario_name": "blind_critic",
        "train_groups": 2,
        "eval_groups": 32,
        "group_size": 8,
        "max_steps": 4,
        "interpretation": "A blind, undercovered critic is a counterexample where terminal group outcomes win.",
    },
]


def winner(case_result: dict[str, Any]) -> str:
    metrics = case_result["metrics"]
    group_corr = metrics["group_relative"]["pearson_correlation"]
    critic_corr = metrics["critic_value_model"]["pearson_correlation"]
    if critic_corr > group_corr:
        return "critic"
    if group_corr > critic_corr:
        return "group"
    return "tie"


def run_sweep(seed: int = 11, cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    selected_cases = cases or SWEEP_CASES
    case_results: list[dict[str, Any]] = []
    for index, case in enumerate(selected_cases):
        result = run_experiment(
            seed=seed + index * 101,
            scenario_name=case["scenario_name"],
            train_groups=case["train_groups"],
            eval_groups=case["eval_groups"],
            group_size=case["group_size"],
            max_steps=case["max_steps"],
        )
        case_results.append(
            {
                "case_name": case["case_name"],
                "interpretation": case["interpretation"],
                "config": result["config"],
                "sample_counts": result["sample_counts"],
                "metrics": result["metrics"],
                "winner_by_correlation": winner(result),
            }
        )

    critic_wins = sum(1 for case in case_results if case["winner_by_correlation"] == "critic")
    group_wins = sum(1 for case in case_results if case["winner_by_correlation"] == "group")
    return {
        "seed": seed,
        "case_count": len(case_results),
        "summary": {
            "critic_wins_by_correlation": critic_wins,
            "group_wins_by_correlation": group_wins,
            "ties_by_correlation": len(case_results) - critic_wins - group_wins,
            "mean_critic_minus_group_correlation": fmean(
                case["metrics"]["comparison"]["critic_minus_group_correlation"]
                for case in case_results
            ),
            "mean_critic_minus_group_calibrated_mse": fmean(
                case["metrics"]["comparison"]["critic_minus_group_calibrated_mse"]
                for case in case_results
            ),
        },
        "cases": case_results,
    }


def build_markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# Toy Scenario Sweep",
        "",
        "This generated report summarizes a deterministic CPU-only sweep over toy",
        "credit-assignment regimes. It measures estimator quality against the",
        "known oracle advantage from the toy dynamics; it is not a closed-loop",
        "PPO or GRPO training benchmark.",
        "",
        f"Seed: `{result['seed']}`",
        "",
        "| Case | Winner | Group r | Critic r | Group MSE | Critic MSE | Zero-var groups | Critic state hit | Interpretation |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for case in result["cases"]:
        metrics = case["metrics"]
        group = metrics["group_relative"]
        critic = metrics["critic_value_model"]
        counts = case["sample_counts"]
        lines.append(
            "| {case} | {winner} | {group_r:.3f} | {critic_r:.3f} | "
            "{group_mse:.5f} | {critic_mse:.5f} | {zero:.2f} | {hit:.2f} | {interp} |".format(
                case=case["case_name"],
                winner=case["winner_by_correlation"],
                group_r=group["pearson_correlation"],
                critic_r=critic["pearson_correlation"],
                group_mse=group["calibrated_mse"],
                critic_mse=critic["calibrated_mse"],
                zero=counts["zero_variance_group_fraction"],
                hit=counts["critic_exact_state_rate"],
                interp=case["interpretation"],
            )
        )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- Critic wins in this toy mean the value estimator had enough relevant state",
            "  information to recover temporal structure.",
            "- Group wins mean terminal group outcome information was more useful than",
            "  the critic's state abstraction in that regime.",
            "- Zero-variance group rates diagnose when group normalization has no",
            "  within-prompt reward contrast.",
            "- Critic state hit rate diagnoses whether the learned value table is using",
            "  exact state estimates or fallbacks.",
            "",
            "## Caveats",
            "",
            "The toy has a known finite-state oracle and a tabular critic, so it is much",
            "cleaner than neural long-horizon RL. The results support mechanism-level",
            "hypotheses only. Stronger GRPO variants, process rewards, KL/clipping",
            "details, optimizer effects, and closed-loop policy learning remain future",
            "work.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-json", type=Path, default=Path("results/toy_sweep_seed11.json"))
    parser.add_argument("--output-md", type=Path, default=Path("results/toy_sweep_seed11.md"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_sweep(seed=args.seed)
    markdown = build_markdown_report(result)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    args.output_md.write_text(markdown)

    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    print(
        "wins: "
        f"critic={result['summary']['critic_wins_by_correlation']} "
        f"group={result['summary']['group_wins_by_correlation']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
