"""Coverage audit for structural critic-free anchor-action contrast.

The variance-credit grid includes ``anchor_action_contrast`` as a stronger
critic-free step-level baseline. This audit asks when that estimator works. It
sweeps the number of evaluation groups, which changes how often exact
state-action anchors repeat in the batch, and compares anchor contrast with
sibling group normalization and critic TD.

The audit is still estimator fidelity, not closed-loop training evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from experiments.deep_matrix import DEFAULT_SEEDS
from experiments.variance_credit_grid import run_grid


DEFAULT_EVAL_GROUPS = [2, 4, 8, 16, 32, 48, 64]


def ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return 1.96 * stdev(values) / (len(values) ** 0.5)


def estimator_metrics(result: dict[str, Any], name: str) -> dict[str, float]:
    return next(
        estimator for estimator in result["estimators"] if estimator["name"] == name
    )["metrics"]


def run_one(
    *,
    seed: int,
    eval_groups: int,
    scenario_name: str,
    train_groups: int,
    group_size: int,
    max_steps: int,
    branches_per_state: int,
) -> dict[str, Any]:
    result = run_grid(
        seed=seed,
        scenario_name=scenario_name,
        train_groups=train_groups,
        eval_groups=eval_groups,
        group_size=group_size,
        max_steps=max_steps,
        branches_per_state=branches_per_state,
    )
    sibling = estimator_metrics(result, "sibling_group_norm")
    anchor = estimator_metrics(result, "anchor_action_contrast")
    critic = estimator_metrics(result, "critic_td")
    sampled = estimator_metrics(result, "sampled_mc_td")
    coverage = result["sample_counts"]["anchor_action"]
    return {
        "seed": seed,
        "eval_groups": eval_groups,
        "eval_tokens": result["sample_counts"]["eval_tokens"],
        "supported_step_fraction": coverage["supported_step_fraction"],
        "unique_anchor_states": coverage["unique_anchor_states"],
        "unique_anchor_actions": coverage["unique_anchor_actions"],
        "sibling_r": sibling["pearson_correlation"],
        "anchor_r": anchor["pearson_correlation"],
        "critic_r": critic["pearson_correlation"],
        "sampled_mc_r": sampled["pearson_correlation"],
        "anchor_minus_sibling_r": (
            anchor["pearson_correlation"] - sibling["pearson_correlation"]
        ),
        "critic_minus_anchor_r": (
            critic["pearson_correlation"] - anchor["pearson_correlation"]
        ),
        "anchor_within_var": anchor["within_trajectory_variance"],
        "anchor_wait_leak": anchor["wait_to_active_abs_ratio"],
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [row["anchor_minus_sibling_r"] for row in rows]
    critic_gaps = [row["critic_minus_anchor_r"] for row in rows]
    return {
        "eval_groups": rows[0]["eval_groups"],
        "seed_count": len(rows),
        "eval_tokens": fmean(row["eval_tokens"] for row in rows),
        "supported_step_fraction": fmean(
            row["supported_step_fraction"] for row in rows
        ),
        "unique_anchor_states": fmean(row["unique_anchor_states"] for row in rows),
        "unique_anchor_actions": fmean(row["unique_anchor_actions"] for row in rows),
        "sibling_r": fmean(row["sibling_r"] for row in rows),
        "anchor_r": fmean(row["anchor_r"] for row in rows),
        "critic_r": fmean(row["critic_r"] for row in rows),
        "sampled_mc_r": fmean(row["sampled_mc_r"] for row in rows),
        "anchor_minus_sibling_r": fmean(deltas),
        "anchor_minus_sibling_ci95": ci95(deltas),
        "critic_minus_anchor_r": fmean(critic_gaps),
        "critic_minus_anchor_ci95": ci95(critic_gaps),
        "anchor_within_var": fmean(row["anchor_within_var"] for row in rows),
        "anchor_wait_leak": fmean(row["anchor_wait_leak"] for row in rows),
    }


def run_audit(
    *,
    seeds: list[int] | None = None,
    eval_groups_values: list[int] | None = None,
    scenario_name: str = "long_wait",
    train_groups: int = 120,
    group_size: int = 6,
    max_steps: int = 14,
    branches_per_state: int = 12,
) -> dict[str, Any]:
    seeds = list(DEFAULT_SEEDS[:5] if seeds is None else seeds)
    eval_groups_values = list(
        DEFAULT_EVAL_GROUPS if eval_groups_values is None else eval_groups_values
    )
    if not seeds or not eval_groups_values:
        raise ValueError("expected at least one seed and eval-groups value")
    if train_groups <= 0 or group_size <= 1 or max_steps < 2:
        raise ValueError("expected train_groups > 0, group_size > 1, max_steps >= 2")
    if branches_per_state <= 0:
        raise ValueError("branches_per_state must be positive")
    if any(value <= 0 for value in eval_groups_values):
        raise ValueError("eval-groups values must be positive")

    seed_results: list[dict[str, Any]] = []
    for eval_groups in eval_groups_values:
        for seed in seeds:
            seed_results.append(
                run_one(
                    seed=seed,
                    eval_groups=eval_groups,
                    scenario_name=scenario_name,
                    train_groups=train_groups,
                    group_size=group_size,
                    max_steps=max_steps,
                    branches_per_state=branches_per_state,
                )
            )

    aggregate_rows = [
        aggregate(
            [row for row in seed_results if row["eval_groups"] == eval_groups]
        )
        for eval_groups in eval_groups_values
    ]
    first_anchor_beats_sibling = next(
        (
            row["eval_groups"]
            for row in aggregate_rows
            if row["anchor_minus_sibling_r"] > 0.0
        ),
        None,
    )
    first_support_ge_80 = next(
        (
            row["eval_groups"]
            for row in aggregate_rows
            if row["supported_step_fraction"] >= 0.80
        ),
        None,
    )
    return {
        "config": {
            "seeds": seeds,
            "eval_groups_values": eval_groups_values,
            "scenario_name": scenario_name,
            "train_groups": train_groups,
            "group_size": group_size,
            "max_steps": max_steps,
            "branches_per_state": branches_per_state,
        },
        "aggregate_rows": aggregate_rows,
        "seed_results": seed_results,
        "summary": {
            "first_eval_groups_anchor_beats_sibling": first_anchor_beats_sibling,
            "first_eval_groups_support_ge_80": first_support_ge_80,
            "anchor_beats_sibling_rows": sum(
                1 for row in aggregate_rows if row["anchor_minus_sibling_r"] > 0.0
            ),
            "critic_above_anchor_rows": sum(
                1 for row in aggregate_rows if row["critic_minus_anchor_r"] > 0.0
            ),
            "max_anchor_r": max(row["anchor_r"] for row in aggregate_rows),
            "max_supported_step_fraction": max(
                row["supported_step_fraction"] for row in aggregate_rows
            ),
        },
    }


def fmt(value: float) -> str:
    return f"{value:.3f}"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Anchor Coverage Audit",
        "",
        "This audit sweeps evaluation batch size to change exact repeated",
        "state-action coverage for anchor-action contrast. It tests when this",
        "critic-free structural batch estimator beats trajectory-level sibling",
        "group normalization, and whether it closes the gap to critic TD.",
        "",
        "| Eval groups | Support | Sibling r | Anchor r | Critic r | A-G | C-A | Anchor wait |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["aggregate_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["eval_groups"]),
                    fmt(row["supported_step_fraction"]),
                    fmt(row["sibling_r"]),
                    fmt(row["anchor_r"]),
                    fmt(row["critic_r"]),
                    fmt(row["anchor_minus_sibling_r"]),
                    fmt(row["critic_minus_anchor_r"]),
                    fmt(row["anchor_wait_leak"]),
                ]
            )
            + " |"
        )
    summary = result["summary"]
    lines.extend(
        [
            "",
            "Summary:",
            "- First eval-groups value where anchor beats sibling: "
            f"{summary['first_eval_groups_anchor_beats_sibling']}.",
            "- First eval-groups value where support >= 0.80: "
            f"{summary['first_eval_groups_support_ge_80']}.",
            f"- Rows where critic remains above anchor: {summary['critic_above_anchor_rows']}.",
            f"- Maximum anchor r: {fmt(summary['max_anchor_r'])}.",
            "",
            "Reading:",
            "- Anchor contrast is coverage-sensitive. At low exact-anchor coverage,",
            "  it can be worse than sibling group normalization.",
            "- With enough repeated anchors, it becomes a strong critic-free",
            "  step-level baseline, but critic TD remains higher in this sweep.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument(
        "--eval-groups-values",
        type=int,
        nargs="*",
        default=None,
    )
    parser.add_argument("--scenario", default="long_wait")
    parser.add_argument("--train-groups", type=int, default=120)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=14)
    parser.add_argument("--branches-per-state", type=int, default=12)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/anchor_coverage_audit_seedset.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/anchor_coverage_audit_seedset.md"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_audit(
        seeds=args.seeds,
        eval_groups_values=args.eval_groups_values,
        scenario_name=args.scenario,
        train_groups=args.train_groups,
        group_size=args.group_size,
        max_steps=args.max_steps,
        branches_per_state=args.branches_per_state,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    write_markdown(result, args.output_md)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    print(
        "coverage: "
        "first_anchor_beats_sibling="
        f"{result['summary']['first_eval_groups_anchor_beats_sibling']} "
        f"first_support_ge_80={result['summary']['first_eval_groups_support_ge_80']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
