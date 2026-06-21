"""Multi-seed toy matrix for PPO/critic vs GRPO-style credit assignment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from experiments.toy_credit_assignment import run_experiment


DEFAULT_SEEDS = [
    11,
    29,
    47,
    83,
    131,
    173,
    211,
    257,
    307,
    359,
    401,
    463,
    509,
    571,
    631,
    701,
    761,
    823,
    887,
    947,
]


DEFAULT_DEEP_CASES: list[dict[str, Any]] = [
    {
        "case_name": "horizon_04_baseline",
        "axis": "horizon",
        "scenario_name": "baseline",
        "train_groups": 120,
        "eval_groups": 32,
        "group_size": 6,
        "max_steps": 4,
    },
    {
        "case_name": "horizon_08_baseline",
        "axis": "horizon",
        "scenario_name": "baseline",
        "train_groups": 120,
        "eval_groups": 32,
        "group_size": 6,
        "max_steps": 8,
    },
    {
        "case_name": "horizon_12_baseline",
        "axis": "horizon",
        "scenario_name": "baseline",
        "train_groups": 120,
        "eval_groups": 32,
        "group_size": 6,
        "max_steps": 12,
    },
    {
        "case_name": "horizon_16_long_wait",
        "axis": "horizon",
        "scenario_name": "long_wait",
        "train_groups": 160,
        "eval_groups": 36,
        "group_size": 6,
        "max_steps": 16,
    },
    {
        "case_name": "group_size_02_long_wait",
        "axis": "group_size",
        "scenario_name": "long_wait",
        "train_groups": 140,
        "train_group_size": 6,
        "eval_groups": 32,
        "group_size": 2,
        "max_steps": 12,
    },
    {
        "case_name": "group_size_04_long_wait",
        "axis": "group_size",
        "scenario_name": "long_wait",
        "train_groups": 140,
        "train_group_size": 6,
        "eval_groups": 32,
        "group_size": 4,
        "max_steps": 12,
    },
    {
        "case_name": "group_size_08_long_wait",
        "axis": "group_size",
        "scenario_name": "long_wait",
        "train_groups": 140,
        "train_group_size": 6,
        "eval_groups": 32,
        "group_size": 8,
        "max_steps": 12,
    },
    {
        "case_name": "group_size_12_long_wait",
        "axis": "group_size",
        "scenario_name": "long_wait",
        "train_groups": 140,
        "train_group_size": 6,
        "eval_groups": 32,
        "group_size": 12,
        "max_steps": 12,
    },
    {
        "case_name": "critic_budget_002_full",
        "axis": "critic_budget",
        "scenario_name": "baseline",
        "train_groups": 2,
        "eval_groups": 32,
        "group_size": 6,
        "max_steps": 10,
    },
    {
        "case_name": "critic_budget_008_full",
        "axis": "critic_budget",
        "scenario_name": "baseline",
        "train_groups": 8,
        "eval_groups": 32,
        "group_size": 6,
        "max_steps": 10,
    },
    {
        "case_name": "critic_budget_032_full",
        "axis": "critic_budget",
        "scenario_name": "baseline",
        "train_groups": 32,
        "eval_groups": 32,
        "group_size": 6,
        "max_steps": 10,
    },
    {
        "case_name": "critic_budget_128_full",
        "axis": "critic_budget",
        "scenario_name": "baseline",
        "train_groups": 128,
        "eval_groups": 32,
        "group_size": 6,
        "max_steps": 10,
    },
    {
        "case_name": "observability_full",
        "axis": "observability",
        "scenario_name": "baseline",
        "train_groups": 80,
        "eval_groups": 32,
        "group_size": 6,
        "max_steps": 10,
    },
    {
        "case_name": "observability_coarse",
        "axis": "observability",
        "scenario_name": "coarse_critic",
        "train_groups": 80,
        "eval_groups": 32,
        "group_size": 6,
        "max_steps": 10,
    },
    {
        "case_name": "observability_blind",
        "axis": "observability",
        "scenario_name": "blind_critic",
        "train_groups": 80,
        "eval_groups": 32,
        "group_size": 6,
        "max_steps": 10,
    },
    {
        "case_name": "blind_undercovered_counterexample",
        "axis": "counterexample",
        "scenario_name": "blind_critic",
        "train_groups": 2,
        "eval_groups": 32,
        "group_size": 8,
        "max_steps": 4,
    },
    {
        "case_name": "sparse_hard_group04",
        "axis": "sparse_reward",
        "scenario_name": "sparse_hard",
        "train_groups": 180,
        "eval_groups": 36,
        "group_size": 4,
        "max_steps": 14,
    },
    {
        "case_name": "sparse_hard_group08",
        "axis": "sparse_reward",
        "scenario_name": "sparse_hard",
        "train_groups": 180,
        "eval_groups": 36,
        "group_size": 8,
        "max_steps": 14,
    },
]


def mean_ci95(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = fmean(values)
    if len(values) == 1:
        return mean, 0.0
    return mean, 1.96 * stdev(values) / math.sqrt(len(values))


def winner_from_delta(delta: float, tolerance: float = 1e-12) -> str:
    if delta > tolerance:
        return "critic"
    if delta < -tolerance:
        return "group"
    return "tie"


def evidence_from_ci95(delta: float, ci95: float) -> str:
    if delta - ci95 > 0:
        return "critic_clear"
    if delta + ci95 < 0:
        return "group_clear"
    return "near_tie"


def _case_seed_result(seed: int, case: dict[str, Any]) -> dict[str, Any]:
    result = run_experiment(
        seed=seed,
        scenario_name=case["scenario_name"],
        train_groups=case["train_groups"],
        eval_groups=case["eval_groups"],
        group_size=case["group_size"],
        train_group_size=case.get("train_group_size"),
        max_steps=case["max_steps"],
    )
    group = result["metrics"]["group_relative"]
    critic = result["metrics"]["critic_value_model"]
    comparison = result["metrics"]["comparison"]
    return {
        "seed": seed,
        "group_correlation": group["pearson_correlation"],
        "critic_correlation": critic["pearson_correlation"],
        "critic_minus_group_correlation": comparison[
            "critic_minus_group_correlation"
        ],
        "group_calibrated_mse": group["calibrated_mse"],
        "critic_calibrated_mse": critic["calibrated_mse"],
        "critic_minus_group_calibrated_mse": comparison[
            "critic_minus_group_calibrated_mse"
        ],
        "wait_token_fraction": result["sample_counts"]["wait_token_fraction"],
        "success_rate": result["sample_counts"]["success_rate"],
        "zero_variance_group_fraction": result["sample_counts"][
            "zero_variance_group_fraction"
        ],
        "critic_exact_state_rate": result["sample_counts"][
            "critic_exact_state_rate"
        ],
        "train_trajectories": result["sample_counts"]["train_trajectories"],
        "critic_is_privileged": result["config"]["critic_is_privileged"],
    }


def run_deep_matrix(
    seeds: list[int] | None = None,
    cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    seeds = seeds or DEFAULT_SEEDS
    cases = cases or DEFAULT_DEEP_CASES
    case_results: list[dict[str, Any]] = []

    for case in cases:
        seed_results = [_case_seed_result(seed, case) for seed in seeds]
        deltas = [row["critic_minus_group_correlation"] for row in seed_results]
        mse_deltas = [row["critic_minus_group_calibrated_mse"] for row in seed_results]
        group_corrs = [row["group_correlation"] for row in seed_results]
        critic_corrs = [row["critic_correlation"] for row in seed_results]
        hit_rates = [row["critic_exact_state_rate"] for row in seed_results]
        train_trajectories = [row["train_trajectories"] for row in seed_results]
        wait_rates = [row["wait_token_fraction"] for row in seed_results]
        success_rates = [row["success_rate"] for row in seed_results]
        zero_rates = [row["zero_variance_group_fraction"] for row in seed_results]

        delta_mean, delta_ci = mean_ci95(deltas)
        mse_delta_mean, mse_delta_ci = mean_ci95(mse_deltas)
        group_mean, group_ci = mean_ci95(group_corrs)
        critic_mean, critic_ci = mean_ci95(critic_corrs)

        case_results.append(
            {
                "case_name": case["case_name"],
                "axis": case["axis"],
                "scenario_name": case["scenario_name"],
                "train_groups": case["train_groups"],
                "eval_groups": case["eval_groups"],
                "group_size": case["group_size"],
                "train_group_size": case.get("train_group_size", case["group_size"]),
                "max_steps": case["max_steps"],
                "seed_results": seed_results,
                "mean_group_correlation": group_mean,
                "ci95_group_correlation": group_ci,
                "mean_critic_correlation": critic_mean,
                "ci95_critic_correlation": critic_ci,
                "mean_critic_minus_group_correlation": delta_mean,
                "ci95_critic_minus_group_correlation": delta_ci,
                "mean_critic_minus_group_calibrated_mse": mse_delta_mean,
                "ci95_critic_minus_group_calibrated_mse": mse_delta_ci,
                "mean_critic_exact_state_rate": fmean(hit_rates),
                "mean_train_trajectories": fmean(train_trajectories),
                "critic_is_privileged": any(
                    row["critic_is_privileged"] for row in seed_results
                ),
                "mean_wait_token_fraction": fmean(wait_rates),
                "mean_success_rate": fmean(success_rates),
                "mean_zero_variance_group_fraction": fmean(zero_rates),
                "winner_by_mean_correlation": winner_from_delta(delta_mean),
                "evidence_by_ci95": evidence_from_ci95(delta_mean, delta_ci),
            }
        )

    critic_wins = sum(
        1 for case in case_results if case["winner_by_mean_correlation"] == "critic"
    )
    group_wins = sum(
        1 for case in case_results if case["winner_by_mean_correlation"] == "group"
    )
    clear_critic = sum(
        1 for case in case_results if case["evidence_by_ci95"] == "critic_clear"
    )
    clear_group = sum(
        1 for case in case_results if case["evidence_by_ci95"] == "group_clear"
    )
    near_ties = sum(
        1 for case in case_results if case["evidence_by_ci95"] == "near_tie"
    )
    return {
        "seeds": seeds,
        "seed_count": len(seeds),
        "case_count": len(case_results),
        "overall": {
            "critic_wins_by_mean_correlation": critic_wins,
            "group_wins_by_mean_correlation": group_wins,
            "ties_by_mean_correlation": len(case_results) - critic_wins - group_wins,
            "clear_critic_cases_by_ci95": clear_critic,
            "clear_group_cases_by_ci95": clear_group,
            "near_tie_cases_by_ci95": near_ties,
            "mean_delta_correlation": fmean(
                case["mean_critic_minus_group_correlation"]
                for case in case_results
            ),
        },
        "cases": case_results,
    }


def flatten_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in result["cases"]:
        for seed_result in case["seed_results"]:
            row = {
                "case_name": case["case_name"],
                "axis": case["axis"],
                "scenario_name": case["scenario_name"],
                "train_groups": case["train_groups"],
                "eval_groups": case["eval_groups"],
                "group_size": case["group_size"],
                "max_steps": case["max_steps"],
            }
            row.update(seed_result)
            rows.append(row)
    return rows


def write_csv(result: dict[str, Any], output: Path) -> None:
    rows = flatten_rows(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("")
        return
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_deep_markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# Deep Toy Matrix",
        "",
        "This report aggregates the toy credit-assignment comparison across",
        f"{result['seed_count']} seeds and {result['case_count']} fixed cases.",
        "Positive delta means the critic-style TD estimator has higher oracle",
        "advantage correlation than the group-relative estimator.",
        "For group-size rows, critic replay is fixed at 840 training trajectories",
        "while only evaluation sibling group size changes.",
        "",
        "## Summary",
        "",
        f"- Critic wins by mean correlation: {result['overall']['critic_wins_by_mean_correlation']}",
        f"- Group wins by mean correlation: {result['overall']['group_wins_by_mean_correlation']}",
        f"- Clear critic-favorable cases by 95% CI: {result['overall']['clear_critic_cases_by_ci95']}",
        f"- Near-tie cases by 95% CI: {result['overall']['near_tie_cases_by_ci95']}",
        f"- Clear group-favorable cases by 95% CI: {result['overall']['clear_group_cases_by_ci95']}",
        f"- Mean critic-minus-group correlation: {result['overall']['mean_delta_correlation']:.3f}",
        "",
        "| Case | Axis | Mean winner | CI read | Group r | Critic r | Delta r | 95% CI | Critic hit | Wait frac | Success |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in result["cases"]:
        lines.append(
            "| {case_name} | {axis} | {winner} | {evidence} | {group:.3f} | {critic:.3f} | "
            "{delta:.3f} | +/- {ci:.3f} | {hit:.2f} | {wait:.2f} | {success:.2f} |".format(
                case_name=case["case_name"],
                axis=case["axis"],
                winner=case["winner_by_mean_correlation"],
                evidence=case["evidence_by_ci95"],
                group=case["mean_group_correlation"],
                critic=case["mean_critic_correlation"],
                delta=case["mean_critic_minus_group_correlation"],
                ci=case["ci95_critic_minus_group_correlation"],
                hit=case["mean_critic_exact_state_rate"],
                wait=case["mean_wait_token_fraction"],
                success=case["mean_success_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The critic estimator is strongest when state coverage is high and traces",
            "  contain mixed token roles.",
            "- Group-relative estimation remains competitive when the critic is blind,",
            "  poorly covered, or when terminal group outcomes carry most of the signal.",
            "- These experiments measure estimator fidelity, not closed-loop policy",
            "  improvement under PPO/GRPO clipping, KL control, or optimizer effects.",
            "",
        ]
    )
    return "\n".join(lines)


def render_delta_chart(result: dict[str, Any], output: Path) -> None:
    width = 1500
    row_h = 48
    top = 118
    left = 420
    right = 90
    height = top + row_h * len(result["cases"]) + 70
    max_abs = max(
        0.1,
        max(abs(case["mean_critic_minus_group_correlation"]) for case in result["cases"]),
    )
    chart_w = width - left - right
    zero_x = left + chart_w // 2
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="40" y="50" font-family="Arial, sans-serif" font-size="30" font-weight="700" fill="#0B2545">Critic minus group correlation by case</text>',
        '<text x="40" y="80" font-family="Arial, sans-serif" font-size="18" fill="#333333">Bars show means; whiskers are 95% across-seed intervals for critic minus group correlation.</text>',
        f'<line x1="{zero_x}" y1="{top - 24}" x2="{zero_x}" y2="{height - 55}" stroke="#666666" stroke-width="2"/>',
        f'<text x="{zero_x - 8}" y="{height - 30}" font-family="Arial, sans-serif" font-size="15" fill="#555555">0</text>',
    ]

    for idx, case in enumerate(result["cases"]):
        y = top + idx * row_h
        delta = case["mean_critic_minus_group_correlation"]
        ci95_delta = case["ci95_critic_minus_group_correlation"]
        bar_len = int(abs(delta) / max_abs * (chart_w / 2 - 20))
        color = "#1F6F4A" if delta >= 0 else "#9B1C1C"
        if delta >= 0:
            x = zero_x
        else:
            x = zero_x - bar_len
        value_x = zero_x + bar_len + 10 if delta >= 0 else zero_x - bar_len - 74
        half_w = chart_w / 2 - 20
        ci_low = int(zero_x + (delta - ci95_delta) / max_abs * half_w)
        ci_high = int(zero_x + (delta + ci95_delta) / max_abs * half_w)
        ci_y = y + 20
        parts.extend(
            [
                f'<text x="40" y="{y + 27}" font-family="Arial, sans-serif" font-size="15" fill="#111111">{case["case_name"]}</text>',
                f'<line x1="{ci_low}" y1="{ci_y}" x2="{ci_high}" y2="{ci_y}" stroke="#333333" stroke-width="2"/>',
                f'<line x1="{ci_low}" y1="{ci_y - 7}" x2="{ci_low}" y2="{ci_y + 7}" stroke="#333333" stroke-width="2"/>',
                f'<line x1="{ci_high}" y1="{ci_y - 7}" x2="{ci_high}" y2="{ci_y + 7}" stroke="#333333" stroke-width="2"/>',
                f'<rect x="{x}" y="{y + 8}" width="{bar_len}" height="24" rx="6" fill="{color}"/>',
                f'<text x="{value_x}" y="{y + 27}" font-family="Arial, sans-serif" font-size="15" fill="{color}">{delta:+.3f}</text>',
            ]
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts + ["</svg>\n"]))


def render_coverage_scatter(result: dict[str, Any], output: Path) -> None:
    width = 1100
    height = 780
    margin = 90
    plot_left = margin
    plot_top = 130
    plot_w = width - 2 * margin
    plot_h = height - 220
    deltas = [case["mean_critic_minus_group_correlation"] for case in result["cases"]]
    min_delta = min(-0.08, min(deltas))
    max_delta = max(0.08, max(deltas))

    def x_of(hit: float) -> int:
        return int(plot_left + hit * plot_w)

    def y_of(delta: float) -> int:
        return int(plot_top + (max_delta - delta) / (max_delta - min_delta) * plot_h)

    zero_y = y_of(0.0)
    label_offsets = {
        "critic_budget_002_full": (16, -12),
        "critic_budget_128_full": (-170, 24),
        "observability_blind": (-150, 28),
        "blind_undercovered_counterexample": (16, -18),
        "horizon_16_long_wait": (-170, -32),
        "sparse_hard_group08": (-170, 42),
        "group_size_02_long_wait": (-150, -18),
    }

    def style_for(case: dict[str, Any]) -> tuple[str, str]:
        name = case["case_name"]
        if "blind" in name:
            return "#9B1C1C", "triangle"
        if "coarse" in name:
            return "#B7791F", "diamond"
        if "observability_full" in name or case.get("critic_is_privileged"):
            return "#2457A7", "square"
        return "#1F6F4A", "circle"

    def marker_svg(x: int, y: int, radius: int, color: str, shape: str) -> str:
        if shape == "triangle":
            points = f"{x},{y - radius} {x - radius},{y + radius} {x + radius},{y + radius}"
            return f'<polygon points="{points}" fill="{color}" stroke="#111111" stroke-width="1.5"/>'
        if shape == "diamond":
            points = f"{x},{y - radius} {x - radius},{y} {x},{y + radius} {x + radius},{y}"
            return f'<polygon points="{points}" fill="{color}" stroke="#111111" stroke-width="1.5"/>'
        if shape == "square":
            side = radius * 2
            return f'<rect x="{x - radius}" y="{y - radius}" width="{side}" height="{side}" fill="{color}" stroke="#111111" stroke-width="1.5"/>'
        return f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}" stroke="#111111" stroke-width="1.5"/>'

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="50" y="50" font-family="Arial, sans-serif" font-size="30" font-weight="700" fill="#0B2545">Critic coverage vs estimator advantage</text>',
        '<text x="50" y="80" font-family="Arial, sans-serif" font-size="18" fill="#333333">Color/shape encode critic observability; marker size scales with wait-token fraction.</text>',
        f'<rect x="{plot_left}" y="{plot_top}" width="{plot_w}" height="{plot_h}" fill="white" stroke="#AAB4C0" stroke-width="2"/>',
        f'<line x1="{plot_left}" y1="{zero_y}" x2="{plot_left + plot_w}" y2="{zero_y}" stroke="#777777" stroke-width="2"/>',
        f'<text x="{plot_left + plot_w // 2 - 100}" y="{height - 38}" font-family="Arial, sans-serif" font-size="18" fill="#333333">critic exact-state hit rate</text>',
        f'<text x="20" y="{plot_top + plot_h // 2}" font-family="Arial, sans-serif" font-size="18" fill="#333333">delta r</text>',
        '<rect x="120" y="150" width="330" height="104" rx="8" fill="#FFFFFF" stroke="#D0D7DE"/>',
        '<circle cx="142" cy="178" r="8" fill="#1F6F4A" stroke="#111111" stroke-width="1.5"/>',
        '<text x="160" y="184" font-family="Arial, sans-serif" font-size="14" fill="#222222">non-blind critic</text>',
        '<rect x="134" y="198" width="16" height="16" fill="#2457A7" stroke="#111111" stroke-width="1.5"/>',
        '<text x="160" y="212" font-family="Arial, sans-serif" font-size="14" fill="#222222">privileged/full-observation row</text>',
        '<polygon points="142,229 134,245 150,245" fill="#9B1C1C" stroke="#111111" stroke-width="1.5"/>',
        '<text x="160" y="244" font-family="Arial, sans-serif" font-size="14" fill="#222222">blind critic</text>',
    ]
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = x_of(tick)
        parts.append(
            f'<line x1="{x}" y1="{plot_top + plot_h}" x2="{x}" y2="{plot_top + plot_h + 8}" stroke="#777777"/>'
        )
        parts.append(
            f'<text x="{x - 14}" y="{plot_top + plot_h + 30}" font-family="Arial, sans-serif" font-size="14" fill="#444444">{tick:.2f}</text>'
        )

    for case in result["cases"]:
        x = x_of(case["mean_critic_exact_state_rate"])
        y = y_of(case["mean_critic_minus_group_correlation"])
        color, shape = style_for(case)
        radius = 6 + int(10 * case["mean_wait_token_fraction"])
        parts.append(marker_svg(x, y, radius, color, shape))
        if case["case_name"] in label_offsets:
            dx, dy = label_offsets[case["case_name"]]
            label = case["case_name"].replace("_", " ")
            text_x = x + dx
            text_y = y + dy
            text_w = max(116, min(240, 7 * len(label) + 12))
            parts.append(
                f'<rect x="{text_x - 4}" y="{text_y - 15}" width="{text_w}" height="20" rx="4" fill="white" opacity="0.82"/>'
            )
            parts.append(
                f'<text x="{text_x}" y="{text_y}" font-family="Arial, sans-serif" font-size="13" fill="#222222">{label}</text>'
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts + ["</svg>\n"]))


def render_deep_charts(result: dict[str, Any], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    delta_chart = output_dir / "deep_matrix_delta.svg"
    coverage_chart = output_dir / "deep_matrix_coverage.svg"
    render_delta_chart(result, delta_chart)
    render_coverage_scatter(result, coverage_chart)
    return [delta_chart, coverage_chart]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--output-json", type=Path, default=Path("results/deep_matrix.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/deep_matrix.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("results/deep_matrix.md"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    result = run_deep_matrix(seeds=seeds)
    markdown = build_deep_markdown_report(result)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    write_csv(result, args.output_csv)
    args.output_md.write_text(markdown)
    charts = render_deep_charts(result, args.figures_dir)

    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_md}")
    for chart in charts:
        print(f"wrote {chart}")
    print(
        "wins: "
        f"critic={result['overall']['critic_wins_by_mean_correlation']} "
        f"group={result['overall']['group_wins_by_mean_correlation']}"
    )
    print(
        "ci95 evidence: "
        f"critic_clear={result['overall']['clear_critic_cases_by_ci95']} "
        f"near_tie={result['overall']['near_tie_cases_by_ci95']} "
        f"group_clear={result['overall']['clear_group_cases_by_ci95']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
