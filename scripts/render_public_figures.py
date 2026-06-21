#!/usr/bin/env python3
"""Render public paper PNG figures from canonical JSON artifacts.

The experiment code itself remains dependency-free. This paper-artifact helper
uses Pillow so the public PDF can be rebuilt without relying on stale PNGs or
platform thumbnailers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - exercised by shell script path.
    raise SystemExit(
        "Pillow is required to regenerate public PNG figures. "
        "Install pillow or set FIGURE_PYTHON to a Python with Pillow."
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.deep_matrix import render_deep_charts


RESULT_JSON = ROOT / "results/deep_matrix_20seed.json"
RESULT_FIGURES = ROOT / "results/figures"
PUBLIC_FIGURES = ROOT / "public/figures"
FONT_REGULAR = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
FONT_BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold and FONT_BOLD.exists() else FONT_REGULAR
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def render_delta_png(result: dict[str, Any], output: Path) -> None:
    width = 1500
    row_h = 48
    top = 118
    left = 420
    right = 90
    height = top + row_h * len(result["cases"]) + 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.text(
        (40, 38),
        "Critic minus group correlation by case",
        font=font(30, bold=True),
        fill="#0B2545",
    )
    draw.text(
        (40, 68),
        "Bars show means; whiskers are 95% across-seed intervals for critic minus group correlation.",
        font=font(18),
        fill="#333333",
    )

    max_abs = max(
        0.1,
        max(
            abs(case["mean_critic_minus_group_correlation"])
            + case["ci95_critic_minus_group_correlation"]
            for case in result["cases"]
        ),
    )
    chart_w = width - left - right
    zero_x = left + chart_w // 2
    half_w = chart_w / 2 - 20
    draw.line((zero_x, top - 24, zero_x, height - 55), fill="#666666", width=2)
    draw.text((zero_x - 8, height - 45), "0", font=font(15), fill="#555555")

    for index, case in enumerate(result["cases"]):
        y = top + index * row_h
        delta = case["mean_critic_minus_group_correlation"]
        ci95 = case["ci95_critic_minus_group_correlation"]
        color = "#1F6F4A" if delta >= 0 else "#9B1C1C"
        draw.text((40, y + 13), case["case_name"], font=font(15), fill="#111111")

        ci_low = int(zero_x + (delta - ci95) / max_abs * half_w)
        ci_high = int(zero_x + (delta + ci95) / max_abs * half_w)
        ci_y = y + 20
        draw.line((ci_low, ci_y, ci_high, ci_y), fill="#333333", width=2)
        draw.line((ci_low, ci_y - 7, ci_low, ci_y + 7), fill="#333333", width=2)
        draw.line((ci_high, ci_y - 7, ci_high, ci_y + 7), fill="#333333", width=2)

        bar_len = int(abs(delta) / max_abs * half_w)
        x0, x1 = (zero_x, zero_x + bar_len) if delta >= 0 else (zero_x - bar_len, zero_x)
        draw.rounded_rectangle((x0, y + 8, x1, y + 32), radius=6, fill=color)
        value_x = x1 + 10 if delta >= 0 else x0 - 74
        draw.text((value_x, y + 13), f"{delta:+.3f}", font=font(15), fill=color)

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def render_coverage_png(result: dict[str, Any], output: Path) -> None:
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

    def x_of(hit: float) -> float:
        return plot_left + hit * plot_w

    def y_of(delta: float) -> float:
        return plot_top + (max_delta - delta) / (max_delta - min_delta) * plot_h

    def style_for(case: dict[str, Any]) -> tuple[str, str]:
        name = case["case_name"]
        if "blind" in name:
            return "#9B1C1C", "triangle"
        if "coarse" in name:
            return "#B7791F", "diamond"
        if "observability_full" in name or case.get("critic_is_privileged"):
            return "#2457A7", "square"
        return "#1F6F4A", "circle"

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    def marker(x: float, y: float, radius: int, color: str, shape: str) -> None:
        if shape == "triangle":
            draw.polygon(
                [(x, y - radius), (x - radius, y + radius), (x + radius, y + radius)],
                fill=color,
                outline="#111111",
            )
        elif shape == "diamond":
            draw.polygon(
                [(x, y - radius), (x - radius, y), (x, y + radius), (x + radius, y)],
                fill=color,
                outline="#111111",
            )
        elif shape == "square":
            draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color, outline="#111111")
        else:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="#111111", width=2)

    draw.text(
        (50, 38),
        "Critic coverage vs estimator advantage",
        font=font(30, bold=True),
        fill="#0B2545",
    )
    draw.text(
        (50, 68),
        "Color/shape encode critic observability; marker size scales with wait-token fraction.",
        font=font(18),
        fill="#333333",
    )
    draw.rectangle(
        (plot_left, plot_top, plot_left + plot_w, plot_top + plot_h),
        outline="#AAB4C0",
        width=2,
    )
    zero_y = y_of(0.0)
    draw.line((plot_left, zero_y, plot_left + plot_w, zero_y), fill="#777777", width=2)
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = x_of(tick)
        draw.line((x, plot_top + plot_h, x, plot_top + plot_h + 8), fill="#777777")
        draw.text((x - 14, plot_top + plot_h + 14), f"{tick:.2f}", font=font(14), fill="#444444")
    draw.text(
        (plot_left + plot_w // 2 - 98, height - 42),
        "critic exact-state hit rate",
        font=font(18),
        fill="#333333",
    )
    draw.text((20, plot_top + plot_h // 2), "delta r", font=font(18), fill="#333333")

    legend_x = 120
    legend_y = 150
    draw.rounded_rectangle(
        (legend_x, legend_y, legend_x + 330, legend_y + 112),
        radius=8,
        fill="white",
        outline="#D0D7DE",
    )
    marker(legend_x + 22, legend_y + 28, 8, "#1F6F4A", "circle")
    draw.text((legend_x + 44, legend_y + 20), "non-blind critic", font=font(14), fill="#222222")
    marker(legend_x + 22, legend_y + 56, 8, "#2457A7", "square")
    draw.text(
        (legend_x + 44, legend_y + 48),
        "privileged/full-observation row",
        font=font(14),
        fill="#222222",
    )
    marker(legend_x + 22, legend_y + 84, 8, "#9B1C1C", "triangle")
    draw.text((legend_x + 44, legend_y + 76), "blind critic", font=font(14), fill="#222222")

    label_offsets = {
        "critic_budget_002_full": (16, -12),
        "critic_budget_128_full": (-170, 24),
        "observability_blind": (-150, 28),
        "blind_undercovered_counterexample": (-260, -18),
        "horizon_16_long_wait": (-170, -32),
        "sparse_hard_group08": (-170, 42),
        "group_size_02_long_wait": (-170, -18),
    }
    for case in result["cases"]:
        x = x_of(case["mean_critic_exact_state_rate"])
        y = y_of(case["mean_critic_minus_group_correlation"])
        color, shape = style_for(case)
        radius = 6 + int(10 * case["mean_wait_token_fraction"])
        marker(x, y, radius, color, shape)
        if case["case_name"] in label_offsets:
            dx, dy = label_offsets[case["case_name"]]
            label = case["case_name"].replace("_", " ")
            text_x = x + dx
            text_y = y + dy
            text_w = max(116, min(240, 7 * len(label) + 12))
            draw.rounded_rectangle(
                (text_x - 4, text_y - 15, text_x + text_w, text_y + 5),
                radius=4,
                fill="white",
            )
            draw.text((text_x, text_y - 13), label, font=font(13), fill="#222222")

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> int:
    result = json.loads(RESULT_JSON.read_text())
    render_deep_charts(result, RESULT_FIGURES)
    render_delta_png(result, PUBLIC_FIGURES / "deep_matrix_delta.png")
    render_coverage_png(result, PUBLIC_FIGURES / "deep_matrix_coverage.png")
    print("wrote public/figures/deep_matrix_delta.png")
    print("wrote public/figures/deep_matrix_coverage.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
