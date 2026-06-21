"""Held-out estimator-selection audit for the phase grid.

This module turns the estimator-selection appendix into a small executable
audit. It uses the existing 48-cell phase grid and cross-fits in two ways:

* even/odd cells split meta-selection training from held-out evaluation;
* alternating seeds inside each cell split audit estimates from held-out
  regret measurements.

The deployable rule chooses the estimator with lower audit MSE plus a fitted
critic-cost penalty. Regret is measured on held-out seeds against the best of
group and critic for that same cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Any


def mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def row_mse(row: dict[str, Any], estimator: str, split: str) -> float:
    seed_results = row["seed_results"]
    if split == "audit":
        selected = seed_results[::2]
    elif split == "heldout":
        selected = seed_results[1::2] or seed_results[::2]
    else:
        raise ValueError(f"unknown split {split!r}")
    key = "group_calibrated_mse" if estimator == "group" else "critic_calibrated_mse"
    return mean([float(seed[key]) for seed in selected])


def critic_cost(row: dict[str, Any], max_train_trajectories: float) -> float:
    if max_train_trajectories <= 0:
        return 0.0
    return float(row["train_trajectories"]) / max_train_trajectories


def choose_by_audit_mse(
    row: dict[str, Any],
    *,
    lambda_cost: float,
    max_train_trajectories: float,
) -> str:
    group_score = row_mse(row, "group", "audit")
    critic_score = (
        row_mse(row, "critic", "audit")
        + lambda_cost * critic_cost(row, max_train_trajectories)
    )
    return "critic" if critic_score < group_score else "group"


def oracle_best(row: dict[str, Any]) -> str:
    group = row_mse(row, "group", "heldout")
    critic = row_mse(row, "critic", "heldout")
    return "critic" if critic < group else "group"


def regret_for_choice(row: dict[str, Any], choice: str) -> float:
    selected = row_mse(row, choice, "heldout")
    oracle = min(row_mse(row, "group", "heldout"), row_mse(row, "critic", "heldout"))
    return max(0.0, selected - oracle)


def evaluate_policy(rows: list[dict[str, Any]], chooser) -> dict[str, float]:
    choices = [chooser(row) for row in rows]
    regrets = [regret_for_choice(row, choice) for row, choice in zip(rows, choices)]
    oracle = [oracle_best(row) for row in rows]
    return {
        "cell_count": float(len(rows)),
        "mean_regret": mean(regrets),
        "max_regret": max(regrets) if regrets else 0.0,
        "selection_accuracy": (
            sum(1 for choice, best in zip(choices, oracle) if choice == best) / len(rows)
            if rows
            else 0.0
        ),
        "critic_choice_rate": (
            sum(1 for choice in choices if choice == "critic") / len(rows)
            if rows
            else 0.0
        ),
    }


def stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16)


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Split cells while avoiding obvious axis confounds.

    The canonical phase grid has paired cells that differ only by reward regime.
    A naive sorted even/odd split puts every contrast cell in one side and every
    sparse cell in the other. This split alternates reward regimes within each
    shared phase stratum when possible, and falls back to a stable hash split
    for grids without paired reward cells.
    """

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        stratum = (
            row.get("heterogeneity"),
            row.get("observability"),
            row.get("coverage"),
            row.get("drift"),
        )
        grouped.setdefault(stratum, []).append(row)

    train_rows: list[dict[str, Any]] = []
    heldout_rows: list[dict[str, Any]] = []
    paired_groups = 0
    singleton_rows: list[dict[str, Any]] = []
    for stratum, bucket in sorted(grouped.items(), key=lambda item: str(item[0])):
        rewards = {row.get("reward") for row in bucket}
        if len(bucket) >= 2 and len(rewards) >= 2:
            paired_groups += 1
            ordered = sorted(bucket, key=lambda row: (str(row.get("reward")), row["cell_name"]))
            parity = stable_int("|".join(str(part) for part in stratum)) % 2
            for index, row in enumerate(ordered):
                if index % 2 == parity:
                    train_rows.append(row)
                else:
                    heldout_rows.append(row)
        else:
            singleton_rows.extend(bucket)

    for row in sorted(singleton_rows, key=lambda item: item["cell_name"]):
        if stable_int(row["cell_name"]) % 2 == 0:
            train_rows.append(row)
        else:
            heldout_rows.append(row)

    if not train_rows or not heldout_rows:
        midpoint = max(1, len(rows) // 2)
        train_rows = rows[:midpoint]
        heldout_rows = rows[midpoint:]
        split_note = "fallback sorted half split"
    elif paired_groups:
        split_note = (
            "reward-paired split within heterogeneity/observability/coverage/drift "
            "strata; stable hash fallback for unpaired cells"
        )
    else:
        split_note = "stable hash split over cell names"
    return sorted(train_rows, key=lambda row: row["cell_name"]), sorted(
        heldout_rows,
        key=lambda row: row["cell_name"],
    ), split_note


def fit_lambda(
    train_rows: list[dict[str, Any]],
    *,
    max_train_trajectories: float,
    candidates: list[float] | None = None,
) -> float:
    candidates = candidates or [
        0.0,
        0.0005,
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
    ]
    scored: list[tuple[float, float]] = []
    for candidate in candidates:
        metrics = evaluate_policy(
            train_rows,
            lambda row, candidate=candidate: choose_by_audit_mse(
                row,
                lambda_cost=candidate,
                max_train_trajectories=max_train_trajectories,
            ),
        )
        scored.append((metrics["mean_regret"], candidate))
    scored.sort()
    return scored[0][1]


def run_selection_regret(phase: dict[str, Any]) -> dict[str, Any]:
    rows = sorted(phase["aggregate_rows"], key=lambda row: row["cell_name"])
    if len(rows) < 2:
        raise ValueError("expected at least two phase-grid cells")
    train_rows, heldout_rows, split_note = split_rows(rows)
    max_train = max(float(row["train_trajectories"]) for row in rows)
    lambda_cost = fit_lambda(train_rows, max_train_trajectories=max_train)

    policies = {
        "audit_mse_cost": lambda row: choose_by_audit_mse(
            row,
            lambda_cost=lambda_cost,
            max_train_trajectories=max_train,
        ),
        "always_group": lambda row: "group",
        "always_critic": lambda row: "critic",
    }
    train_metrics = {
        name: evaluate_policy(train_rows, chooser)
        for name, chooser in policies.items()
    }
    heldout_metrics = {
        name: evaluate_policy(heldout_rows, chooser)
        for name, chooser in policies.items()
    }

    return {
        "config": {
            "source_cell_count": len(rows),
            "train_cell_count": len(train_rows),
            "heldout_cell_count": len(heldout_rows),
            "seed_split": "alternating audit/heldout seeds within each cell",
            "cell_split": split_note,
            "fitted_lambda_cost": lambda_cost,
            "max_train_trajectories": max_train,
        },
        "train_metrics": train_metrics,
        "heldout_metrics": heldout_metrics,
        "heldout_choices": [
            {
                "cell_name": row["cell_name"],
                "realized_credit_heterogeneity": row["credit_heterogeneity"],
                "oracle_best": oracle_best(row),
                "audit_mse_cost_choice": policies["audit_mse_cost"](row),
                "group_mse_audit": row_mse(row, "group", "audit"),
                "critic_mse_audit": row_mse(row, "critic", "audit"),
                "group_mse_heldout": row_mse(row, "group", "heldout"),
                "critic_mse_heldout": row_mse(row, "critic", "heldout"),
                "audit_mse_cost_regret": regret_for_choice(
                    row,
                    policies["audit_mse_cost"](row),
                ),
            }
            for row in heldout_rows
        ],
    }


def fmt(value: float) -> str:
    return f"{value:.5f}"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Held-out estimator-selection regret",
        "",
        "The selection rule fits a critic-cost penalty on half the phase-grid",
        "cells and evaluates regret on held-out cells. Within each cell,",
        "alternating seeds split audit MSE estimates from held-out regret.",
        "",
        f"- Train cells: {result['config']['train_cell_count']}",
        f"- Held-out cells: {result['config']['heldout_cell_count']}",
        f"- Fitted lambda cost: {fmt(result['config']['fitted_lambda_cost'])}",
        "",
        "| Policy | Held-out regret | Max regret | Accuracy | Critic rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in result["heldout_metrics"].items():
        lines.append(
            "| {name} | {regret} | {max_regret} | {acc} | {rate} |".format(
                name=name,
                regret=fmt(metrics["mean_regret"]),
                max_regret=fmt(metrics["max_regret"]),
                acc=fmt(metrics["selection_accuracy"]),
                rate=fmt(metrics["critic_choice_rate"]),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase-json",
        type=Path,
        default=Path("results/credit_phase_diagram_seedset.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/selection_regret_seedset.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/selection_regret_seedset.md"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    phase = json.loads(args.phase_json.read_text())
    result = run_selection_regret(phase)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_markdown(result, args.output_md)
    best = result["heldout_metrics"]["audit_mse_cost"]
    print(
        "selection regret: "
        f"mean={best['mean_regret']:.4f} acc={best['selection_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
