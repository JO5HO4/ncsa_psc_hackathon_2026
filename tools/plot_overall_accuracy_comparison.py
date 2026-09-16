#!/usr/bin/env python3
"""Render an overall-accuracy comparison chart from a CSV summary."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="ATLAS Open Data SFT: overall benchmark accuracy")
    args = parser.parse_args()

    with args.summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("summary CSV is empty")

    labels = [
        row["model"] + (f"\n{row['plot_note']}" if row.get("plot_note") else "")
        for row in rows
    ]
    values = [float(row["accuracy"]) for row in rows]
    counts = [f"{row['passed']}/{row['total']}\n{value:.1%}" for row, value in zip(rows, values)]
    colors = [row.get("color", "#4C78A8") for row in rows]

    figure, axis = plt.subplots(figsize=(10, 6))
    bars = axis.bar(labels, values, color=colors, width=0.68)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Execution pass rate")
    axis.set_title(args.title, pad=20)
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="x", labelrotation=15)
    for bar, label, value in zip(bars, counts, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.025, 1.02),
            label,
            ha="center",
            va="bottom",
            fontsize=10,
        )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
