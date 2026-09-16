#!/usr/bin/env python3
"""Summarize and plot V2 command-execution results by model and topic."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path, help="V2 test catalog used for execution")
    parser.add_argument("--result", action="append", required=True, metavar="LABEL=PATH",
                        help="execution JSONL, repeat once per model")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    topics = {row["id"]: row["topic"] for row in rows(args.catalog)}
    reports: dict[str, dict[str, bool]] = {}
    for value in args.result:
        if "=" not in value:
            raise ValueError("--result must be LABEL=PATH")
        # Labels may themselves contain '=' (for example, a LoRA rank label
        # such as 'V2 SFT (r=32)'). Paths in this interface are absolute or
        # relative filesystem paths and do not use '='.
        label, path = value.rsplit("=", 1)
        report = {row["id"]: row.get("status") == "passed" for row in rows(Path(path))}
        if set(report) != set(topics):
            missing, unexpected = set(topics) - set(report), set(report) - set(topics)
            raise ValueError(f"{label}: IDs differ from catalog; missing={len(missing)}, unexpected={len(unexpected)}")
        reports[label] = report

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overall: list[dict[str, object]] = []
    per_topic: list[dict[str, object]] = []
    topic_order = sorted(set(topics.values()))
    for label, report in reports.items():
        passed = sum(report.values())
        overall.append({"model": label, "passed": passed, "total": len(report), "success_rate": passed / len(report)})
        for topic in topic_order:
            selected = [identifier for identifier, value in topics.items() if value == topic]
            count = sum(report[identifier] for identifier in selected)
            per_topic.append({"model": label, "topic": topic, "passed": count, "total": len(selected), "success_rate": count / len(selected)})

    with (args.output_dir / "overall.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=overall[0].keys())
        writer.writeheader(); writer.writerows(overall)
    with (args.output_dir / "by_topic.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_topic[0].keys())
        writer.writeheader(); writer.writerows(per_topic)
    (args.output_dir / "summary.json").write_text(json.dumps({"overall": overall, "by_topic": per_topic}, indent=2) + "\n")

    # Keep the first four colors stable for the existing Qwen comparison and
    # provide distinct colors for manually evaluated frontier models too.
    colors = ["#9AA0A6", "#147D92", "#E07A35", "#7A5195", "#2A9D55", "#D1497A"]
    if len(reports) > len(colors):
        colors.extend(plt.get_cmap("tab20").colors[:len(reports) - len(colors)])
    figure, axis = plt.subplots(figsize=(10, 5.5))
    bars = axis.bar([row["model"] for row in overall], [row["success_rate"] for row in overall], color=colors[:len(overall)])
    axis.set_ylim(0, 1.05); axis.set_ylabel("Execution success rate"); axis.set_title("V2 held-out test set (158 commands)")
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}"); axis.grid(axis="y", alpha=.25); axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="x", labelsize=8)
    for tick in axis.get_xticklabels():
        tick.set_rotation(18)
        tick.set_ha("right")
    for bar, row in zip(bars, overall):
        axis.text(bar.get_x() + bar.get_width()/2, bar.get_height() + .025, f"{row['passed']}/{row['total']}\n{row['success_rate']:.1%}", ha="center", va="bottom")
    figure.tight_layout(); figure.savefig(args.output_dir / "overall_success_rate.png", dpi=200); figure.savefig(args.output_dir / "overall_success_rate.pdf"); plt.close(figure)

    width = .8 / len(reports)
    figure, axis = plt.subplots(figsize=(11, max(6, .42 * len(topic_order) + 2)))
    positions = list(range(len(topic_order)))
    for index, (label, report) in enumerate(reports.items()):
        values = [next(row["success_rate"] for row in per_topic if row["model"] == label and row["topic"] == topic) for topic in topic_order]
        offset = (index - (len(reports) - 1) / 2) * width
        axis.barh([position + offset for position in positions], values, height=width, label=label, color=colors[index])
    axis.set_yticks(positions, [topic.replace("_", " ") for topic in topic_order]); axis.set_xlim(0, 1.0)
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}"); axis.set_xlabel("Execution success rate"); axis.set_title("V2 held-out success rate by topic")
    axis.grid(axis="x", alpha=.25); axis.spines[["top", "right"]].set_visible(False); axis.legend(loc="lower right")
    figure.tight_layout(); figure.savefig(args.output_dir / "success_rate_by_topic.png", dpi=200); figure.savefig(args.output_dir / "success_rate_by_topic.pdf"); plt.close(figure)
    print(json.dumps(overall))


if __name__ == "__main__":
    main()
