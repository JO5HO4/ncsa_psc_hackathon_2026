#!/usr/bin/env python3
"""Plot validation improvement from the aligned Qwen ROOT-command contract."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def score(path: Path, model: str) -> int:
    summary = json.loads((path / "pipeline-summary.json").read_text())
    return summary["scores"][model]["passed"]


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} REPOSITORY_ROOT OUTPUT_DIR")
    root, output = map(Path, sys.argv[1:])
    old = root / "results/root-command-qwen35-9b/grid-58288690-row-2-lr1e-5-e15"
    new = root / "results/root-command-qwen35-9b-qwen-root-command-chat-v1/qwen35-9b-qwen-chat-contract-lr1e-5-e15-validation"
    entries = [
        ("Original\nbaseline", score(old, "base"), "#4d4d4d"),
        ("Original\nLoRA", score(old, "lora"), "#9aa0a6"),
        ("Aligned prompt\nbaseline", score(new, "base"), "#4e79a7"),
        ("Aligned prompt\nLoRA", score(new, "lora"), "#2a9d8f"),
    ]
    labels, passed, colors = map(list, zip(*entries))
    figure, axis = plt.subplots(figsize=(8.0, 4.25), layout="constrained")
    bars = axis.bar(range(4), passed, color=colors)
    axis.set_xticks(range(4), labels, fontsize=10)
    axis.set(title="Validation ROOT execution success", ylabel="Passed commands (of 560)", ylim=(0, 600))
    for bar, value in zip(bars, passed):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 12, str(value), ha="center", va="bottom", fontsize=11)
    figure.suptitle("Qwen3.5-9B: refined prompt structure improves executable ROOT output", fontsize=14, fontweight="bold")
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"QWEN_CHAT_CONTRACT_IMPROVEMENT.{suffix}", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
