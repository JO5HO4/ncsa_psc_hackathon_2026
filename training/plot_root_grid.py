#!/usr/bin/env python3
"""Plot ROOT execution success and final validation loss for a grid campaign."""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_run(path: Path, model: str):
    cfg = dict(line.split("=", 1) for line in (path / "configuration.txt").read_text().splitlines() if "=" in line)
    summary = json.loads((path / "pipeline-summary.json").read_text())
    log = (path / "02-train-export.log").read_text()
    values = re.findall(r"step:(\d+) - val/loss:([\d.eE+-]+)", log)
    if not values:
        raise ValueError(f"No validation losses in {path}")
    return {
        "label": f"{model}\n{cfg['LR']} · {cfg['TOTAL_EPOCHS']} ep",
        "accuracy": summary["scores"]["lora"]["accuracy"] * 100,
        "passed": summary["scores"]["lora"]["passed"],
        "loss": float(values[-1][1]),
    }


def read_baseline(path: Path):
    summary = json.loads((path / "pipeline-summary.json").read_text())
    return {
        "label": "9B base\nno training",
        "accuracy": summary["scores"]["base"]["accuracy"] * 100,
        "passed": summary["scores"]["base"]["passed"],
        "loss": None,
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} RESULTS_ROOT")
    root = Path(sys.argv[1]).resolve()
    nine_root = root / "root-command-qwen35-9b"
    small_root = root / "root-command-qwen35-0.8b"
    grid_paths = sorted(nine_root.glob("grid-58288690-row-*"))
    runs = [read_run(path, "9B") for path in grid_paths]
    if len(runs) != 3:
        raise SystemExit(f"Expected three 9B grid runs below {nine_root}; found {len(runs)}")
    runs.insert(0, read_baseline(grid_paths[-1]))
    small_runs = sorted(small_root.glob("qwen35-0.8b-*-validation"))
    if small_runs:
        runs.insert(1, read_run(small_runs[-1], "0.8B"))
    labels = [run["label"] for run in runs]
    colors = ["#4d4d4d", "#e76f51", "#9aa0a6", "#4e79a7", "#2a9d8f"][:len(runs)]
    fig, (success, loss) = plt.subplots(1, 2, figsize=(12.8, 5.0), layout="constrained")
    positions = list(range(len(runs)))
    bars = success.bar(positions, [r["accuracy"] for r in runs], color=colors)
    success.set_xticks(positions, labels)
    success.set(title="Validation ROOT execution success", ylabel="Success rate (%)", ylim=(0, 50))
    for bar, run in zip(bars, runs):
        success.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{run['passed']}/560\n{run['accuracy']:.2f}%", ha="center", va="bottom", fontsize=9)
    success.tick_params(axis="x", labelsize=9)
    trained = [(i, run) for i, run in enumerate(runs) if run["loss"] is not None]
    bars = loss.bar([i for i, _ in trained], [run["loss"] for _, run in trained],
                    color=[colors[i] for i, _ in trained])
    loss.set_xticks(positions, labels)
    loss.set(title="Final validation loss", ylabel="Cross-entropy loss", ylim=(0, 0.025))
    for bar, (_, run) in zip(bars, trained):
        loss.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0006,
                  f"{run['loss']:.4f}", ha="center", va="bottom", fontsize=9)
    loss.tick_params(axis="x", labelsize=9)
    loss.text(0, 0.001, "not trained", ha="center", va="bottom", fontsize=9, color="#4d4d4d")
    fig.suptitle("Qwen3.5 ROOT-command LoRA comparison", fontsize=14, fontweight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"ROOT_QWEN_MODEL_COMPARISON.{suffix}", dpi=200)
    plt.close(fig)
    print(root / "ROOT_QWEN_MODEL_COMPARISON.png")


if __name__ == "__main__":
    main()
