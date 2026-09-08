#!/usr/bin/env python3
"""Score prompt completions against the ATLAS ROOT query-task verifier."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completions", required=True, type=Path, help="JSONL from inference/run_prompts.py")
    parser.add_argument("--dataset-root", required=True, type=Path, help="atlas-open-data-sft-dataset checkout")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=None,
        help="benchmark manifest (default: benchmarks/base_model_test.json under --dataset-root)",
    )
    parser.add_argument("--output", required=True, type=Path, help="score-report JSON")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    benchmark_path = (args.benchmark or dataset_root / "benchmarks/base_model_test.json").resolve()
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    query_ids = benchmark["query_task_ids"]
    query_tasks = dataset_root / benchmark["task_sources"]["query"]
    verifier = dataset_root / "verifiers/verify_query_final.py"

    rows = load_jsonl(args.completions)
    metadata = next((row for row in rows if row.get("record_type") == "metadata"), {})
    completions = {str(row["id"]): row["output"] for row in rows if row.get("record_type") == "completion"}
    results = []
    for task_id in query_ids:
        answer = completions.get(task_id)
        if answer is None:
            results.append({"id": task_id, "passed": False, "reason": "missing_completion"})
            continue
        completed = subprocess.run(
            [sys.executable, str(verifier), "--task", task_id, "--answer", answer, "--tasks", str(query_tasks)],
            text=True,
            capture_output=True,
        )
        results.append({
            "id": task_id,
            "passed": completed.returncode == 0,
            "model_output": answer,
            "verifier_output": completed.stdout.strip(),
            "verifier_error": completed.stderr.strip(),
        })

    passed = sum(result["passed"] for result in results)
    report = {
        "benchmark": benchmark["name"],
        "model": metadata.get("model"),
        "query_total": len(results),
        "query_passed": passed,
        "query_accuracy": passed / len(results) if results else 0.0,
        "artifact_tasks_not_scored": benchmark["artifact_task_ids"],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("benchmark", "model", "query_total", "query_passed", "query_accuracy")}))


if __name__ == "__main__":
    main()
