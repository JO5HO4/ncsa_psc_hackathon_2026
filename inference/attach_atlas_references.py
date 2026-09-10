#!/usr/bin/env python3
"""Attach canonical ATLAS command targets to a run_prompts JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completions", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    queries = jsonl(args.dataset_root / "data/tasks/query_tasks.jsonl")
    results = {str(row["id"]): row["answer"] for row in queries}
    commands = {str(row["id"]): row["reference_actions"][0]["command"] for row in queries}

    rows = jsonl(args.completions)
    completion_count = 0
    for row in rows:
        if row.get("record_type") != "completion":
            continue
        task_id = str(row["id"])
        if task_id not in results or task_id not in commands:
            raise ValueError(f"completion {task_id!r} has no ATLAS reference")
        row["reference_command"] = commands[task_id]
        row["expected_result"] = results[task_id]
        completion_count += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(f"wrote {completion_count} annotated completions to {args.output}")


if __name__ == "__main__":
    main()
