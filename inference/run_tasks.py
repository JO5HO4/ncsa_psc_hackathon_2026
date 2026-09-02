#!/usr/bin/env python3
"""Run a trained config-repair model over JSON or JSONL task records.

This script intentionally performs generation only. Applying the patch and
running TRExFitter remain separate evaluator steps.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from model_runtime import generate_chat, load_model

SYSTEM_PROMPT = """You repair native TRExFitter .config files.
Return exactly one unified diff inside <patch> and </patch> tags. Edit only the
task's stated config path. Do not run commands, call tools, or explain the fit.
If the task cannot be completed, still return your best valid patch."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Hugging Face model ID, local model, or verl checkpoint")
    parser.add_argument("--tasks", required=True, type=Path, help="one task JSON file or JSONL file")
    parser.add_argument("--output", required=True, type=Path, help="JSONL file to create")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def read_tasks(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        loaded = json.loads(text)
        return loaded if isinstance(loaded, list) else [loaded]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def prompt_for(task: dict[str, Any]) -> list[dict[str, str]]:
    diagnostics = task.get("diagnostics", [])
    diagnostic_text = "\n".join(
        f"- {item.get('category', 'diagnostic')}: {item.get('message', '')}" for item in diagnostics
    ) or "- No diagnostics were provided."
    constraints = "\n".join(f"- {item}" for item in task.get("constraints", [])) or "- None provided."
    fixture_path = task.get("fixture", {}).get("config_path", "the task config")
    user_prompt = f"""Task ID: {task.get('id', 'unknown')}
Task type: {task.get('task_type', 'repair')}
Config path: {fixture_path}

Physics goal:
{task.get('physics_goal', '')}

Constraints:
{constraints}

Diagnostics:
{diagnostic_text}

Starting config:
```config
{task.get('initial_config', '')}
```
"""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]


def extract_patch(answer: str) -> str | None:
    match = re.search(r"<patch>\s*(.*?)\s*</patch>", answer, flags=re.DOTALL)
    return match.group(1) if match else None


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative")

    tasks = read_tasks(args.tasks)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        raise ValueError(f"No tasks found in {args.tasks}")

    model, tokenizer, resolved_checkpoint, selected_device = load_model(args.checkpoint, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "record_type": "metadata",
            "checkpoint": str(resolved_checkpoint),
            "device": selected_device,
            "task_file": str(args.tasks.resolve()),
        }) + "\n")
        for task in tasks:
            messages = prompt_for(task)
            started = time.perf_counter()
            answer = generate_chat(
                model, tokenizer, messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            elapsed_seconds = time.perf_counter() - started
            record = {
                "record_type": "prediction",
                "task_id": task.get("id"),
                "fixture_path": task.get("fixture", {}).get("config_path"),
                "answer": answer,
                "patch": extract_patch(answer),
                "elapsed_seconds": elapsed_seconds,
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(json.dumps({"task_id": record["task_id"], "elapsed_seconds": round(elapsed_seconds, 2), "has_patch": record["patch"] is not None}))


if __name__ == "__main__":
    main()
