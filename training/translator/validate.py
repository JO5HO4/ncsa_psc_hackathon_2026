"""Checkup/lint functions, mirroring the idiom in data/root_io/validate_tasks.py:
plain functions, fail-fast bare ``ValueError(f"{id}: ...")``, no exception
hierarchy, no error collection.

Two checkpoints:

- ``validate_source_records`` runs *before* translation, on the canonical
  ``SourceRecord`` shape (regardless of which raw format it was adapted
  from) -- catches authoring mistakes like duplicate or empty ids/questions.
- ``validate_output_rows`` runs *after* translation -- re-checks every row's
  assistant content against the task's output contract, catching translator
  or profile bugs before a row is written to disk.
"""
from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def validate_source_records(records: list) -> None:
    seen = set()
    for record in records:
        if not record.id:
            raise ValueError("Source record missing a non-empty id")
        if record.id in seen:
            raise ValueError(f"{record.id}: duplicate id")
        seen.add(record.id)
        if not record.question.strip():
            raise ValueError(f"{record.id}: empty question")
        if not record.answer.strip():
            raise ValueError(f"{record.id}: empty answer")


def validate_output_rows(rows: list, task_profile) -> None:
    for row in rows:
        identifier = row.get("id", "<unknown>")
        messages = row.get("messages") or []
        if [message["role"] for message in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"{identifier}: expected system/user/assistant messages")
        try:
            task_profile.validate_answer(messages[-1]["content"])
        except ValueError as error:
            raise ValueError(f"{identifier}: {error}") from error


def main() -> None:
    import argparse

    from .records import ADAPTERS
    from .task_profiles import get_task_profile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check-source", "check-output"])
    parser.add_argument("path", type=Path)
    parser.add_argument("--source-format", choices=sorted(ADAPTERS), help="required for check-source")
    parser.add_argument("--task-profile", help="required for check-output")
    args = parser.parse_args()
    records = read_jsonl(args.path)
    if args.mode == "check-source":
        if not args.source_format:
            parser.error("--source-format is required for check-source")
        adapted = [ADAPTERS[args.source_format](record) for record in records]
        validate_source_records(adapted)
    else:
        if not args.task_profile:
            parser.error("--task-profile is required for check-output")
        validate_output_rows(records, get_task_profile(args.task_profile))
    print(f"Validated {len(records)} record(s) in {args.path}")


if __name__ == "__main__":
    main()
