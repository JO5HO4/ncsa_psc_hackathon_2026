#!/usr/bin/env python3
"""Validate ROOT-I/O source records against schema and trajectory invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA_PATH = Path(__file__).with_name("schema") / "task.schema.json"
ZERO_SHA256 = "0" * 64


def read_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                yield record
        return

    value = json.loads(path.read_text(encoding="utf-8"))
    records = value if isinstance(value, list) else [value]
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {index}: expected a JSON object")
        yield record


def validate_record(record: dict[str, Any], schema: dict[str, Any]) -> None:
    import jsonschema

    jsonschema.validate(record, schema)
    record_id = record["id"]
    trajectory = record["trajectory"]
    if trajectory[0].get("role") != "user" or trajectory[0].get("content") != record["user_request"]:
        raise ValueError(f"{record_id}: first turn must reproduce user_request")
    if trajectory[-1].get("role") != "assistant" or not trajectory[-1].get("content"):
        raise ValueError(f"{record_id}: final turn must be an assistant answer")
    if trajectory[-1]["content"] != record["expected"]["reference_answer"]:
        raise ValueError(f"{record_id}: final answer must equal expected.reference_answer")

    allowed = set(record["allowed_tools"])
    pending: dict[str, str] = {}
    completed: set[str] = set()
    for turn in trajectory:
        call = turn.get("tool_call")
        if call is not None:
            call_id = call["id"]
            if call_id in pending or call_id in completed:
                raise ValueError(f"{record_id}: duplicate tool-call ID {call_id}")
            if call["name"] not in allowed:
                raise ValueError(f"{record_id}: tool {call['name']} is not allowed")
            pending[call_id] = call["name"]
        if turn["role"] == "tool":
            call_id = turn["tool_call_id"]
            if call_id not in pending:
                raise ValueError(f"{record_id}: result has no unmatched call: {call_id}")
            completed.add(call_id)
            del pending[call_id]
    if pending:
        raise ValueError(f"{record_id}: calls without results: {sorted(pending)}")
    if not completed:
        raise ValueError(f"{record_id}: trajectory must execute at least one tool")

    if (
        record["metadata"]["review_status"] == "reviewed"
        and record["fixture"]["sha256"] == ZERO_SHA256
    ):
        raise ValueError(f"{record_id}: reviewed records cannot use the placeholder SHA-256")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    count = 0
    for path in args.paths:
        for record in read_records(path):
            validate_record(record, schema)
            count += 1
    print(f"Validated {count} ROOT-I/O source record(s).")


if __name__ == "__main__":
    main()
