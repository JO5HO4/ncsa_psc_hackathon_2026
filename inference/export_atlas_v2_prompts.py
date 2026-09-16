#!/usr/bin/env python3
"""Export one V2 catalog split as JSONL prompts for model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True, help="V2 candidates.jsonl")
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.split_manifest or args.catalog.parent / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assignments = manifest.get("assignments")
    groups = manifest.get("groups")
    if not isinstance(assignments, dict) or not isinstance(groups, dict):
        raise ValueError(f"{manifest_path}: expected assignments and groups mappings")

    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(args.catalog.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        identifier = row.get("id")
        question = row.get("question")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError(f"{args.catalog}:{line_number}: invalid or duplicate id")
        if not isinstance(question, str) or not question:
            raise ValueError(f"{identifier}: missing question")
        seen.add(identifier)
        if assignments.get(groups.get(identifier)) == args.split:
            records.append({"id": identifier, "prompt": question})

    if not records:
        raise ValueError(f"No {args.split} records found in {args.catalog}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row) + "\n")
    print(f"Wrote {len(records)} {args.split} prompts to {args.output}")


if __name__ == "__main__":
    main()
