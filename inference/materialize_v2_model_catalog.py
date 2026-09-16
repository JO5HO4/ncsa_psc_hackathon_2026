#!/usr/bin/env python3
"""Replace V2 gold commands with model completions for ROOT execution scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path, help="Gold V2 split catalog")
    parser.add_argument("--split-manifest", type=Path, default=None, help="Filter a full V2 catalog to this manifest")
    parser.add_argument("--split", choices=("train", "validation", "test"), default=None)
    parser.add_argument("--completions", required=True, type=Path, help="JSONL from inference/run_prompts.py")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    completions: dict[str, str] = {}
    for row in read_jsonl(args.completions):
        if row.get("record_type") != "completion":
            continue
        identifier, output = row.get("id"), row.get("output")
        if not isinstance(identifier, str) or not isinstance(output, str):
            raise ValueError(f"Malformed completion in {args.completions}: {row!r}")
        if identifier in completions:
            raise ValueError(f"Duplicate completion ID: {identifier}")
        completions[identifier] = output

    assignments: dict[str, str] | None = None
    groups: dict[str, str] | None = None
    if args.split_manifest:
        manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
        assignments, groups = manifest["assignments"], manifest["groups"]
        if args.split is None:
            raise ValueError("--split is required with --split-manifest")

    rendered: list[dict[str, object]] = []
    for row in read_jsonl(args.catalog):
        identifier = row.get("id")
        if not isinstance(identifier, str):
            raise ValueError(f"Malformed catalog record: {row!r}")
        if assignments is not None and assignments.get(groups.get(identifier)) != args.split:
            continue
        if identifier not in completions:
            raise ValueError(f"Missing completion for {identifier}")
        rendered.append({**row, "answer": completions.pop(identifier)})
    if completions:
        raise ValueError(f"Completions not in catalog: {sorted(completions)[:5]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rendered:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"Wrote {len(rendered)} model-command records to {args.output}")


if __name__ == "__main__":
    main()
