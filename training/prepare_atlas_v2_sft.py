#!/usr/bin/env python3
"""Render the versioned ATLAS ROOT-command V2 catalog into VERL SFT Parquet."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SYSTEM = (
    "You are a ROOT and ATLAS Open Data assistant. Return exactly one executable "
    "shell command, with no prose or Markdown."
)
MESSAGE = pa.list_(pa.struct([
    pa.field("role", pa.string()), pa.field("content", pa.string()),
    pa.field("tool_calls", pa.list_(pa.struct([
        pa.field("id", pa.string()), pa.field("type", pa.string()),
        pa.field("function", pa.struct([
            pa.field("name", pa.string()), pa.field("arguments", pa.string()),
        ])),
    ]))),
    pa.field("tool_call_id", pa.string()),
]))
SCHEMA = pa.schema([
    pa.field("id", pa.string()), pa.field("topic", pa.string()),
    pa.field("source", pa.string()), pa.field("messages", MESSAGE),
    pa.field("tools", pa.string()), pa.field("ground_truth", pa.string()),
    pa.field("source_split", pa.string()),
])


def message(role: str, content: str) -> dict[str, object]:
    return {"role": role, "content": content, "tool_calls": None, "tool_call_id": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = args.split_manifest or args.source.parent / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assignments = manifest.get("assignments")
    groups = manifest.get("groups")
    if not isinstance(assignments, dict) or not isinstance(groups, dict):
        raise ValueError(f"{manifest_path}: expected assignments and groups mappings")

    rows = {"train": [], "validation": []}
    seen: set[str] = set()
    for number, line in enumerate(args.source.read_text(encoding="utf-8").splitlines(), 1):
        record = json.loads(line)
        identifier = record.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError(f"line {number}: invalid or duplicate id")
        seen.add(identifier)
        split = assignments.get(groups.get(identifier))
        if split == "test":
            continue
        if split not in rows:
            raise ValueError(f"{identifier}: unsupported split {split!r}")
        answer = record.get("answer")
        if not isinstance(answer, str) or not answer:
            raise ValueError(f"{identifier}: missing answer")
        rows[split].append({
            "id": identifier, "topic": record.get("topic", ""), "source": record.get("source", ""),
            "messages": [message("system", SYSTEM), message("user", record["question"]), message("assistant", answer)],
            "tools": "[]", "ground_truth": answer, "source_split": split,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rendered in rows.items():
        output = args.output_dir / f"{split}.parquet"
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        pq.write_table(pa.Table.from_pylist(rendered, schema=SCHEMA), output, compression="zstd")
    print(f"Wrote {Counter({split: len(data) for split, data in rows.items()})} to {args.output_dir}")


if __name__ == "__main__":
    main()
