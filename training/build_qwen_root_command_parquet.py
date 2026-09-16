#!/usr/bin/env python3
"""Build VERL-ready Parquet files from the derived Qwen ROOT-command JSONL splits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path, expected_count: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != expected_count or len({row["id"] for row in rows}) != expected_count:
        raise ValueError(f"Expected {expected_count} unique records in {path}; found {len(rows)}")
    prompts = {row["messages"][0]["content"] for row in rows}
    if len(prompts) != 1:
        raise ValueError(f"Expected one shared system prompt in {path}")
    for row in rows:
        if row.get("format_version") != "qwen-root-command-chat-v1" or row.get("tools") != "[]":
            raise ValueError(f"Invalid format contract for {row['id']}")
        if row.get("enable_thinking") is not False:
            raise ValueError(f"Thinking must be disabled explicitly for {row['id']}")
        messages = row["messages"]
        if [item["role"] for item in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"Invalid message roles for {row['id']}")
        if messages[-1]["content"] != row["ground_truth"]:
            raise ValueError(f"Target mismatch for {row['id']}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [args.output_dir / "train.parquet", args.output_dir / "validation.parquet", args.output_dir / "parquet-manifest.json"]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Refusing to overwrite an existing derived Parquet artifact")
    train, validation = read(args.train_jsonl, 4480), read(args.validation_jsonl, 560)
    if set(row["id"] for row in train) & set(row["id"] for row in validation):
        raise ValueError("Training and validation IDs overlap")
    for name, rows in (("train", train), ("validation", validation)):
        pq.write_table(pa.Table.from_pylist(rows), args.output_dir / f"{name}.parquet", compression="zstd")
    manifest = {
        "format_version": "qwen-root-command-chat-v1",
        "train": {"records": len(train), "jsonl_sha256": digest(args.train_jsonl), "parquet": "train.parquet"},
        "validation": {"records": len(validation), "jsonl_sha256": digest(args.validation_jsonl), "parquet": "validation.parquet"},
        "notes": "Derived from canonical ROOT-command splits; questions and assistant command targets are preserved.",
    }
    outputs[-1].write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(train)} training and {len(validation)} validation rows to {args.output_dir}")


if __name__ == "__main__":
    main()
