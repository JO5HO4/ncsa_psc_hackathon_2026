#!/usr/bin/env python3
"""Render ROOT source-task JSONL into verl-compatible SFT Parquet splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SYSTEM_PROMPT = (
    "You are a careful high-energy-physics assistant. Answer ROOT questions "
    "accurately and concisely. Do not invent unsupported details."
)
MESSAGE_SCHEMA = pa.list_(
    pa.struct(
        [
            pa.field("role", pa.string()),
            pa.field("content", pa.string()),
            pa.field(
                "tool_calls",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("id", pa.string()),
                            pa.field("type", pa.string()),
                            pa.field(
                                "function",
                                pa.struct(
                                    [
                                        pa.field("name", pa.string()),
                                        pa.field("arguments", pa.string()),
                                    ]
                                ),
                            ),
                        ]
                    )
                ),
            ),
            pa.field("tool_call_id", pa.string()),
        ]
    )
)
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("data_source", pa.string()),
        pa.field("task_family", pa.string()),
        pa.field("topic", pa.string()),
        pa.field("question_kind", pa.string()),
        pa.field("messages", MESSAGE_SCHEMA),
        pa.field("tools", pa.string()),
        pa.field("ground_truth", pa.string()),
        pa.field("required_facts", pa.string()),
        pa.field("forbidden_claims", pa.string()),
        pa.field("provenance", pa.string()),
        pa.field("source_split", pa.string()),
        pa.field("source_schema_version", pa.string()),
        pa.field("review_status", pa.string()),
    ]
)


def message(role: str, content: str) -> dict[str, object]:
    return {"role": role, "content": content, "tool_calls": None, "tool_call_id": None}


def render(record: dict[str, object]) -> dict[str, object]:
    required = {"id", "question", "answer", "split", "training_use"}
    missing = sorted(required - record.keys())
    if missing:
        raise ValueError(f"{record.get('id', '<unknown>')}: missing {', '.join(missing)}")
    if "sft" not in record["training_use"]:
        raise ValueError(f"{record['id']}: record is not marked for SFT")

    return {
        "id": record["id"],
        "data_source": "root-sft-dataset",
        "task_family": "root_knowledge",
        "topic": record.get("topic", ""),
        "question_kind": record.get("question_kind", ""),
        "messages": [
            message("system", SYSTEM_PROMPT),
            message("user", record["question"]),
            message("assistant", record["answer"]),
        ],
        "tools": "[]",
        "ground_truth": record["answer"],
        "required_facts": json.dumps(record.get("required_facts", []), sort_keys=True),
        "forbidden_claims": json.dumps(record.get("forbidden_claims", []), sort_keys=True),
        "provenance": json.dumps(record.get("provenance", []), sort_keys=True),
        "source_split": record["split"],
        "source_schema_version": record.get("schema_version", ""),
        "review_status": record.get("review_status", ""),
    }


def load_records(source: Path) -> dict[str, list[dict[str, object]]]:
    splits = {"train": [], "validation": []}
    seen_ids: set[str] = set()
    with source.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = record.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"line {line_number}: missing non-empty id")
            if record_id in seen_ids:
                raise ValueError(f"line {line_number}: duplicate id {record_id}")
            seen_ids.add(record_id)
            split = record.get("split")
            if split == "test":
                continue
            if split not in splits:
                raise ValueError(f"{record_id}: unsupported split {split!r}")
            splits[split].append(render(record))
    if not splits["train"] or not splits["validation"]:
        raise ValueError("both train and validation splits must contain SFT records")
    return splits


def write_splits(splits: dict[str, list[dict[str, object]]], output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        output = output_dir / f"{split}.parquet"
        if output.exists() and not overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
        pq.write_table(pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA), output, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    splits = load_records(args.source)
    write_splits(splits, args.output_dir, args.overwrite)
    counts = Counter({split: len(rows) for split, rows in splits.items()})
    print(f"Wrote {counts['train']} train and {counts['validation']} validation rows to {args.output_dir}")


if __name__ == "__main__":
    main()
