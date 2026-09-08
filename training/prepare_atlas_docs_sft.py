#!/usr/bin/env python3
"""Render ROOT documentation command pairs into verl-compatible Parquet splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SYSTEM = "Return exactly one executable command requested by the user. Do not execute it or add explanation, Markdown, or a guessed result."
MESSAGE = pa.list_(pa.struct([pa.field("role", pa.string()), pa.field("content", pa.string()), pa.field("tool_calls", pa.null()), pa.field("tool_call_id", pa.null())]))
SCHEMA = pa.schema([pa.field("id", pa.string()), pa.field("data_source", pa.string()), pa.field("api_family", pa.string()), pa.field("messages", MESSAGE), pa.field("tools", pa.string()), pa.field("ground_truth", pa.string())])


def rows(path: Path) -> list[dict[str, object]]:
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        output.append({
            "id": record["id"], "data_source": "atlas-open-data-sft-dataset",
            "api_family": record.get("api_family", "manual_example"),
            "messages": [
                {"role": "system", "content": SYSTEM, "tool_calls": None, "tool_call_id": None},
                {"role": "user", "content": record["question"], "tool_calls": None, "tool_call_id": None},
                {"role": "assistant", "content": record["answer"], "tool_calls": None, "tool_call_id": None},
            ],
            "tools": "[]", "ground_truth": record["answer"],
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        source = args.dataset_root / "sft" / f"root_docs_{split}.jsonl"
        manual = args.dataset_root / "sft" / f"manual_examples_{split}.jsonl"
        rendered = rows(source) + rows(manual)
        pq.write_table(pa.Table.from_pylist(rendered, schema=SCHEMA), args.output_dir / f"{split}.parquet", compression="zstd")
        print(f"wrote {len(rendered)} {split} rows")


if __name__ == "__main__":
    main()
