#!/usr/bin/env python3
"""Render execution-verified ROOT command pairs into canonical Parquet splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SYSTEM = "Return exactly one executable command requested by the user. Do not execute it or add explanation, Markdown, or a guessed result."
MESSAGE = pa.list_(pa.struct([pa.field("role", pa.string()), pa.field("content", pa.string()), pa.field("tool_calls", pa.null()), pa.field("tool_call_id", pa.null())]))
EXECUTION = pa.struct([
    pa.field("expected_result", pa.string()),
    pa.field("returncode", pa.int32()),
    pa.field("result_line_count", pa.int32()),
    pa.field("status", pa.string()),
])
SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("api_family", pa.string()),
    pa.field("source", pa.string()),
    pa.field("messages", MESSAGE),
    pa.field("tools", pa.string()),
    pa.field("ground_truth", pa.string()),
    pa.field("execution", EXECUTION),
])


def expected_results(path: Path) -> dict[str, str]:
    return {
        str(record["id"]): str(record["expected_result"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and (record := json.loads(line))
    }


def rows(path: Path, expected: dict[str, str]) -> list[dict[str, object]]:
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        identifier = str(record["id"])
        if identifier not in expected:
            raise ValueError(f"missing verified execution result: {identifier}")
        output.append({
            "id": identifier,
            "api_family": record["api_family"],
            "source": record["source"],
            "messages": [
                {"role": "system", "content": SYSTEM, "tool_calls": None, "tool_call_id": None},
                {"role": "user", "content": record["question"], "tool_calls": None, "tool_call_id": None},
                {"role": "assistant", "content": record["answer"], "tool_calls": None, "tool_call_id": None},
            ],
            "tools": "[]", "ground_truth": record["answer"],
            "execution": {
                "expected_result": expected[identifier],
                "returncode": 0,
                "result_line_count": 1,
                "status": "verified",
            },
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True, help="temporary directory containing generated JSONL splits")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        source = args.source_dir / f"root_docs_generated_{split}.jsonl"
        expected = expected_results(args.dataset_root / "benchmarks" / f"root_docs_generated_{split}_expected.jsonl")
        rendered = rows(source, expected)
        pq.write_table(pa.Table.from_pylist(rendered, schema=SCHEMA), args.output_dir / f"{split}.parquet", compression="zstd")
        print(f"wrote {len(rendered)} {split} rows")


if __name__ == "__main__":
    main()
