#!/usr/bin/env python3
"""Compose Hyy tool trajectories and ATLAS V2 commands into one SFT corpus."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import zip_longest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("source_dataset", pa.string()),
    pa.field("messages", pa.string()),
    pa.field("tools", pa.string()),
    pa.field("enable_thinking", pa.bool_()),
])


def decode(value: object, field: str, identifier: str) -> list[object]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError(f"{identifier}: {field} must be a list")
    # Hyy's Arrow JSON extension materializes each message as a JSON string;
    # ATLAS records already materialize them as dictionaries.
    return [json.loads(item) if isinstance(item, str) else item for item in value]


def load(path: Path, source: str) -> list[dict[str, object]]:
    field_names = set(pq.read_schema(path).names)
    identifier_field = "id" if "id" in field_names else "uuid" if "uuid" in field_names else None
    if identifier_field is None:
        raise ValueError(f"{path}: expected an id or uuid field")
    columns = [identifier_field, "messages", "tools"]
    rows = pq.read_table(path, columns=columns).to_pylist()
    rendered: list[dict[str, object]] = []
    for number, row in enumerate(rows, 1):
        identifier = row.get(identifier_field)
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"{path}:{number}: missing {identifier_field}")
        messages = decode(row.get("messages"), "messages", identifier)
        tools = decode(row.get("tools"), "tools", identifier)
        if not any(turn.get("role") == "assistant" for turn in messages if isinstance(turn, dict)):
            raise ValueError(f"{identifier}: trajectory has no assistant turn")
        rendered.append({
            "id": f"{source}:{identifier}",
            "source_dataset": source,
            "messages": json.dumps(messages, separators=(",", ":"), ensure_ascii=False),
            "tools": json.dumps(tools, separators=(",", ":"), ensure_ascii=False),
            "enable_thinking": False,
        })
    return rendered


def interleave(hyy: list[dict[str, object]], atlas: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for hyy_row, atlas_row in zip_longest(hyy, atlas):
        if hyy_row is not None:
            output.append(hyy_row)
        if atlas_row is not None:
            output.append(atlas_row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hyy-dir", type=Path, required=True)
    parser.add_argument("--atlas-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation"):
        output = args.output_dir / f"{split}.parquet"
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        hyy = load(args.hyy_dir / f"{split}.parquet", "hyy-sft")
        atlas = load(args.atlas_dir / f"{split}.parquet", "atlas-v2")
        mixed = interleave(hyy, atlas)
        ids = [row["id"] for row in mixed]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{split}: duplicate source-qualified IDs")
        pq.write_table(pa.Table.from_pylist(mixed, schema=SCHEMA), output, compression="zstd")
        counts[split] = dict(Counter(row["source_dataset"] for row in mixed))

    (args.output_dir / "manifest.json").write_text(
        json.dumps({"sources": ["hyy-sft", "atlas-v2"], "counts": counts}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote mixed SFT splits to {args.output_dir}: {counts}")


if __name__ == "__main__":
    main()
