#!/usr/bin/env python3
"""Write a deterministic prefix of an SFT Parquet file for overfit checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows", required=True, type=int)
    args = parser.parse_args()

    if args.rows < 1:
        raise ValueError("--rows must be positive")
    table = pq.read_table(args.input)
    if len(table) < args.rows:
        raise ValueError(f"requested {args.rows} rows, but {args.input} contains only {len(table)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.slice(0, args.rows), args.output, compression="zstd")
    print(f"wrote rows [0, {args.rows}) from {args.input} to {args.output}")


if __name__ == "__main__":
    main()
