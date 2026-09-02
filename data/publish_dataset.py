#!/usr/bin/env python3
"""Publish one reviewed JSON, JSONL, or Parquet split to a Hugging Face dataset."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="local .json, .jsonl, or .parquet file")
    parser.add_argument("--repo-id", required=True, help="target dataset repository, for example ho22joshua/trex-config-tasks")
    parser.add_argument("--split", required=True, help="target split, such as train, validation, or test")
    parser.add_argument("--config-name", default="default", help="dataset configuration name")
    parser.add_argument("--private", action="store_true", help="create or update a private dataset repository")
    args = parser.parse_args()

    if not args.source.is_file():
        raise FileNotFoundError(f"Dataset source does not exist: {args.source}")
    suffix = args.source.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        builder = "json"
    elif suffix == ".parquet":
        builder = "parquet"
    else:
        raise ValueError("--source must end in .json, .jsonl, or .parquet")

    from datasets import load_dataset

    dataset = load_dataset(builder, data_files=str(args.source), split="train")
    dataset.push_to_hub(
        args.repo_id,
        config_name=args.config_name,
        split=args.split,
        private=args.private,
    )
    print(
        f"Published {len(dataset)} rows from {args.source} to "
        f"{args.repo_id} ({args.config_name}/{args.split})."
    )


if __name__ == "__main__":
    main()
