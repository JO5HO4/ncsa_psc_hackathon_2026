"""Write a translated split to disk: JSONL + Parquet, with a no-overwrite policy.

Generalizes the write-side of the two scripts this package replaces
(``training/prepare_qwen_root_command_sft.py`` and
``training/build_qwen_root_command_parquet.py``) into reusable primitives for
any task/model combination and any split size. Manifest-shape decisions stay
in ``cli.py``, which knows about the task/model that produced the rows.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(rows: list, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows))


def write_parquet(rows: list, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
