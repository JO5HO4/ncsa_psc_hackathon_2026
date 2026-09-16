"""Single entry point: read -> adapt -> checkup(source) -> translate -> checkup(output) -> write.

Example (rendering a freshly authored simple-QA dataset for Qwen3.5):

    python -m training.translator.cli \\
      --source data/root_io/my_new_questions.jsonl --source-format simple-qa \\
      --task-profile root_command --model-profile qwen3.5 \\
      --split train --output-dir data/derived/my-new-dataset

This replaces training/prepare_qwen_root_command_sft.py and
training/build_qwen_root_command_parquet.py, which did the same two steps
(rewrite messages/tools/enable_thinking, then write Parquet) as one-off,
hardcoded-to-one-model-and-one-task scripts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .build_parquet import sha256, write_jsonl, write_parquet
from .model_profiles import get_model_profile
from .records import ADAPTERS
from .task_profiles import get_task_profile
from .translate import render_split
from .validate import validate_output_rows, validate_source_records

REPO_ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def run(
    *,
    source: Path,
    source_format: str,
    task_name: str,
    model_name: str,
    split: str,
    output_dir: Path,
    repo_root: Path = REPO_ROOT,
    expected_count: Optional[int] = None,
) -> dict:
    adapt = ADAPTERS[source_format]
    task = get_task_profile(task_name)
    model = get_model_profile(model_name)

    records = [adapt(record) for record in read_jsonl(source)]
    validate_source_records(records)
    rows = render_split(records, task, model)
    validate_output_rows(rows, task)

    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} rows; found {len(rows)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{split}.jsonl"
    parquet_path = output_dir / f"{split}.parquet"
    write_jsonl(rows, jsonl_path)
    write_parquet(rows, parquet_path)

    manifest = {
        "format_version": task.format_version,
        "source": repo_relative(source, repo_root),
        "source_sha256": sha256(source),
        "output": repo_relative(jsonl_path, repo_root),
        "output_sha256": sha256(jsonl_path),
        "records": len(rows),
        "system_prompt": task.system_prompt,
        "tools": model.tools_value,
        "enable_thinking": model.enable_thinking,
        "notes": (
            "Questions and assistant answers are preserved from the source split; "
            "only the system prompt and tools/enable_thinking policy are added by the translator."
        ),
    }
    manifest_path = output_dir / f"{split}.manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(rows)} rows: {jsonl_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-format", choices=sorted(ADAPTERS), required=True)
    parser.add_argument("--task-profile", required=True)
    parser.add_argument("--model-profile", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--expected-count", type=int, default=None)
    args = parser.parse_args()
    run(
        source=args.source,
        source_format=args.source_format,
        task_name=args.task_profile,
        model_name=args.model_profile,
        split=args.split,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        expected_count=args.expected_count,
    )


if __name__ == "__main__":
    main()
