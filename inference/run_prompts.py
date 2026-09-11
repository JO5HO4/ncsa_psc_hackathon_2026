#!/usr/bin/env python3
"""Generate raw model completions for a JSON or JSONL list of prompts."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any, Iterable

from model_runtime import generate_chat, generate_text, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model ID, local HF model, verl checkpoint, or LoRA adapter")
    parser.add_argument(
        "--base-model",
        default=None,
        help="base model for an adapter-only --model (uses adapter/export metadata when omitted)",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--prompts", type=Path, help="JSON or JSONL input prompts")
    input_group.add_argument("--dataset", help="Hugging Face dataset ID")
    parser.add_argument("--dataset-config", default=None, help="optional Hugging Face dataset configuration")
    parser.add_argument("--split", default="test", help="dataset split used with --dataset (default: test)")
    parser.add_argument("--revision", default=None, help="optional dataset revision on the Hub")
    parser.add_argument("--prompt-field", default="prompt", help="record or dataset column containing prompt text")
    parser.add_argument("--id-field", default=None, help="optional record or dataset column used as the output ID")
    parser.add_argument("--filter-field", default=None, help="optional record or dataset column used to filter inputs")
    parser.add_argument("--filter-value", default=None, help="value required in --filter-field")
    parser.add_argument("--streaming", action="store_true", help="stream a Hub dataset instead of downloading it all")
    parser.add_argument("--output", required=True, type=Path, help="JSONL output path")
    parser.add_argument("--format", choices=("raw", "chat"), default="raw", help="raw text or a model chat template")
    parser.add_argument("--system-prompt", default=None, help="system message used only with --format chat")
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable Qwen3.5 reasoning traces in chat output (default: disabled)",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def read_prompt_records(
    path: Path,
    *,
    prompt_field: str,
    id_field: str | None,
    filter_field: str | None,
    filter_value: str | None,
) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    loaded: Any
    if path.suffix == ".json":
        loaded = json.loads(text)
        records = loaded if isinstance(loaded, list) else [loaded]
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]

    normalized = []
    for index, record in enumerate(records):
        if isinstance(record, str):
            if filter_field is not None:
                raise ValueError("--filter-field requires object records, not string prompt records.")
            normalized.append({"id": index, "prompt": record})
        elif isinstance(record, dict) and isinstance(record.get(prompt_field), str):
            if filter_field is not None and str(record.get(filter_field)) != filter_value:
                continue
            record_id = record.get(id_field, index) if id_field else record.get("id", index)
            normalized.append({"id": record_id, "prompt": record[prompt_field]})
        else:
            raise ValueError(
                f"Prompt record {index} must be a string or an object with a string '{prompt_field}' field."
            )
    return normalized


def read_dataset_records(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    """Yield prompts from a Hugging Face dataset without imposing a task schema."""
    from datasets import load_dataset

    dataset = load_dataset(
        args.dataset,
        name=args.dataset_config,
        split=args.split,
        revision=args.revision,
        streaming=args.streaming,
    )
    for index, row in enumerate(dataset):
        if args.filter_field is not None and str(row.get(args.filter_field)) != args.filter_value:
            continue
        prompt = row.get(args.prompt_field)
        if not isinstance(prompt, str):
            raise ValueError(
                f"Dataset row {index} has no string '{args.prompt_field}' field. "
                "Choose the prompt column with --prompt-field."
            )
        record_id = row.get(args.id_field, index) if args.id_field else index
        yield {"id": record_id, "prompt": prompt}


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if (args.filter_field is None) != (args.filter_value is None):
        raise ValueError("--filter-field and --filter-value must be supplied together")

    records: Iterable[dict[str, Any]]
    input_metadata: dict[str, Any]
    if args.dataset:
        records = read_dataset_records(args)
        input_metadata = {
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "revision": args.revision,
            "prompt_field": args.prompt_field,
            "id_field": args.id_field,
            "streaming": args.streaming,
        }
    else:
        prompts = read_prompt_records(
            args.prompts,
            prompt_field=args.prompt_field,
            id_field=args.id_field,
            filter_field=args.filter_field,
            filter_value=args.filter_value,
        )
        if not prompts:
            raise ValueError(f"No prompts found in {args.prompts}")
        records = prompts
        input_metadata = {
            "prompt_file": str(args.prompts.resolve()),
            "prompt_field": args.prompt_field,
            "id_field": args.id_field,
            "filter_field": args.filter_field,
            "filter_value": args.filter_value,
        }
    if args.limit is not None:
        records = itertools.islice(records, args.limit)

    model, tokenizer, resolved_model, selected_device = load_model(
        args.model, args.device, trust_remote_code=args.trust_remote_code, base_model=args.base_model
    )
    print(f"Loaded {resolved_model} on {selected_device}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "record_type": "metadata",
            "model": resolved_model,
            "device": selected_device,
            "format": args.format,
            "enable_thinking": args.enable_thinking,
            "input": input_metadata,
        }) + "\n")
        generated_count = 0
        for record in records:
            generated_count += 1
            started = time.perf_counter()
            if args.format == "chat":
                messages = []
                if args.system_prompt:
                    messages.append({"role": "system", "content": args.system_prompt})
                messages.append({"role": "user", "content": record["prompt"]})
                output = generate_chat(
                    model, tokenizer, messages,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    enable_thinking=args.enable_thinking,
                )
            else:
                output = generate_text(
                    model, tokenizer, record["prompt"],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            elapsed_seconds = time.perf_counter() - started
            result = {
                "record_type": "completion",
                "id": record["id"],
                "prompt": record["prompt"],
                "output": output,
                "elapsed_seconds": elapsed_seconds,
            }
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            print(json.dumps({"id": result["id"], "elapsed_seconds": round(elapsed_seconds, 2)}))
    if generated_count == 0:
        raise ValueError("No prompt records were available from the selected input.")


if __name__ == "__main__":
    main()
