#!/usr/bin/env python3
"""Derive a Qwen chat-contract JSONL training split from canonical ROOT commands."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "Return exactly one executable ROOT command on one line. "
    "The entire response must begin with root -l -b -q -e ' and end with one matching single quote. "
    "Inside it, print exactly one RESULT=<value> line and call gSystem->Exit(0). "
    "Do not add Markdown, backticks, explanations, XML, JSON, tool calls, or any other text."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive(source: Path, output: Path) -> int:
    records = [json.loads(line) for line in source.read_text().splitlines() if line]
    seen = set()
    rendered = []
    for record in records:
        identifier = record["id"]
        if identifier in seen:
            raise ValueError(f"Duplicate id: {identifier}")
        seen.add(identifier)
        messages = record["messages"]
        if [message["role"] for message in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"Unexpected message roles for {identifier}")
        command = messages[-1]["content"]
        if command != record["ground_truth"] or "\n" in command:
            raise ValueError(f"Invalid command target for {identifier}")
        if not command.startswith("root -l -b -q -e '") or not command.endswith("'"):
            raise ValueError(f"Unexpected ROOT command wrapper for {identifier}")
        updated = dict(record)
        updated["messages"] = [
            {"role": "system", "content": SYSTEM_PROMPT, "tool_calls": None, "tool_call_id": None},
            messages[1],
            messages[2],
        ]
        # Empty tools deliberately selects Qwen's normal chat template, rather
        # than the XML function-calling template.
        updated["tools"] = "[]"
        updated["enable_thinking"] = False
        updated["format_version"] = "qwen-root-command-chat-v1"
        rendered.append(updated)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in rendered))
    return len(rendered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    count = derive(args.source, args.output)
    manifest = args.manifest or args.output.with_name("manifest.json")
    manifest.write_text(json.dumps({
        "format_version": "qwen-root-command-chat-v1",
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "records": count,
        "system_prompt": SYSTEM_PROMPT,
        "tools": "[]",
        "enable_thinking": False,
        "notes": "Questions and assistant command targets are byte-for-byte preserved from the canonical train split.",
    }, indent=2) + "\n")
    print(f"Wrote {count} records: {args.output}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
