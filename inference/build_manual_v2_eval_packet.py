#!/usr/bin/env python3
"""Build a paste-ready manual evaluation packet for ChatGPT or Claude."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a ROOT and ATLAS Open Data assistant. Return exactly one executable "
    "shell command, with no prose or Markdown."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True, help="JSONL records with id and prompt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not questions or any(not isinstance(row.get("id"), str) or not isinstance(row.get("prompt"), str) for row in questions):
        raise ValueError("Prompt input must contain nonempty JSONL id/prompt records")

    metadata = {
        "record_type": "metadata",
        "model": "REPLACE_WITH_MODEL_NAME",
        "device": "manual_chat",
        "format": "chat",
        "enable_thinking": False,
        "input": {"benchmark": "ATLAS Open Data ROOT-command V2 held-out test", "question_count": len(questions)},
    }
    template = {
        "record_type": "completion",
        "id": "COPY_THE_QUESTION_ID_EXACTLY",
        "prompt": "COPY_THE_QUESTION_TEXT_EXACTLY",
        "output": "ONE_EXECUTABLE_SHELL_COMMAND_ONLY",
        "elapsed_seconds": 0.0,
    }
    text = """# ATLAS Open Data ROOT-command V2 — manual model evaluation

Paste this entire packet into ChatGPT or Claude. The model must return a **JSONL file only**: no prose, no Markdown fences, and no explanation.

## System prompt

```text
%s
```

## Required response format

Return exactly 159 JSONL lines: first the metadata line below, then one completion line for every question. Preserve each question `id` and `prompt` exactly. Each `output` must be exactly one executable shell command. Set `elapsed_seconds` to `0.0`.

Metadata line:

```json
%s
```

Completion-line shape:

```json
%s
```

## Questions (JSONL)

```jsonl
%s
```
""" % (
        SYSTEM_PROMPT,
        json.dumps(metadata, ensure_ascii=False),
        json.dumps(template, ensure_ascii=False),
        "\n".join(json.dumps(row, ensure_ascii=False) for row in questions),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {len(questions)} questions to {args.output}")


if __name__ == "__main__":
    main()
