#!/usr/bin/env python3
"""Check that a checkpoint can load and produce one config-repair response."""

from __future__ import annotations

import argparse
import json

from model_runtime import generate_chat, load_model
from run_tasks import SYSTEM_PROMPT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Hugging Face model ID, local model, or verl checkpoint")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    model, tokenizer, resolved_checkpoint, selected_device = load_model(args.checkpoint, args.device)
    answer = generate_chat(
        model,
        tokenizer,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Return a unified diff that changes `# TODO` to `# done` in configs/example.config."},
        ],
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )
    print(json.dumps({"checkpoint": str(resolved_checkpoint), "device": selected_device, "answer": answer}, indent=2))


if __name__ == "__main__":
    main()
