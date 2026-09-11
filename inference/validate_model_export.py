#!/usr/bin/env python3
"""Verify that an inference export has complete weights or a PEFT adapter."""

from __future__ import annotations

import argparse
from pathlib import Path


WEIGHTS = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    return parser.parse_args()


def nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def main() -> None:
    model_dir = parse_args().model_dir.resolve()
    if not nonempty(model_dir / "config.json"):
        raise ValueError(f"Missing or empty config.json in {model_dir}")
    if any(nonempty(model_dir / name) for name in WEIGHTS):
        print(f"Validated full Hugging Face export: {model_dir}")
        return

    adapter = model_dir / "lora_adapter"
    if nonempty(adapter / "adapter_config.json") and nonempty(adapter / "adapter_model.safetensors"):
        print(f"Validated LoRA adapter export: {adapter}")
        return
    raise ValueError(
        f"Incomplete inference export at {model_dir}: expected full model weights or "
        "lora_adapter/{adapter_config.json,adapter_model.safetensors}."
    )


if __name__ == "__main__":
    main()
