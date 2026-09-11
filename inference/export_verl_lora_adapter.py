#!/usr/bin/env python3
"""Export a single-GPU VERL FSDP LoRA checkpoint as a PEFT adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="VERL global_step_<N> directory")
    parser.add_argument("--base-model", required=True, help="Hugging Face ID or local base model path")
    parser.add_argument("--output", type=Path, default=None, help="adapter directory (default: <checkpoint>/huggingface/lora_adapter)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    fsdp_config_path = checkpoint / "fsdp_config.json"
    if not fsdp_config_path.is_file():
        raise ValueError(f"Not a VERL FSDP checkpoint: {checkpoint}")
    fsdp_config = json.loads(fsdp_config_path.read_text(encoding="utf-8"))
    if fsdp_config.get("world_size") != 1:
        raise ValueError(
            "This LoRA exporter supports single-GPU checkpoints only. "
            "Use VERL's distributed model merger for a multi-GPU checkpoint."
        )

    import torch
    from peft import LoraConfig, TaskType
    from safetensors.torch import save_file

    state_path = checkpoint / "model_world_size_1_rank_0.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    lora_state = {
        name.replace(".default.weight", ".weight"): tensor.contiguous()
        for name, tensor in state.items()
        if "lora_" in name
    }
    if not lora_state:
        raise ValueError(f"No LoRA tensors found in {state_path}")

    metadata_path = checkpoint / "lora_train_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    rank = metadata.get("r")
    alpha = metadata.get("lora_alpha")
    if not isinstance(rank, int) or rank < 1 or not isinstance(alpha, (int, float)):
        raise ValueError(f"Invalid or missing LoRA metadata in {metadata_path}")

    target_modules = sorted({name.rsplit(".lora_", 1)[0].rsplit(".", 1)[-1] for name in lora_state})
    adapter_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        base_model_name_or_path=args.base_model,
        inference_mode=True,
    )
    output = (args.output or checkpoint / "huggingface" / "lora_adapter").resolve()
    output.mkdir(parents=True, exist_ok=True)
    adapter_config.save_pretrained(output)
    save_file(lora_state, output / "adapter_model.safetensors")
    print(f"Exported {len(lora_state)} LoRA tensors to {output}")


if __name__ == "__main__":
    main()
