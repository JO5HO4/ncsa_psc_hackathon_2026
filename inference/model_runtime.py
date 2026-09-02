"""Small, local Hugging Face runtime used by the hackathon inference scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_model(model: str | Path) -> str:
    """Resolve a local verl/HF export, or retain a Hugging Face model ID."""
    source = str(model)
    candidate = Path(source).expanduser()
    if not candidate.is_dir():
        if source.startswith((".", "/", "~")):
            raise ValueError(f"Local model directory does not exist: {candidate}")
        return source

    candidate = candidate.resolve()
    export = candidate / "huggingface"
    if export.is_dir():
        candidate = export

    if not (candidate / "config.json").is_file():
        raise ValueError(
            f"{candidate} is not a Transformers model directory. Pass a directory "
            "containing config.json, or a verl checkpoint containing huggingface/."
        )
    return str(candidate)


def select_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda was requested, but CUDA is not available.")
    return requested


def load_model(
    model: str | Path,
    device: str = "auto",
    trust_remote_code: bool = False,
) -> tuple[Any, Any, str, str]:
    """Load a local verl/HF export or a model directly from the Hugging Face Hub."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = resolve_model(model)
    selected_device = select_device(device)
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if selected_device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    if selected_device == "cpu":
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(resolved, torch_dtype=dtype, trust_remote_code=trust_remote_code)
    adapter_path = Path(resolved) / "lora_adapter"
    if (adapter_path / "adapter_config.json").is_file():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.to(selected_device)
    model.eval()
    return model, tokenizer, resolved, selected_device


def generation_kwargs(tokenizer: Any, max_new_tokens: int, temperature: float, top_p: float | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
    return kwargs


def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float | None = None,
) -> str:
    """Generate a completion for a literal text prompt."""
    import torch

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            input_ids,
            **generation_kwargs(tokenizer, max_new_tokens, temperature, top_p),
        )
    completion_ids = generated[0, input_ids.shape[-1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def generate_chat(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float | None = None,
) -> str:
    """Generate one assistant turn with the checkpoint's native chat template."""
    import torch

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            input_ids,
            **generation_kwargs(tokenizer, max_new_tokens, temperature, top_p),
        )
    completion_ids = generated[0, input_ids.shape[-1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)
