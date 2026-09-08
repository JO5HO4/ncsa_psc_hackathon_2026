"""Small, local Hugging Face runtime used by the hackathon inference scripts."""

from __future__ import annotations

import os
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
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer

    resolved = resolve_model(model)
    selected_device = select_device(device)
    config = AutoConfig.from_pretrained(resolved, trust_remote_code=trust_remote_code)
    is_qwen35 = config.model_type in {"qwen3_5", "qwen3_5_moe"}
    if is_qwen35:
        tokenizer = AutoProcessor.from_pretrained(resolved, trust_remote_code=trust_remote_code)
    else:
        tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=trust_remote_code)

    underlying_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if underlying_tokenizer.pad_token_id is None:
        underlying_tokenizer.pad_token = underlying_tokenizer.eos_token

    dtype = torch.bfloat16 if selected_device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    if selected_device == "cpu":
        dtype = torch.float32
    model_class = AutoModelForMultimodalLM if is_qwen35 else AutoModelForCausalLM
    load_kwargs: dict[str, Any] = {"dtype": dtype, "trust_remote_code": trust_remote_code}
    if selected_device == "cuda":
        # Stream checkpoint shards directly to the available accelerator rather
        # than materializing a full CPU copy before model.to(cuda). This avoids
        # an otherwise large host-memory spike for Qwen3.5-9B.
        load_kwargs.update(low_cpu_mem_usage=True, device_map="auto")
        if is_qwen35:
            # Qwen3.5-9B fits comfortably in BF16 on Perlmutter's 40 GB A100.
            # Keep roughly 10 GB for generation and allocator headroom while
            # streaming the checkpoint directly to the accelerator.
            gpu_limit = os.environ.get("QWEN35_GPU_MEMORY_GIB", "30")
            load_kwargs["max_memory"] = {0: f"{gpu_limit}GiB", "cpu": "40GiB"}
    model = model_class.from_pretrained(resolved, **load_kwargs)
    adapter_path = Path(resolved) / "lora_adapter"
    if (adapter_path / "adapter_config.json").is_file():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    if selected_device != "cuda":
        model.to(selected_device)
    model.eval()
    return model, tokenizer, resolved, selected_device


def _uses_multimodal_processor(runtime: Any) -> bool:
    """Return whether the runtime is a Qwen3.5-style AutoProcessor."""
    return hasattr(runtime, "image_processor") and hasattr(runtime, "tokenizer")


def _move_inputs_to_model(inputs: Any, device: str) -> dict[str, Any]:
    """Move a tokenizer/processor batch to the generation device."""
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in dict(inputs).items()}


def _decode(runtime: Any, token_ids: Any) -> str:
    """Decode generated IDs using either an AutoTokenizer or AutoProcessor."""
    return runtime.decode(token_ids, skip_special_tokens=True)


def generation_kwargs(tokenizer: Any, max_new_tokens: int, temperature: float, top_p: float | None) -> dict[str, Any]:
    underlying_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": underlying_tokenizer.pad_token_id,
        "eos_token_id": underlying_tokenizer.eos_token_id,
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

    if _uses_multimodal_processor(tokenizer):
        inputs = _move_inputs_to_model(tokenizer(text=prompt, return_tensors="pt"), model.device)
    else:
        inputs = _move_inputs_to_model(tokenizer(prompt, return_tensors="pt"), model.device)
    input_ids = inputs["input_ids"]
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            **generation_kwargs(tokenizer, max_new_tokens, temperature, top_p),
        )
    completion_ids = generated[0, input_ids.shape[-1] :]
    return _decode(tokenizer, completion_ids)


def generate_chat(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float | None = None,
    enable_thinking: bool = False,
) -> str:
    """Generate one assistant turn with the checkpoint's native chat template."""
    import torch

    if _uses_multimodal_processor(tokenizer):
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            # Qwen3.5 otherwise spends completion tokens on a visible
            # reasoning trace. ROOT benchmark answers are concise finals.
            enable_thinking=enable_thinking,
        )
        inputs = _move_inputs_to_model(inputs, model.device)
    else:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        inputs = {"input_ids": input_ids}
    input_ids = inputs["input_ids"]
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            **generation_kwargs(tokenizer, max_new_tokens, temperature, top_p),
        )
    completion_ids = generated[0, input_ids.shape[-1] :]
    return _decode(tokenizer, completion_ids)
