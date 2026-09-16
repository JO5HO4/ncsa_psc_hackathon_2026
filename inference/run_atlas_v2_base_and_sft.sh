#!/usr/bin/env bash
# Compare the Qwen3.5 base model and a V2 LoRA SFT checkpoint on identical test prompts.
# Run after `source training/scripts/setup.sh` inside the GPU VERL container.
set -euo pipefail

repo=/workspace
catalog="$repo/data/datasets/atlas-open-data-sft-dataset/v2/candidates.jsonl"
checkpoint="${CHECKPOINT:-$repo/artifacts/checkpoints/atlas-v2-qwen35-0.8b-r32/global_step_785}"
base_model="${BASE_MODEL:-Qwen/Qwen3.5-0.8B}"
prompt_file="$repo/artifacts/datasets/atlas-open-data-sft-v2/test-prompts.jsonl"
output_dir="${OUTPUT_DIR:-$repo/artifacts/inference/atlas-v2-qwen35-0.8b-r32}"
adapter="$checkpoint/huggingface/lora_adapter"
system_prompt="You are a ROOT and ATLAS Open Data assistant. Return exactly one executable shell command, with no prose or Markdown."

if [[ ! -f "$catalog" ]]; then
  echo "Missing V2 catalog: $catalog" >&2
  exit 1
fi
if [[ ! -f "$checkpoint/fsdp_config.json" ]]; then
  echo "Not a VERL checkpoint: $checkpoint" >&2
  exit 1
fi

mkdir -p "$output_dir"
if [[ ! -f "$prompt_file" ]]; then
  uv run --project "$repo/verl" --no-sync python "$repo/inference/export_atlas_v2_prompts.py" \
    --catalog "$catalog" --split test --output "$prompt_file"
fi
if [[ ! -f "$adapter/adapter_model.safetensors" ]]; then
  uv run --project "$repo/verl" --no-sync python "$repo/inference/export_verl_lora_adapter.py" \
    --checkpoint "$checkpoint" --base-model "$base_model" --output "$adapter"
fi

uv run --project "$repo/verl" --no-sync python "$repo/inference/run_prompts.py" \
  --model "$base_model" \
  --prompts "$prompt_file" --prompt-field prompt --id-field id \
  --format chat --no-enable-thinking --system-prompt "$system_prompt" \
  --device cuda --temperature 0 --max-new-tokens 1024 \
  --output "$output_dir/base.jsonl"

uv run --project "$repo/verl" --no-sync python "$repo/inference/run_prompts.py" \
  --model "$adapter" --base-model "$base_model" \
  --prompts "$prompt_file" --prompt-field prompt --id-field id \
  --format chat --no-enable-thinking --system-prompt "$system_prompt" \
  --device cuda --temperature 0 --max-new-tokens 1024 \
  --output "$output_dir/sft-global_step_785.jsonl"

echo "Base:    $output_dir/base.jsonl"
echo "Trained: $output_dir/sft-global_step_785.jsonl"
