#!/usr/bin/env bash
# Run Qwen3.5-9B base and V2-SFT inference on identical held-out prompts.
# Run after `source training/scripts/setup.sh` inside the VERL GPU container.
set -euo pipefail

repo=/workspace
base_model="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
catalog="$repo/data/datasets/atlas-open-data-sft-dataset/v2/candidates.jsonl"
checkpoint_root="${CHECKPOINT_ROOT:-$repo/artifacts/checkpoints/atlas-v2-qwen35-9b-r32-e10}"
prompt_file="$repo/artifacts/datasets/atlas-open-data-sft-v2/test-prompts.jsonl"
output_dir="${OUTPUT_DIR:-$repo/artifacts/inference/atlas-v2-qwen35-9b-r32-e10}"
system_prompt="You are a ROOT and ATLAS Open Data assistant. Return exactly one executable shell command, with no prose or Markdown."

if [[ ! -f "$checkpoint_root/latest_checkpointed_iteration.txt" ]]; then
  echo "Missing checkpoint tracker: $checkpoint_root/latest_checkpointed_iteration.txt" >&2
  exit 1
fi
# VERL's tracker is often written without a final newline. `read` then returns
# EOF/nonzero despite assigning the step, which would silently terminate this
# `set -e` script. Command substitution preserves the complete tracker value.
step="$(<"$checkpoint_root/latest_checkpointed_iteration.txt")"
if ! [[ "$step" =~ ^[0-9]+$ ]]; then
  echo "Invalid checkpoint step: $step" >&2
  exit 1
fi
checkpoint="$checkpoint_root/global_step_$step"
adapter="$checkpoint/huggingface/lora_adapter"
if [[ ! -f "$checkpoint/fsdp_config.json" ]]; then
  echo "Missing checkpoint: $checkpoint" >&2
  exit 1
fi

mkdir -p "$output_dir"
if [[ ! -f "$prompt_file" ]]; then
  if [[ ! -f "$catalog" ]]; then
    echo "Missing V2 catalog needed to create prompts: $catalog" >&2
    exit 1
  fi
  uv run --project "$repo/verl" --no-sync python "$repo/inference/export_atlas_v2_prompts.py" \
    --catalog "$catalog" --split test --output "$prompt_file"
fi
if [[ ! -f "$adapter/adapter_model.safetensors" ]]; then
  uv run --project "$repo/verl" --no-sync python "$repo/inference/export_verl_lora_adapter.py" \
    --checkpoint "$checkpoint" --base-model "$base_model" --output "$adapter"
fi

uv run --project "$repo/verl" --no-sync python "$repo/inference/run_prompts.py" \
  --model "$base_model" --prompts "$prompt_file" --prompt-field prompt --id-field id \
  --format chat --no-enable-thinking --system-prompt "$system_prompt" \
  --device cuda --temperature 0 --max-new-tokens 1024 \
  --output "$output_dir/base.jsonl"

uv run --project "$repo/verl" --no-sync python "$repo/inference/run_prompts.py" \
  --model "$adapter" --base-model "$base_model" \
  --prompts "$prompt_file" --prompt-field prompt --id-field id \
  --format chat --no-enable-thinking --system-prompt "$system_prompt" \
  --device cuda --temperature 0 --max-new-tokens 1024 \
  --output "$output_dir/sft.jsonl"

printf 'Checkpoint: %s\nBase: %s/base.jsonl\nSFT: %s/sft.jsonl\n' "$checkpoint" "$output_dir" "$output_dir"
