#!/usr/bin/env bash
# Convert a raw FSDP checkpoint from the local verl launcher to Hugging Face.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 CHECKPOINT_DIR [OUTPUT_DIR]" >&2
  exit 2
fi

checkpoint_dir="$1"
output_dir="${2:-$checkpoint_dir/huggingface}"

if [[ ! -f "$checkpoint_dir/fsdp_config.json" ]]; then
  echo "Not a raw verl FSDP checkpoint: $checkpoint_dir" >&2
  exit 2
fi

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$checkpoint_dir" \
  --target_dir "$output_dir"
