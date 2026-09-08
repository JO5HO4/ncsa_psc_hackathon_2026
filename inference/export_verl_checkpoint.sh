#!/usr/bin/env bash
# Convert a raw FSDP checkpoint from the local verl launcher to Hugging Face.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_DIR="${VERL_DIR:-$REPO_ROOT/verl}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/verl-venv}"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 CHECKPOINT_DIR [OUTPUT_DIR]" >&2
  exit 2
fi

caller_dir="$PWD"
checkpoint_dir="$1"
if [[ "$checkpoint_dir" != /* ]]; then
  checkpoint_dir="$caller_dir/$checkpoint_dir"
fi
checkpoint_dir="$(cd "$checkpoint_dir" && pwd)"
output_dir="${2:-$checkpoint_dir/huggingface}"
if [[ "$output_dir" != /* ]]; then
  output_dir="$caller_dir/$output_dir"
fi

if [[ ! -f "$checkpoint_dir/fsdp_config.json" ]]; then
  echo "Not a raw verl FSDP checkpoint: $checkpoint_dir" >&2
  exit 2
fi

cd "$VERL_DIR"
uv run --frozen --extra fsdp --extra sglang python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$checkpoint_dir" \
  --target_dir "$output_dir"
