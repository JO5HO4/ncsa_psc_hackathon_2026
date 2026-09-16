#!/usr/bin/env bash
# Evaluate the completed mixed ATLAS V2 + Hyy Qwen3.5-27B LoRA smoke run.
# This writes an immediately scored query benchmark and V2 held-out command
# completions ready for ROOT execution scoring on a ROOT-capable node.
#
# Submit from the repository root:
#   sbatch inference/run_atlas_v2_hyy_27b_gh200.sbatch
#SBATCH --account=bccu-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/slurm/atlas-v2-hyy-27b-inference-%j.out
#SBATCH --error=artifacts/slurm/atlas-v2-hyy-27b-inference-%j.err

set -euo pipefail

# sbatch runs a spool-file copy, so BASH_SOURCE points under /var/spool. Keep
# the submit directory as the repository root when Slurm provides it.
repo="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cache_root="${DELTA_GH_CACHE_ROOT:-$repo/artifacts/delta-gh-cache}"
run_name="${MIXED_SFT_RUN_NAME:-atlas-v2-hyy-27b-interactive-smoke}"
checkpoint_root="${MIXED_SFT_CHECKPOINT_ROOT:-$repo/artifacts/checkpoints/$run_name}"
step="$(<"$checkpoint_root/latest_checkpointed_iteration.txt")"
checkpoint_host="$checkpoint_root/global_step_$step"
checkpoint="/workspace/artifacts/checkpoints/$run_name/global_step_$step"
output_dir_host="${MIXED_SFT_INFERENCE_DIR:-$repo/artifacts/inference/$run_name}"
output_dir="/workspace/artifacts/inference/$run_name"
image_sif="$cache_root/images/verl-uv-cu130-arm64.sif"
image_uri="docker://verlai/verl@sha256:094943c1909590564feda8122bd090ba0a93aa42345245111a05225c1a7972f1"
hf_cache="$cache_root/hf-cache"
uv_cache="$cache_root/uv-cache"
container_tmp="$cache_root/tmp"

if [[ ! "$step" =~ ^[0-9]+$ ]] || [[ ! -f "$checkpoint_host/fsdp_config.json" ]]; then
  echo "Missing completed checkpoint under $checkpoint_root" >&2
  exit 2
fi

mkdir -p "$repo/artifacts/slurm" "$output_dir_host" "$hf_cache" "$uv_cache" "$container_tmp" "$(dirname "$image_sif")"
if [[ ! -f "$image_sif" ]]; then
  apptainer pull "$image_sif" "$image_uri"
fi

apptainer exec --nv --cleanenv \
  --bind "$repo:/workspace,$hf_cache:/hf_cache,$uv_cache:/uv_cache,$container_tmp:/tmp" \
  --env HF_HOME=/hf_cache,HF_HUB_CACHE=/hf_cache/hub,HF_ASSETS_CACHE=/hf_cache/assets,HF_XET_CACHE=/hf_cache/xet,HF_HUB_DISABLE_XET=1,UV_CACHE_DIR=/uv_cache,TMPDIR=/tmp,VERL_ENABLE_SGLANG=false,QWEN35_GPU_MEMORY_GIB=85 \
  --pwd /workspace \
  "$image_sif" /bin/bash -lc '
    set -euo pipefail
    source /workspace/training/scripts/setup.sh
    uv run --project /workspace/verl --no-sync --extra fsdp python /workspace/inference/export_verl_lora_adapter.py \
      --checkpoint '"$checkpoint"' --base-model Qwen/Qwen3.5-27B
    uv run --project /workspace/verl --no-sync --extra fsdp python /workspace/inference/validate_model_export.py \
      '"$checkpoint"'/huggingface
    uv run --project /workspace/verl --no-sync --extra fsdp python /workspace/inference/export_atlas_v2_prompts.py \
      --catalog /workspace/data/datasets/atlas-open-data-sft-dataset/v2/candidates.jsonl \
      --split-manifest /workspace/data/datasets/atlas-open-data-sft-dataset/v2/split_manifest.json \
      --split test --output /workspace/artifacts/datasets/atlas-open-data-sft-v2/test-prompts.jsonl
    uv run --project /workspace/verl --no-sync --extra fsdp python /workspace/inference/run_prompts.py \
      --model '"$checkpoint"'/huggingface/lora_adapter --base-model Qwen/Qwen3.5-27B \
      --prompts /workspace/artifacts/datasets/atlas-open-data-sft-v2/test-prompts.jsonl \
      --prompt-field prompt --id-field id --format chat --no-enable-thinking \
      --system-prompt "You are a ROOT and ATLAS Open Data assistant. Return exactly one executable shell command, with no prose or Markdown." \
      --device cuda --temperature 0 --max-new-tokens 1024 \
      --output '"$output_dir"'/v2-test-commands.jsonl
    uv run --project /workspace/verl --no-sync --extra fsdp python /workspace/inference/materialize_v2_model_catalog.py \
      --catalog /workspace/data/datasets/atlas-open-data-sft-dataset/v2/candidates.jsonl \
      --split-manifest /workspace/data/datasets/atlas-open-data-sft-dataset/v2/split_manifest.json \
      --split test --completions '"$output_dir"'/v2-test-commands.jsonl \
      --output '"$output_dir"'/v2-test-catalog.jsonl
    uv run --project /workspace/verl --no-sync --extra fsdp python /workspace/inference/run_prompts.py \
      --model '"$checkpoint"'/huggingface/lora_adapter --base-model Qwen/Qwen3.5-27B \
      --prompts /workspace/data/datasets/atlas-open-data-sft-dataset/data/tasks/query_tasks.jsonl \
      --prompt-field question --id-field id --format chat --no-enable-thinking \
      --system-prompt "Reply with only the requested final answer. Do not explain your reasoning." \
      --device cuda --temperature 0 --max-new-tokens 32 \
      --output '"$output_dir"'/query-completions.jsonl
    python3 /workspace/inference/evaluate_atlas_benchmark.py \
      --completions '"$output_dir"'/query-completions.jsonl \
      --dataset-root /workspace/data/datasets/atlas-open-data-sft-dataset \
      --output '"$output_dir"'/query-score.json
  '

printf 'Evaluation outputs: %s\n' "$output_dir_host"
