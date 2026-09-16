#!/usr/bin/env bash
# Delta-GH launcher for a mixed ATLAS V2 + Hyy Qwen3.5-27B LoRA SFT run.
# Submit from this repository:
#   sbatch training/scripts/train_atlas_v2_hyy_sft_qwen35_27b_gh200.sbatch
#
# Delta-GH provides four 120 GB GH200 GPUs per node. This job uses two.
# It uses a pinned ARM64 VERL base image through Apptainer; the existing
# Podman-HPC launchers and their AMD64 image are for other clusters.
#SBATCH --account=bccu-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/slurm/atlas-v2-hyy-qwen35-27b-gh200-%j.out
#SBATCH --error=artifacts/slurm/atlas-v2-hyy-qwen35-27b-gh200-%j.err

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cache_root="${DELTA_GH_CACHE_ROOT:-$repo/artifacts/delta-gh-cache}"
hyy_dir="$repo/artifacts/datasets/hyy-sft"
atlas_dir=/workspace/artifacts/datasets/atlas-open-data-sft-v2
mixed_dir=/workspace/artifacts/datasets/atlas-v2-hyy-sft
save_dir=/workspace/artifacts/checkpoints/atlas-v2-hyy-qwen35-27b-r32-e10
image_sif="$cache_root/images/verl-uv-cu130-arm64.sif"
image_uri="docker://verlai/verl@sha256:094943c1909590564feda8122bd090ba0a93aa42345245111a05225c1a7972f1"
hf_cache="$cache_root/hf-cache"
uv_cache="$cache_root/uv-cache"
container_tmp="$cache_root/tmp"
train_batch_size="${MIXED_SFT_TRAIN_BATCH_SIZE:-2}"
max_length="${MIXED_SFT_MAX_LENGTH:-32768}"
max_token_len="${MIXED_SFT_MAX_TOKEN_LEN_PER_GPU:-32768}"

mkdir -p "$repo/artifacts/slurm" "$hyy_dir" "$hf_cache" "$uv_cache" "$container_tmp" "$(dirname "$image_sif")"
if [[ ! -f "$image_sif" ]]; then
  apptainer pull "$image_sif" "$image_uri"
fi
python3 "$repo/training/fetch_hyy_sft.py" --output-dir "$hyy_dir"

apptainer exec --nv --cleanenv \
  --bind "$repo:/workspace,$hf_cache:/hf_cache,$uv_cache:/uv_cache,$container_tmp:/tmp" \
  --env HF_HOME=/hf_cache,HF_HUB_CACHE=/hf_cache/hub,HF_ASSETS_CACHE=/hf_cache/assets,HF_XET_CACHE=/hf_cache/xet,HF_HUB_DISABLE_XET=1,UV_CACHE_DIR=/uv_cache,TMPDIR=/tmp,VERL_ENABLE_SGLANG=false \
  --pwd /workspace \
  "$image_sif" /bin/bash -lc '
    source /workspace/training/scripts/setup.sh
    uv run --frozen --extra fsdp python /workspace/training/prepare_atlas_v2_sft.py \
      --source /workspace/data/datasets/atlas-open-data-sft-dataset/v2/candidates.jsonl \
      --split-manifest /workspace/data/datasets/atlas-open-data-sft-dataset/v2/split_manifest.json \
      --output-dir '"$atlas_dir"' --overwrite
    uv run --frozen --extra fsdp python /workspace/training/compose_hyy_atlas_v2_sft.py \
      --hyy-dir /workspace/artifacts/datasets/hyy-sft \
      --atlas-dir '"$atlas_dir"' \
      --output-dir '"$mixed_dir"' --overwrite
    QWEN35_MODEL_SIZE=27b \
    NPROC_PER_NODE=2 \
    TRAIN_FILE='"$mixed_dir"'/train.parquet \
    VAL_FILE='"$mixed_dir"'/validation.parquet \
    SAVE_DIR='"$save_dir"' \
    TRAIN_BATCH_SIZE='"$train_batch_size"' \
    MICRO_BATCH_SIZE_PER_GPU=1 \
    MAX_LENGTH='"$max_length"' \
    MAX_TOKEN_LEN_PER_GPU='"$max_token_len"' \
    LR=1e-5 \
    TOTAL_EPOCHS=10 \
    LORA_RANK=32 \
    LORA_ALPHA=32 \
    ENABLE_THINKING_DEFAULT=false \
    VERL_ENABLE_SGLANG=false \
    RESUME_MODE=disable \
    PROJECT_NAME=atlas-v2-hyy-sft \
    EXPERIMENT_NAME=qwen35-27b-r32-e10 \
    bash /workspace/training/scripts/run_verl_sft.sh
  '
