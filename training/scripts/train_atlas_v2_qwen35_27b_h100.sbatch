#!/usr/bin/env bash
# Submit from the repository root on an H100 cluster:
#   sbatch training/scripts/train_atlas_v2_qwen35_27b_h100.sbatch
#
# Two-H100 Qwen3.5-27B LoRA SFT on the V2 ROOT-command training split.
# Adjust account, QoS, and constraint directives for the local H100 scheduler.
#SBATCH --account=atlas
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --constraint=h100
#SBATCH --time=08:00:00
#SBATCH --output=/global/cfs/cdirs/atlas/joshua/ncsa_psc_hackathon_2026/artifacts/slurm/atlas-v2-qwen35-27b-%j.out
#SBATCH --error=/global/cfs/cdirs/atlas/joshua/ncsa_psc_hackathon_2026/artifacts/slurm/atlas-v2-qwen35-27b-%j.err

set -euo pipefail

repo=/global/cfs/cdirs/atlas/joshua/ncsa_psc_hackathon_2026
image="${VERL_IMAGE:-docker.io/verlai/verl@sha256:26b2b1de89333cfbddbf639be5f0ab9d2dbaed19ee5b4892765f59617671e915}"
cache_root="${PSCRATCH:-$repo/artifacts/hf_cache}"
hf_cache="$cache_root/qwen35-hf-cache"
uv_cache="$cache_root/qwen35-atlas-v2-27b-uv-cache"
container_tmp="$cache_root/qwen35-atlas-v2-27b-container-tmp"
save_dir=/workspace/artifacts/checkpoints/atlas-v2-qwen35-27b-r32-e10

mkdir -p "$repo/artifacts/slurm" "$hf_cache" "$uv_cache" "$container_tmp"

podman-hpc run --rm --gpu --ipc=host --entrypoint= \
  -v "$repo":/workspace \
  -v "$hf_cache":/hf_cache \
  -v "$uv_cache":/uv_cache \
  -v "$container_tmp":/tmp \
  -e HF_HOME=/hf_cache \
  -e HF_HUB_CACHE=/hf_cache/hub \
  -e HF_ASSETS_CACHE=/hf_cache/assets \
  -e HF_XET_CACHE=/hf_cache/xet \
  -e HF_HUB_DISABLE_XET=1 \
  -e UV_CACHE_DIR=/uv_cache \
  -e TMPDIR=/tmp \
  -w /workspace \
  "$image" /bin/bash -lc '
    source /workspace/training/scripts/setup.sh
    uv run --frozen --extra fsdp --extra sglang python /workspace/training/prepare_atlas_v2_sft.py \
      --source /workspace/data/datasets/atlas-open-data-sft-dataset/v2/candidates.jsonl \
      --split-manifest /workspace/data/datasets/atlas-open-data-sft-dataset/v2/split_manifest.json \
      --output-dir /workspace/artifacts/datasets/atlas-open-data-sft-v2 \
      --overwrite
    QWEN35_MODEL_SIZE=27b \
    NPROC_PER_NODE=2 \
    TRAIN_FILE=/workspace/artifacts/datasets/atlas-open-data-sft-v2/train.parquet \
    VAL_FILE=/workspace/artifacts/datasets/atlas-open-data-sft-v2/validation.parquet \
    SAVE_DIR='"$save_dir"' \
    TRAIN_BATCH_SIZE=4 \
    MICRO_BATCH_SIZE_PER_GPU=1 \
    MAX_LENGTH=2048 \
    MAX_TOKEN_LEN_PER_GPU=8192 \
    LR=1e-5 \
    TOTAL_EPOCHS=10 \
    LORA_RANK=32 \
    LORA_ALPHA=32 \
    RESUME_MODE=disable \
    PROJECT_NAME=atlas-open-data-sft-v2 \
    EXPERIMENT_NAME=qwen35-27b-r32-e10 \
    bash /workspace/training/scripts/run_verl_sft.sh
  '
