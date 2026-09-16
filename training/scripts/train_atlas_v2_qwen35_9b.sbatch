#!/usr/bin/env bash
# Submit from the repository root:
#   sbatch training/scripts/train_atlas_v2_qwen35_9b.sbatch
#
# Four-GPU Qwen3.5-9B LoRA SFT on the V2 ROOT-command training split.
# The final launcher step exports the adapter needed by inference/run_prompts.py.
#SBATCH --account=atlas
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --constraint=gpu
#SBATCH --time=04:00:00
#SBATCH --output=/global/cfs/cdirs/atlas/joshua/ncsa_psc_hackathon_2026/artifacts/slurm/atlas-v2-qwen35-9b-%j.out
#SBATCH --error=/global/cfs/cdirs/atlas/joshua/ncsa_psc_hackathon_2026/artifacts/slurm/atlas-v2-qwen35-9b-%j.err

set -euo pipefail

repo=/global/cfs/cdirs/atlas/joshua/ncsa_psc_hackathon_2026
image="${VERL_IMAGE:-docker.io/verlai/verl@sha256:26b2b1de89333cfbddbf639be5f0ab9d2dbaed19ee5b4892765f59617671e915}"
cache_root="${PSCRATCH:-$repo/artifacts/hf_cache}"
hf_cache="$cache_root/qwen35-hf-cache"
container_tmp="$cache_root/qwen35-container-tmp"
save_dir=/workspace/artifacts/checkpoints/atlas-v2-qwen35-9b-r32-e10

mkdir -p "$repo/artifacts/slurm" "$hf_cache" "$container_tmp"

podman-hpc run --rm --gpu --ipc=host --entrypoint= \
  -v "$repo":/workspace \
  -v "$hf_cache":/hf_cache \
  -v "$container_tmp":/tmp \
  -e HF_HOME=/hf_cache \
  -e HF_HUB_CACHE=/hf_cache/hub \
  -e HF_ASSETS_CACHE=/hf_cache/assets \
  -e HF_XET_CACHE=/hf_cache/xet \
  -e HF_HUB_DISABLE_XET=1 \
  -e UV_CACHE_DIR=/hf_cache/uv \
  -e TMPDIR=/tmp \
  -w /workspace \
  "$image" /bin/bash -lc '
    source /workspace/training/scripts/setup.sh
    QWEN35_MODEL_SIZE=9b \
    NPROC_PER_NODE=4 \
    TRAIN_FILE=/workspace/artifacts/datasets/atlas-open-data-sft-v2/train.parquet \
    VAL_FILE=/workspace/artifacts/datasets/atlas-open-data-sft-v2/validation.parquet \
    SAVE_DIR='"$save_dir"' \
    TRAIN_BATCH_SIZE=16 \
    MICRO_BATCH_SIZE_PER_GPU=1 \
    MAX_LENGTH=2048 \
    MAX_TOKEN_LEN_PER_GPU=8192 \
    LR=1e-5 \
    TOTAL_EPOCHS=10 \
    LORA_RANK=32 \
    LORA_ALPHA=32 \
    RESUME_MODE=disable \
    PROJECT_NAME=atlas-open-data-sft-v2 \
    EXPERIMENT_NAME=qwen35-9b-r32-e10 \
    bash /workspace/training/scripts/run_verl_sft.sh
  '
