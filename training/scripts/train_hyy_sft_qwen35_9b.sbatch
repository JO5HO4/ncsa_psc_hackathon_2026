#!/usr/bin/env bash
# Submit from the repository root:
#   sbatch training/scripts/train_hyy_sft_qwen35_9b.sbatch
#
# Four-GPU Qwen3.5-9B LoRA SFT on the pinned Hyy native-tool trajectories.
# The Hyy corpus is downloaded with checksums before the container starts.
# Its tool-use conversations require a larger context window than V2.
#SBATCH --account=atlas
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --constraint=gpu
#SBATCH --time=04:00:00
#SBATCH --output=/global/cfs/cdirs/atlas/joshua/ncsa_psc_hackathon_2026/artifacts/slurm/hyy-sft-qwen35-9b-%j.out
#SBATCH --error=/global/cfs/cdirs/atlas/joshua/ncsa_psc_hackathon_2026/artifacts/slurm/hyy-sft-qwen35-9b-%j.err

set -euo pipefail

repo=/global/cfs/cdirs/atlas/joshua/ncsa_psc_hackathon_2026
image="${VERL_IMAGE:-docker.io/verlai/verl@sha256:26b2b1de89333cfbddbf639be5f0ab9d2dbaed19ee5b4892765f59617671e915}"
cache_root="${PSCRATCH:-$repo/artifacts/hf_cache}"
hf_cache="$cache_root/qwen35-hf-cache"
uv_cache="$cache_root/qwen35-hyy-uv-cache"
container_tmp="$cache_root/qwen35-hyy-container-tmp"
dataset_dir="$repo/artifacts/datasets/hyy-sft"
save_dir=/workspace/artifacts/checkpoints/hyy-sft-qwen35-9b-r32-e10

mkdir -p "$repo/artifacts/slurm" "$hf_cache" "$uv_cache" "$container_tmp"
python3 "$repo/training/fetch_hyy_sft.py" --output-dir "$dataset_dir"

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
    QWEN35_MODEL_SIZE=9b \
    NPROC_PER_NODE=4 \
    TRAIN_FILE=/workspace/artifacts/datasets/hyy-sft/train.parquet \
    VAL_FILE=/workspace/artifacts/datasets/hyy-sft/validation.parquet \
    SAVE_DIR='"$save_dir"' \
    TRAIN_BATCH_SIZE=4 \
    MICRO_BATCH_SIZE_PER_GPU=1 \
    MAX_LENGTH=32768 \
    MAX_TOKEN_LEN_PER_GPU=32768 \
    LR=1e-5 \
    TOTAL_EPOCHS=10 \
    LORA_RANK=32 \
    LORA_ALPHA=32 \
    ENABLE_THINKING_DEFAULT=false \
    RESUME_MODE=disable \
    PROJECT_NAME=hyy-sft \
    EXPERIMENT_NAME=qwen35-9b-r32-e10 \
    bash /workspace/training/scripts/run_verl_sft.sh
  '
