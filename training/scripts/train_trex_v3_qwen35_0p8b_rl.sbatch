#!/usr/bin/env bash
# Low-cost GRPO launcher for development on one shared GPU.
# Submit from the repository root:
#   sbatch training/scripts/train_trex_v3_qwen35_0p8b_rl.sbatch
# Override a short smoke run, for example:
#   sbatch --export=ALL,TREX_RL_TOTAL_STEPS=1 training/scripts/train_trex_v3_qwen35_0p8b_rl.sbatch
#
# Native TREx rewards are deliberately evaluated one at a time by the host
# reward server; increasing GPU count does not accelerate that part.
#SBATCH --account=atlas
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
# NERSC's gpu_shared_ss11 queue charges/reserves 32 CPU cores per GPU.
#SBATCH --cpus-per-task=32
#SBATCH --constraint=gpu
#SBATCH --time=04:00:00
#SBATCH --output=/global/cfs/cdirs/atlas/joshua/ncsa_psc_hackathon_2026/artifacts/slurm/trex-v3-rl-0p8b-%j.out
#SBATCH --error=/global/cfs/cdirs/atlas/joshua/ncsa_psc_hackathon_2026/artifacts/slurm/trex-v3-rl-0p8b-%j.err

set -euo pipefail

repo=/global/cfs/cdirs/atlas/joshua/ncsa_psc_hackathon_2026
image="${VERL_IMAGE:-docker.io/verlai/verl@sha256:26b2b1de89333cfbddbf639be5f0ab9d2dbaed19ee5b4892765f59617671e915}"
cache_root="${PSCRATCH:-$repo/artifacts/hf_cache}"
hf_cache="$cache_root/qwen35-hf-cache"
container_tmp="$cache_root/qwen35-container-tmp"
dataset_dir=/workspace/artifacts/datasets/trex-v3-rl
reward_socket=artifacts/trex_fitter/rl-reward/trex-v3-0p8b.sock
train_steps="${TREX_RL_TOTAL_STEPS:-8}"
rollout_count="${TREX_RL_ROLLOUT_N:-2}"
train_batch_size="${TREX_RL_TRAIN_BATCH_SIZE:-2}"
save_dir="${TREX_RL_SAVE_DIR:-/workspace/artifacts/checkpoints/trex-v3-qwen35-0p8b-grpo}"

mkdir -p "$repo/artifacts/slurm" "$hf_cache" "$container_tmp"
python3 "$repo/training/prepare_trex_v3_rl.py" --output-dir "$repo/artifacts/datasets/trex-v3-rl"
TREX_RL_REWARD_LOG="$repo/artifacts/trex_fitter/rl-reward/diagnostics-0p8b-$SLURM_JOB_ID.jsonl" \
  python3 "$repo/training/rewards/trex_v3_reward_server.py" --socket "$reward_socket" --timeout 600 \
  >"$repo/artifacts/slurm/trex-v3-reward-0p8b-$SLURM_JOB_ID.log" 2>&1 &
reward_server_pid=$!
trap 'kill "$reward_server_pid" 2>/dev/null || true; wait "$reward_server_pid" 2>/dev/null || true' EXIT

podman-hpc run --rm --gpu --ipc=host --entrypoint= \
  -v "$repo":/workspace -v "$hf_cache":/hf_cache -v "$container_tmp":/tmp \
  -e HF_HOME=/hf_cache -e HF_HUB_CACHE=/hf_cache/hub -e HF_ASSETS_CACHE=/hf_cache/assets \
  -e HF_XET_CACHE=/hf_cache/xet -e HF_HUB_DISABLE_XET=1 -e UV_CACHE_DIR=/hf_cache/uv -e TMPDIR=/tmp \
  -e TREX_RL_REWARD_SOCKET="$reward_socket" -e TREX_RL_REWARD_SECONDS=2100 \
  -w /workspace "$image" /bin/bash -lc '
    source /workspace/training/scripts/setup.sh
    QWEN35_MODEL_SIZE=0.8b NPROC_PER_NODE=1 \
    TRAIN_FILE='"$dataset_dir"'/train.parquet VAL_FILE='"$dataset_dir"'/validation.parquet \
    REWARD_FUNCTION=/workspace/training/rewards/trex_v3_histfit_reward.py \
    SAVE_DIR='"$save_dir"' \
    TRAIN_BATCH_SIZE='"$train_batch_size"' PPO_MINI_BATCH_SIZE=2 ROLLOUT_N='"$rollout_count"' \
    MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=2048 \
    LORA_MERGE=True ACTOR_PARAM_OFFLOAD=False ROLLOUT_TP=1 ROLLOUT_GPU_MEM_UTIL=0.30 \
    ROLLOUT_MAX_NUM_BATCHED_TOKENS=2048 ROLLOUT_MAX_NUM_SEQS=2 \
    ROLLOUT_FREE_CACHE_ENGINE=True AGENT_LOOP_WORKERS=1 \
    ACTOR_USE_DYNAMIC_BSZ=False ROLLOUT_LOG_PROB_USE_DYNAMIC_BSZ=False REF_LOG_PROB_USE_DYNAMIC_BSZ=False \
    ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=1 LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1 REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1 \
    USE_REMOVE_PADDING=False \
    TOTAL_TRAINING_STEPS='"$train_steps"' TOTAL_EPOCHS=1 SAVE_FREQ=1 TEST_FREQ=1 \
    REWARD_NUM_WORKERS=1 PROJECT_NAME=trex-v3-rl EXPERIMENT_NAME=qwen35-0p8b-grpo \
    TREX_RL_TIMEOUT_SECONDS=600 \
    bash /workspace/training/scripts/run_verl_rl.sh
  '
