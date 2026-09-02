#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERL_DIR="${VERL_DIR:-$REPO_ROOT/verl}"
cd "$VERL_DIR"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
# These defaults run the included reference config dataset. Override them when
# training on a hackathon-created split.
TRAIN_FILE="${TRAIN_FILE:-$REPO_ROOT/data/reference/hep-config-sft/data/train.parquet}"
VAL_FILE="${VAL_FILE:-$REPO_ROOT/data/reference/hep-config-sft/data/validation.parquet}"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/artifacts/checkpoints/sft}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-8192}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
LR="${LR:-1e-5}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
PROJECT_NAME="${PROJECT_NAME:-trex-config-hackathon}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-config-sft-baseline}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_TARGETS="${LORA_TARGETS:-[\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\",\"gate_proj\",\"up_proj\",\"down_proj\"]}"
# Current verl uses -1 to save or test only at the final step.
SAVE_FREQ="${SAVE_FREQ:--1}"
TEST_FREQ="${TEST_FREQ:--1}"
RESUME_MODE="${RESUME_MODE:-auto}"
MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-null}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
ENGINE_MODEL_DTYPE="${ENGINE_MODEL_DTYPE:-fp32}"
ENGINE_USE_TORCH_COMPILE="${ENGINE_USE_TORCH_COMPILE:-true}"

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" --master_addr="$MASTER_ADDR" \
  -m verl.trainer.sft_trainer \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.messages_key=messages \
  data.tools_key=tools \
  data.custom_cls.path="$REPO_ROOT/training/verl_dataset.py" \
  data.custom_cls.name=TReXConfigSFTDataset \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU" \
  data.max_token_len_per_gpu="$MAX_TOKEN_LEN_PER_GPU" \
  data.max_length="$MAX_LENGTH" \
  data.truncation=error \
  data.ignore_input_ids_mismatch=True \
  data.num_workers=2 \
  optim.lr="$LR" \
  engine=fsdp \
  engine.model_dtype="$ENGINE_MODEL_DTYPE" \
  engine.use_torch_compile="$ENGINE_USE_TORCH_COMPILE" \
  model.path="$MODEL_PATH" \
  model.use_remove_padding=true \
  model.lora_rank="$LORA_RANK" \
  model.lora_alpha="$LORA_ALPHA" \
  model.target_modules="$LORA_TARGETS" \
  trainer.default_local_dir="$SAVE_DIR" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.logger=console \
  trainer.n_gpus_per_node="$NPROC_PER_NODE" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.max_ckpt_to_keep="$MAX_CKPT_TO_KEEP" \
  trainer.resume_mode="$RESUME_MODE" \
  "$@"
