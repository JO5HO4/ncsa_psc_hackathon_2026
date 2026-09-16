#!/usr/bin/env bash
# Run INSIDE the existing verl container. Does not replace the training framework.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
MODE="${1:-smoke}"
case "$MODE" in smoke|full) ;; *) echo 'Usage: run.sh smoke|full' >&2; exit 2;; esac
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/root_io/splits/root-sft-v1}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to a new absolute output directory}"
[[ "$RUN_DIR" = /* && "$DATA_DIR" = /* ]] || { echo 'Use absolute RUN_DIR and DATA_DIR' >&2; exit 2; }
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing existing RUN_DIR: $RUN_DIR" >&2; exit 2; }
mkdir -p "$RUN_DIR"
source "$REPO_ROOT/training/root_sft/logging.sh"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export MAX_LENGTH="${MAX_LENGTH:-2048}"
export NPROC_PER_NODE=1
export MICRO_BATCH_SIZE_PER_GPU=1
export MAX_TOKEN_LEN_PER_GPU=4096
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export LR="${LR:-1e-5}"
export LORA_RANK=16 LORA_ALPHA=16
export LORA_TARGETS='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]'
export RESUME_MODE=disable EXPORT_FOR_INFERENCE=false
export ENGINE_USE_TORCH_COMPILE=false
export PROJECT_NAME=root-knowledge-sft EXPERIMENT_NAME="root-$MODE"
export TRAIN_FILE="$DATA_DIR/train.parquet" VAL_FILE="$DATA_DIR/validation.parquet"
export SAVE_DIR="$RUN_DIR/checkpoints"
root_stage preflight
python training/root_sft/preflight.py --data "$DATA_DIR" --output "$RUN_DIR" \
  --model "${MODEL_PATH:-Qwen/Qwen2.5-Coder-1.5B-Instruct}" --revision "${MODEL_REVISION:-main}" \
  --max-length "$MAX_LENGTH" --batch-size "$TRAIN_BATCH_SIZE"
read -r MODEL_PATH < "$RUN_DIR/model_path.txt"
export MODEL_PATH
read -r SYSTEM_PROMPT < "$RUN_DIR/system_prompt.txt"
INFERENCE_ARGS=(--format chat --device cuda --temperature 0 --max-new-tokens 512 --system-prompt "$SYSTEM_PROMPT")
TRAIN_ARGS=(data.ignore_input_ids_mismatch=False)
REVIEW_ARGS=()
if [[ "$MODE" == smoke ]]; then
  INFERENCE_ARGS+=(--limit 4)
  TRAIN_ARGS+=(trainer.total_training_steps=2)
  REVIEW_ARGS+=(--limit 4)
fi
# Validation is used during development. Keep the test set sealed until settings freeze.
root_stage baseline_inference
python inference/run_prompts.py --model "$MODEL_PATH" --prompts "$DATA_DIR/validation_prompts.jsonl" \
  --output "$RUN_DIR/before.jsonl" "${INFERENCE_ARGS[@]}"
root_stage training
bash training/scripts/run_verl_sft.sh "${TRAIN_ARGS[@]}" 2>&1 | tee "$RUN_DIR/training.log"
root_stage checkpoint_export
# verl's tracker has no trailing newline. Preserve its value at EOF under set -e.
STEP=""
read -r STEP < "$SAVE_DIR/latest_checkpointed_iteration.txt" || [[ -n "$STEP" ]]
[[ "$STEP" =~ ^[0-9]+$ ]] || { echo 'Invalid checkpoint step' >&2; exit 1; }
EXPORT_DIR="$SAVE_DIR/global_step_$STEP/huggingface"
bash inference/export_verl_checkpoint.sh "$SAVE_DIR/global_step_$STEP" "$EXPORT_DIR"
root_stage weight_update_check
python training/root_sft/check_adapter.py "$EXPORT_DIR" --output "$RUN_DIR/weight_update.json"
root_stage post_training_inference
python inference/run_prompts.py --model "$EXPORT_DIR" --prompts "$DATA_DIR/validation_prompts.jsonl" \
  --output "$RUN_DIR/after.jsonl" "${INFERENCE_ARGS[@]}"
root_stage paired_review
python training/root_sft/evaluate.py prepare --references "$DATA_DIR/validation_references.jsonl" \
  --before "$RUN_DIR/before.jsonl" --after "$RUN_DIR/after.jsonl" --output "$RUN_DIR/review" "${REVIEW_ARGS[@]}"
root_stage complete
