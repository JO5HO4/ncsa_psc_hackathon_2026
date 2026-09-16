#!/usr/bin/env bash
# Train and execution-score a deliberately tiny, fixed SFT subset.
#
# Run this inside the VERL training container.  A high score is expected: this
# is an overfit/pipeline check, not a generalization benchmark.
set -euo pipefail
unset VIRTUAL_ENV

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERL_DIR="${VERL_DIR:-$REPO_ROOT/verl}"
DATASET_ROOT="${ATLAS_DATASET_ROOT:-$REPO_ROOT/data/datasets/atlas-open-data-sft-dataset}"
ROWS="${ROWS:-32}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-50}"
LR="${LR:-1e-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-8192}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/artifacts/diagnostics/qwen35-0.8b-sft-overfit-${ROWS}}"
SUBSET="$OUT_DIR/train-${ROWS}.parquet"
CHECKPOINT_DIR="$OUT_DIR/sft-checkpoint"
PROMPTS="$OUT_DIR/prompts.jsonl"
COMPLETIONS="$OUT_DIR/completions.jsonl"
EXECUTED="$OUT_DIR/executed.jsonl"

mkdir -p "$OUT_DIR"

uv run --project "$VERL_DIR" --no-sync python "$REPO_ROOT/training/create_sft_subset.py" \
  --input "$DATASET_ROOT/data/sft/train.parquet" --output "$SUBSET" --rows "$ROWS"

# Save only the mandatory final checkpoint.  The trainer always saves its last
# step, avoiding repeated large FSDP checkpoint writes during this diagnostic.
QWEN35_MODEL_SIZE=0.8b \
TRAIN_FILE="$SUBSET" \
VAL_FILE="$SUBSET" \
SAVE_DIR="$CHECKPOINT_DIR" \
TOTAL_EPOCHS="$TOTAL_EPOCHS" \
LR="$LR" \
TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
MICRO_BATCH_SIZE_PER_GPU="$MICRO_BATCH_SIZE_PER_GPU" \
MAX_LENGTH="$MAX_LENGTH" \
MAX_TOKEN_LEN_PER_GPU="$MAX_TOKEN_LEN_PER_GPU" \
RESUME_MODE=disable \
SAVE_FREQ=-1 \
TEST_FREQ=-1 \
MAX_CKPT_TO_KEEP=1 \
EXPORT_FOR_INFERENCE=true \
bash "$REPO_ROOT/training/scripts/run_verl_sft.sh"

MODEL="$CHECKPOINT_DIR/huggingface"
uv run --project "$VERL_DIR" --no-sync python "$DATASET_ROOT/tools/make-data/export_parquet_prompts.py" \
  --input "$SUBSET" --output "$PROMPTS"
uv run --project "$VERL_DIR" --no-sync python "$REPO_ROOT/inference/run_prompts.py" \
  --model "$MODEL" --prompts "$PROMPTS" --prompt-field prompt --id-field id \
  --format chat --no-enable-thinking \
  --system-prompt "Return exactly one executable ROOT command. Do not add explanation." \
  --device cuda --temperature 0 --max-new-tokens 256 --output "$COMPLETIONS"

"$REPO_ROOT/training/scripts/root_runtime.sh" root -l -b -q \
  "$DATASET_ROOT/benchmark/evaluate_docs_completions.C(\"$COMPLETIONS\",\"$DATASET_ROOT/benchmark/expected-results/train.jsonl\",\"$EXECUTED\")"
uv run --project "$VERL_DIR" --no-sync python "$DATASET_ROOT/tools/benchmark/build_report.py" \
  --dataset "$SUBSET" --expected "$DATASET_ROOT/benchmark/expected-results/train.jsonl" --ids "$COMPLETIONS" \
  --completion "overfit-${ROWS}=$EXECUTED" \
  --output-json "$OUT_DIR/report.json" --output-csv "$OUT_DIR/report.csv" \
  --summary-csv "$OUT_DIR/summary.csv"

jq '.models' "$OUT_DIR/report.json"
printf 'Overfit diagnostic report: %s\n' "$OUT_DIR/report.json"
