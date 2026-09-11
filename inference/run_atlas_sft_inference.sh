#!/usr/bin/env bash
# Generate held-out ATLAS command completions from an SFT export.
set -euo pipefail

if [[ -z "${MODEL:-}" ]]; then
  echo "Set MODEL to a Hugging Face model or exported SFT checkpoint directory." >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ATLAS_DATA="${ATLAS_DATA:-$REPO_ROOT/data/datasets/atlas-open-data-sft-dataset}"
PROMPTS="${PROMPTS:-$REPO_ROOT/artifacts/atlas-test-prompts.jsonl}"
COMPLETIONS="${COMPLETIONS:-$REPO_ROOT/artifacts/atlas-sft-completions.jsonl}"

uv run --project "$REPO_ROOT/verl" --no-sync python "$ATLAS_DATA/tools/make-data/export_parquet_prompts.py" \
  --input "$ATLAS_DATA/data/sft/test.parquet" --output "$PROMPTS"
uv run --project "$REPO_ROOT/verl" --no-sync python "$REPO_ROOT/inference/run_prompts.py" \
  --model "$MODEL" --prompts "$PROMPTS" --prompt-field prompt --id-field id \
  --format chat --no-enable-thinking \
  --system-prompt "Return exactly one executable ROOT command. Do not add explanation." \
  --device "${DEVICE:-cuda}" --temperature 0 --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
  --output "$COMPLETIONS"

printf 'Completions: %s\n' "$COMPLETIONS"
