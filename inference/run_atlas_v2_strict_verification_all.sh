#!/usr/bin/env bash
# Strictly score every V2 held-out completion set against the gold RESULT oracle.
# Run in a ROOT 6.40.02 container: bash inference/run_atlas_v2_strict_verification_all.sh
set -euo pipefail

repo="${REPO_ROOT:-/global/cfs/cdirs/atlas/joshua/ncsa_psc_hackathon_2026}"
dataset="$repo/data/datasets/atlas-open-data-sft-dataset"
catalog="$dataset/v2/candidates.jsonl"
manifest="$dataset/v2/split_manifest.json"
oracle="$dataset/v2/result_oracle.jsonl"
output="$repo/artifacts/inference/atlas-v2-strict-comparison"
shards="${SHARDS:-8}"

if [[ $(root-config --version) != "6.40.02" ]]; then
  echo "ROOT 6.40.02 is required; found $(root-config --version)" >&2
  exit 2
fi
if ! [[ "$shards" =~ ^[1-9][0-9]*$ ]]; then
  echo "SHARDS must be a positive integer" >&2
  exit 2
fi
if [[ ! -s "$oracle" ]]; then
  echo "Strict result oracle is missing or empty: $oracle" >&2
  exit 2
fi
if [[ -e "$output" ]]; then
  echo "Refusing to overwrite existing strict benchmark output: $output" >&2
  exit 2
fi

declare -a labels=(
  "Base 0.8B"
  "V2 SFT 0.8B (r=32)"
  "Base 9B"
  "V2 SFT 9B (r=32)"
  "GPT-6 Astra"
  "Claude Opus 5"
)
declare -a slugs=(base-08b sft-08b-r32 base-9b sft-9b-r32 gpt-6-astra claude-opus-5)
declare -a completions=(
  "$repo/artifacts/inference/atlas-v2-qwen35-0.8b-r32/base.jsonl"
  "$repo/artifacts/inference/atlas-v2-qwen35-0.8b-r32/sft-global_step_785.jsonl"
  "$repo/artifacts/inference/atlas-v2-qwen35-9b-r32-e10/base.jsonl"
  "$repo/artifacts/inference/atlas-v2-qwen35-9b-r32-e10/sft.jsonl"
  "$repo/artifacts/inference/gpt-6-astra.jsonl"
  "$repo/artifacts/inference/claude-opus-5.jsonl"
)

mkdir -p "$output"

for index in "${!slugs[@]}"; do
  slug="${slugs[$index]}"
  completion="${completions[$index]}"
  model_catalog="$output/$slug-catalog.jsonl"
  report_dir="$output/$slug-shards"
  result="$output/$slug-results.jsonl"
  if [[ ! -f "$completion" ]]; then
    echo "Missing completion file: $completion" >&2
    exit 2
  fi

  python3 "$repo/inference/materialize_v2_model_catalog.py" \
    --catalog "$catalog" --split-manifest "$manifest" --split test \
    --completions "$completion" --output "$model_catalog"

  mkdir -p "$report_dir"
  pids=()
  for ((shard = 0; shard < shards; ++shard)); do
    (
      cd "$dataset"
      root -l -b -q \
        "v2/run_candidates.C(\"$model_catalog\",\"$report_dir/shard_$shard.jsonl\",\"\",$shard,$shards,\"$oracle\")" \
        >"$report_dir/shard_$shard.log" 2>&1
    ) &
    pids+=("$!")
  done
  # Nonzero ROOT exits represent failed benchmark rows; retain the JSONL reports.
  for pid in "${pids[@]}"; do wait "$pid" || true; done
  cat "$report_dir"/shard_*.jsonl >"$result"
  printf '%s strict report: %s\n' "${labels[$index]}" "$result"
done

plot_args=(--catalog "$output/${slugs[0]}-catalog.jsonl")
for index in "${!slugs[@]}"; do
  plot_args+=(--result "${labels[$index]}=$output/${slugs[$index]}-results.jsonl")
done
python3 "$repo/inference/plot_v2_execution_results.py" "${plot_args[@]}" --output-dir "$output/plots"

printf 'Strict reports and plots: %s\n' "$output"
