#!/usr/bin/env bash
# ROOT execution benchmark for Qwen3.5-9B base and V2-SFT completions.
# Run from a ROOT 6.40.02 container or shell:
#   bash inference/verify_atlas_v2_qwen35_9b.sh
set -euo pipefail

repo=/global/cfs/cdirs/atlas/joshua/ncsa_psc_hackathon_2026
dataset="$repo/data/datasets/atlas-open-data-sft-dataset"
catalog="$dataset/v2/candidates.jsonl"
manifest="$dataset/v2/split_manifest.json"
oracle="$dataset/v2/result_oracle.jsonl"
run_dir="$repo/artifacts/inference/atlas-v2-qwen35-9b-r32-e10"
execution_dir="$run_dir/execution"
shards="${SHARDS:-8}"

if [[ $(root-config --version) != "6.40.02" ]]; then
  echo "ROOT 6.40.02 is required; found $(root-config --version)" >&2
  exit 2
fi
if ! [[ "$shards" =~ ^[1-9][0-9]*$ ]]; then
  echo "SHARDS must be a positive integer" >&2
  exit 2
fi
if [[ ! -f "$oracle" ]]; then
  echo "Strict result oracle is missing; first run: (cd \"$dataset\" && bash v2/run_sharded_oracle.sh 8)" >&2
  exit 2
fi
mkdir -p "$execution_dir"

for model in base sft; do
  python3 "$repo/inference/materialize_v2_model_catalog.py" \
    --catalog "$catalog" --split-manifest "$manifest" --split test \
    --completions "$run_dir/$model.jsonl" \
    --output "$execution_dir/$model-test-catalog.jsonl"
done

run_model() {
  local model="$1" report_dir="$execution_dir/$model-shards"
  local pids=() shard
  if [[ -e "$report_dir" || -e "$execution_dir/$model-results.jsonl" ]]; then
    echo "Refusing to overwrite existing $model execution results under $execution_dir" >&2
    exit 2
  fi
  mkdir -p "$report_dir"
  for ((shard = 0; shard < shards; ++shard)); do
    (
      cd "$dataset"
      root -l -b -q \
        "v2/run_candidates.C(\"$execution_dir/$model-test-catalog.jsonl\",\"$report_dir/shard_$shard.jsonl\",\"\",$shard,$shards,\"$oracle\")" \
        >"$report_dir/shard_$shard.log" 2>&1
    ) &
    pids+=("$!")
  done
  # Individual commands failing is expected benchmark data, not a launcher failure.
  for pid in "${pids[@]}"; do wait "$pid" || true; done
  cat "$report_dir"/shard_*.jsonl >"$execution_dir/$model-results.jsonl"
}

run_model base
run_model sft

python3 "$repo/inference/plot_v2_execution_results.py" \
  --catalog "$execution_dir/base-test-catalog.jsonl" \
  --result "Base Qwen3.5-9B=$execution_dir/base-results.jsonl" \
  --result "V2 SFT 9B (r=32)=$execution_dir/sft-results.jsonl" \
  --output-dir "$execution_dir/plots"

printf 'Reports and plots: %s\n' "$execution_dir"
