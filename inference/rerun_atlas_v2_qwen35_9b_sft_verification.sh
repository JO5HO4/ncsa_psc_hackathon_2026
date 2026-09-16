#!/usr/bin/env bash
# Re-run only the stale 9B-SFT ROOT verification; preserve its old reports.
set -euo pipefail

repo=/global/cfs/cdirs/atlas/joshua/ncsa_psc_hackathon_2026
dataset="$repo/data/datasets/atlas-open-data-sft-dataset"
oracle="$dataset/v2/result_oracle.jsonl"
run_dir="$repo/artifacts/inference/atlas-v2-qwen35-9b-r32-e10/execution"
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
if [[ ! -f "$run_dir/sft-test-catalog.jsonl" || ! -f "$run_dir/base-results.jsonl" ]]; then
  echo "Expected 9B inference catalogs and base report are missing." >&2
  exit 1
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
for path in "$run_dir/sft-shards" "$run_dir/sft-results.jsonl"; do
  if [[ -e "$path" ]]; then mv "$path" "$path.stale-$stamp"; fi
done
mkdir -p "$run_dir/sft-shards"

pids=()
for ((shard = 0; shard < shards; ++shard)); do
  (
    cd "$dataset"
    root -l -b -q \
      "v2/run_candidates.C(\"$run_dir/sft-test-catalog.jsonl\",\"$run_dir/sft-shards/shard_$shard.jsonl\",\"\",$shard,$shards,\"$oracle\")" \
      >"$run_dir/sft-shards/shard_$shard.log" 2>&1
  ) &
  pids+=("$!")
done
# Failures are expected model results; records are captured in JSONL.
for pid in "${pids[@]}"; do wait "$pid" || true; done
cat "$run_dir"/sft-shards/shard_*.jsonl >"$run_dir/sft-results.jsonl"

python3 "$repo/inference/plot_v2_execution_results.py" \
  --catalog "$run_dir/base-test-catalog.jsonl" \
  --result "Base Qwen3.5-9B=$run_dir/base-results.jsonl" \
  --result "V2 SFT 9B (r=32)=$run_dir/sft-results.jsonl" \
  --output-dir "$run_dir/plots"
