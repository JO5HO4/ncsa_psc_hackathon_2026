#!/usr/bin/env bash
# Execute and plot base-vs-SFT V2 test results. Run in a ROOT 6.40.02 shell.
set -euo pipefail

repo="${REPO_ROOT:-/global/cfs/cdirs/atlas/joshua/ncsa_psc_hackathon_2026}"
dataset="$repo/data/datasets/atlas-open-data-sft-dataset"
oracle="$dataset/v2/result_oracle.jsonl"
run_dir="$repo/artifacts/inference/atlas-v2-qwen35-0.8b-r32/execution"
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

run_model() {
  local label="$1" catalog="$run_dir/$2-test-catalog.jsonl"
  local report_dir="$run_dir/$label-shards"
  local pids=() shard
  if [[ -e "$report_dir" || -e "$run_dir/$label-results.jsonl" ]]; then
    echo "Refusing to overwrite existing $label execution output" >&2
    exit 2
  fi
  mkdir -p "$report_dir"
  for ((shard = 0; shard < shards; ++shard)); do
    (
      cd "$dataset"
      root -l -b -q \
        "v2/run_candidates.C(\"$catalog\",\"$report_dir/shard_$shard.jsonl\",\"\",$shard,$shards,\"$oracle\")" \
        >"$report_dir/shard_$shard.log" 2>&1
    ) &
    pids+=("$!")
  done
  # A nonzero ROOT exit means that this shard has failed model commands, which
  # is benchmark data rather than a launcher error. The JSONL report remains
  # the source of truth for plotting.
  for pid in "${pids[@]}"; do wait "$pid" || true; done
  cat "$report_dir"/shard_*.jsonl >"$run_dir/$label-results.jsonl"
}

run_model base base
run_model sft sft

python3 "$repo/inference/plot_v2_execution_results.py" \
  --catalog "$run_dir/base-test-catalog.jsonl" \
  --result "Base Qwen3.5-0.8B=$run_dir/base-results.jsonl" \
  --result "V2 SFT (r=32)=$run_dir/sft-results.jsonl" \
  --output-dir "$run_dir/plots"

printf 'Plots: %s\n' "$run_dir/plots"
