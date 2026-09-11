#!/usr/bin/env bash
# Execute ATLAS held-out completions in ROOT and build a machine-readable report.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${ATLAS_DATASET_ROOT:-$REPO_ROOT/data/datasets/atlas-open-data-sft-dataset}"
ROOT_RUNNER="${ROOT_RUNNER:-$REPO_ROOT/training/scripts/root_runtime.sh}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 COMPLETIONS.jsonl [LABEL] [OUTPUT_DIR]" >&2
  exit 2
fi

completions="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
label="${2:-sft}"
output_dir="${3:-$REPO_ROOT/artifacts/benchmarks/$label}"
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"

if [[ ! -r "$completions" ]]; then
  echo "Completion file not found: $completions" >&2
  exit 2
fi
if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "ATLAS dataset checkout not found: $DATASET_ROOT" >&2
  exit 2
fi

executed="$output_dir/$label-executed.jsonl"
report_json="$output_dir/$label-report.json"
report_csv="$output_dir/$label-report.csv"
summary_csv="$output_dir/$label-summary.csv"
macro="$DATASET_ROOT/benchmark/evaluate_docs_completions.C"
expected="$DATASET_ROOT/benchmark/expected-results/test.jsonl"

"$ROOT_RUNNER" root -l -b -q "${macro}(\"${completions}\",\"${expected}\",\"${executed}\")"
"$PYTHON_BIN" "$DATASET_ROOT/tools/benchmark/build_report.py" \
  --dataset "$DATASET_ROOT/data/sft/test.parquet" \
  --expected "$expected" \
  --completion "$label=$executed" \
  --output-json "$report_json" \
  --output-csv "$report_csv" \
  --summary-csv "$summary_csv"

printf 'Benchmark complete\n  Executed: %s\n  Report: %s\n  Summary: %s\n' "$executed" "$report_json" "$summary_csv"
