#!/usr/bin/env bash
# Opt-in GPU integration check. Run inside training/scripts/container.sh.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
run_id="${ROOT_SFT_GPU_RUN:-root-sft-gpu-smoke}"

./root-sft train --model qwen3.5-0.8b --epochs 1 --run "$run_id"
./root-sft infer --run "$run_id"
