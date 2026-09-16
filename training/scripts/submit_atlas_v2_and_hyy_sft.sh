#!/usr/bin/env bash
# Queue the independent V2 and Hyy Qwen3.5-9B SFT jobs together.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
v2_job="$(sbatch --parsable "$repo_root/training/scripts/train_atlas_v2_qwen35_9b.sbatch")"
if ! hyy_job="$(sbatch --parsable "$repo_root/training/scripts/train_hyy_sft_qwen35_9b.sbatch")"; then
  scancel "$v2_job" || true
  exit 1
fi

printf 'Queued V2 SFT job: %s\nQueued Hyy SFT job: %s\n' "$v2_job" "$hyy_job"
