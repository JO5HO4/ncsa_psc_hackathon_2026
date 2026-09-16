#!/usr/bin/env bash
# Submit from the repository root; see inference/PERLMUTTER_ROOT_EVAL.md.
#SBATCH --job-name=root-qwen35-eval
#SBATCH --account=atlas_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/root-eval-%j.out
#SBATCH --error=results/slurm/root-eval-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
repo_root=$PWD
mode=${1:-base}
[[ "$mode" == base || "$mode" == lora ]] || { echo 'Mode must be base or lora'; exit 2; }
baseline_args=()
adapter_args=()
if [[ "$mode" == lora ]]; then
  [[ $# == 3 ]] || { echo 'Usage: sbatch SCRIPT lora BASELINE_DIRECTORY ADAPTER_DIRECTORY'; exit 2; }
  baseline=$(realpath "$2")
  adapter=$(realpath "$3")
  [[ -f "$baseline/protocol.json" && -f "$baseline/completions.jsonl" ]] || exit 2
  [[ -f "$adapter/adapter_config.json" ]] || { echo 'Pass the directory containing adapter_config.json'; exit 2; }
  baseline_args=(-v "$baseline":/baseline:ro)
  adapter_args=(-v "$adapter":/adapter:ro)
fi
run_relative="results/root-command-qwen35-9b/${mode}-${SLURM_JOB_ID}"
run_dir="$repo_root/$run_relative"
mkdir -p "$repo_root/results/root-command-qwen35-9b"
mkdir "$run_dir"
exec > >(tee "$run_dir/job.log") 2>&1
trap 'code=$?; printf "exit_code=%s finished=%s\n" "$code" "$(date -u +%FT%TZ)" > "$run_dir/status.txt"' EXIT
printf 'mode=%s job_id=%s started=%s\n' "$mode" "$SLURM_JOB_ID" "$(date -u +%FT%TZ)"
git rev-parse HEAD
git submodule status data/datasets/atlas-open-data-sft-dataset verl
nvidia-smi
image=docker.io/verlai/verl@sha256:26b2b1de89333cfbddbf639be5f0ab9d2dbaed19ee5b4892765f59617671e915
cache_root="${PSCRATCH:-${SCRATCH:?}}"
model_cache="$cache_root/qwen35-hf-cache"
job_tmp="$cache_root/qwen35-eval-${SLURM_JOB_ID}"
mkdir -p "$model_cache" "$job_tmp"
# Preserve the driver used by this attempt.
cp inference/root_eval_pair.py "$run_dir/root_eval_pair.py"
export SLURM_CPU_BIND=cores
srun --ntasks=1 --gpus-per-task=1 podman-hpc run --rm --gpu --ipc=host --entrypoint= \
  -v "$repo_root":/workspace -v "$model_cache":/hf_cache -v "$job_tmp":/tmp \
  "${baseline_args[@]}" "${adapter_args[@]}" \
  -w /workspace -e CUDA_VISIBLE_DEVICES -e OMP_NUM_THREADS=8 \
  -e HF_HOME=/hf_cache -e HF_HUB_CACHE=/hf_cache/hub -e HF_HUB_DISABLE_XET=1 \
  -e UV_CACHE_DIR=/hf_cache/uv -e UV_PROJECT_ENVIRONMENT=/tmp/verl-venv \
  -e TMPDIR=/tmp -e PYTHONUNBUFFERED=1 \
  "$image" bash -c '
    set -euo pipefail
    unset VIRTUAL_ENV
    cd /workspace/verl
    uv sync --frozen --extra fsdp --extra sglang
    cd /workspace
    uv run --project /workspace/verl --no-sync python "$1/root_eval_pair.py" "$2" "$1"
  ' bash "/workspace/$run_relative" "$mode"
