#!/usr/bin/env bash
# Submit from repository root after mkdir -p results/slurm.
#SBATCH --job-name=root-qwen35-before-after
#SBATCH --account=atlas_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --time=06:00:00
#SBATCH --output=results/slurm/root-before-after-%j.out
#SBATCH --error=results/slurm/root-before-after-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
[[ "$TOTAL_EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo 'TOTAL_EPOCHS must be a positive integer'; exit 2; }
run_relative="results/root-command-qwen35-9b/pipeline-${SLURM_JOB_ID}"
mkdir -p results/root-command-qwen35-9b
mkdir "$run_relative"
run_dir="$PWD/$run_relative"
exec > >(tee "$run_dir/job.log") 2>&1
trap 'code=$?; printf "exit_code=%s finished=%s\n" "$code" "$(date -u +%FT%TZ)" > "$run_dir/status.txt"' EXIT
printf 'job=%s epochs=%s started=%s\n' "$SLURM_JOB_ID" "$TOTAL_EPOCHS" "$(date -u +%FT%TZ)"
git rev-parse HEAD
git submodule status data/datasets/atlas-open-data-sft-dataset verl
nvidia-smi
cp training/root_before_after.py "$run_dir/"
cp inference/root_eval_pair.py "$run_dir/"
image=docker.io/verlai/verl@sha256:26b2b1de89333cfbddbf639be5f0ab9d2dbaed19ee5b4892765f59617671e915
cache_root="${PSCRATCH:-${SCRATCH:?}}"
mkdir -p "$cache_root/qwen35-hf-cache" "$cache_root/qwen35-pipeline-${SLURM_JOB_ID}"
export SLURM_CPU_BIND=cores
srun --ntasks=1 --gpus-per-task=4 podman-hpc run --rm --gpu --ipc=host --entrypoint= \
  -v "$PWD":/workspace -v "$cache_root/qwen35-hf-cache":/hf_cache \
  -v "$cache_root/qwen35-pipeline-${SLURM_JOB_ID}":/tmp \
  -w /workspace -e CUDA_VISIBLE_DEVICES -e OMP_NUM_THREADS=8 \
  -e HF_HOME=/hf_cache -e HF_HUB_CACHE=/hf_cache/hub -e HF_HUB_DISABLE_XET=1 \
  -e UV_CACHE_DIR=/hf_cache/uv -e UV_PROJECT_ENVIRONMENT=/tmp/verl-venv \
  -e TMPDIR=/tmp -e PYTHONUNBUFFERED=1 -e TOTAL_EPOCHS="$TOTAL_EPOCHS" \
  "$image" bash -c '
    set -euo pipefail
    unset VIRTUAL_ENV
    cd /workspace/verl
    uv sync --frozen --extra fsdp --extra sglang
    cd /workspace
    uv run --project /workspace/verl --no-sync python "$1/root_before_after.py" "$1"
  ' bash "/workspace/$run_relative"
