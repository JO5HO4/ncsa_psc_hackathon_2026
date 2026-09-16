#!/usr/bin/env bash
# Submit from the repository root; supply your GPU account with sbatch --account=...
#SBATCH --job-name=root-sft
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=root-sft-%j.out
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"
[[ -f "$REPO_ROOT/training/root_sft/run.sh" ]] || { echo 'Submit from repository root' >&2; exit 2; }
MODE="${MODE:-smoke}"
VERL_IMAGE="${VERL_IMAGE:-docker.io/verlai/verl:sgl059.latest}"
# New per-job output avoids accidental resuming or overwriting. Repository is mounted /workspace.
RUN_DIR="/workspace/artifacts/root-sft/${MODE}-${SLURM_JOB_ID}"
export SLURM_CPU_BIND=cores
export OMP_NUM_THREADS=8
VERL_IMAGE_ID="$(podman-hpc image inspect "$VERL_IMAGE" --format '{{.Id}}')"
srun --ntasks=1 --gpus-per-task=1 podman-hpc run --rm --gpu --ipc=host --entrypoint= \
  -v "$REPO_ROOT":/workspace -w /workspace \
  -e RUN_DIR="$RUN_DIR" -e MODE="$MODE" \
  -e CUDA_VISIBLE_DEVICES \
  -e VERL_IMAGE="$VERL_IMAGE" -e VERL_IMAGE_ID="$VERL_IMAGE_ID" \
  -e DATA_DIR=/workspace/data/root_io/splits/root-sft-v1 \
  -e MODEL_REVISION="${MODEL_REVISION:-main}" \
  -e HF_HOME=/tmp/root-sft-hf -e OMP_NUM_THREADS=8 \
  "$VERL_IMAGE" /bin/bash -lc \
  'source training/setup.sh && python -m pip install -r training/root_sft/requirements.txt && bash training/root_sft/run.sh "$MODE"'
