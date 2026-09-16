#!/usr/bin/env bash
# Run from the repository root; pass --account=atlas_g and a prepared BENCHMARK_DIR.
#SBATCH --job-name=qwen-root-repeat
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=qwen-root-repeat-%j.out
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
BENCHMARK_DIR="${BENCHMARK_DIR:?Set a prepared repository-relative benchmark directory}"
[[ "$BENCHMARK_DIR" == artifacts/root-sft/* && "$BENCHMARK_DIR" != *..* ]] || exit 2
[[ -f "$REPO_ROOT/$BENCHMARK_DIR/repeat_source.json" ]] || exit 2
VERL_IMAGE="${VERL_IMAGE:-docker.io/verlai/verl:sgl059.latest}"
export SLURM_CPU_BIND=cores
srun --ntasks=1 --gpus-per-task=1 podman-hpc run --rm --gpu --ipc=host --entrypoint= \
  -v "$REPO_ROOT":/workspace -w /workspace -e CUDA_VISIBLE_DEVICES \
  -e OMP_NUM_THREADS=8 -e HF_HOME=/tmp/root-repeat-hf \
  "$VERL_IMAGE" python -u training/root_sft/repeat_benchmark.py run \
  --directory "/workspace/$BENCHMARK_DIR"
