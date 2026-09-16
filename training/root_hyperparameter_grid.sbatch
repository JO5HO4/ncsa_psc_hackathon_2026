#!/usr/bin/env bash
# Submit from repository root: mkdir -p results/slurm; sbatch training/root_hyperparameter_grid.sbatch
#SBATCH --job-name=root-qwen-grid
#SBATCH --account=atlas_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --time=06:00:00
#SBATCH --array=0-2%1
#SBATCH --output=results/slurm/root-grid-%A_%a.out
#SBATCH --error=results/slurm/root-grid-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
# Controlled comparisons; all other settings match the existing experiments.
case "${SLURM_ARRAY_TASK_ID:?}" in
  0) export LR=5e-6 TOTAL_EPOCHS=10 ;;
  1) export LR=1e-5 TOTAL_EPOCHS=3 ;;
  2) export LR=1e-5 TOTAL_EPOCHS=15 ;;
  *) echo 'Unknown grid row'; exit 2 ;;
esac
export EVAL_SPLIT=validation
export RECOVERY_CHECKPOINT=''
export RUN_ROOT_SCORING=true
export TRAIN_BATCH_SIZE=16 MICRO_BATCH_SIZE_PER_GPU=1
export MAX_LENGTH=2048 MAX_TOKEN_LEN_PER_GPU=8192
export LORA_RANK=16 LORA_ALPHA=16 MAX_CKPT_TO_KEEP=3
export BASE_REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a
export RUN_NAME="grid-${SLURM_ARRAY_JOB_ID}-row-${SLURM_ARRAY_TASK_ID}-lr${LR}-e${TOTAL_EPOCHS}"
export SLURM_CPU_BIND=cores
srun --ntasks=1 --gpus-per-task=4 bash training/run_root_pipeline.sh
