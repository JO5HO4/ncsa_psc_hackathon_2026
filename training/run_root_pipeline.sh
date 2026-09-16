#!/usr/bin/env bash
# Run on the HOST shell of an interactive Perlmutter GPU compute node.
set -euo pipefail

# ---- Editable configuration (environment overrides are also accepted) ----
# 9b uses four GPUs; 0.8b uses one. The source below selects the matching model.
QWEN35_MODEL_SIZE=${QWEN35_MODEL_SIZE:-9b}
DATASET_PROFILE=${DATASET_PROFILE:-canonical}
LR=${LR:-5e-6}
export EVAL_SPLIT=${EVAL_SPLIT:-test}
[[ "$EVAL_SPLIT" == test || "$EVAL_SPLIT" == validation ]] || { echo 'Invalid EVAL_SPLIT'; exit 2; }
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_LENGTH=${MAX_LENGTH:-2048}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-8192}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-16}
MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP:-3}
RUN_NAME=${RUN_NAME:-pipeline-$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID:-interactive}}
# Optional existing checkpoint, relative to repository root: skips training.
RECOVERY_CHECKPOINT=${RECOVERY_CHECKPOINT:-}
# Execute generated commands with the repository's ROOT benchmark scorer.
RUN_ROOT_SCORING=${RUN_ROOT_SCORING:-true}
# Reporting uses the existing llm_env; an explicit interpreter may override it.
REPORT_ENV=${REPORT_ENV:-llm_env}
PYTHON_BIN=${PYTHON_BIN:-}
IMAGE=docker.io/verlai/verl@sha256:26b2b1de89333cfbddbf639be5f0ab9d2dbaed19ee5b4892765f59617671e915
# -----------------------------------------------------------------------

# Perlmutter supplies Conda through the Python module.
# Activation scripts may reference optional unset shell variables.
set +u
module load python && conda activate "$REPORT_ENV" || exit "$?"
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source training/scripts/qwen35_profile.sh
BASE_MODEL=${BASE_MODEL:-$QWEN35_MODEL_PATH}
# Keep historical 9B runs pinned. The first 0.8B run records its resolved
# snapshot in the protocol, so it can be reproduced without assuming a hash.
if [[ "$QWEN35_MODEL_SIZE" == 9b ]]; then
  BASE_REVISION=${BASE_REVISION:-c202236235762e1c871ad0ccb60c8ee5ba337b9a}
else
  BASE_REVISION=${BASE_REVISION:-}
fi
NPROC_PER_NODE=${NPROC_PER_NODE:-$QWEN35_DEFAULT_GPUS}
case "$DATASET_PROFILE" in
  canonical)
    SOURCE_TRAIN_FILE=/workspace/data/datasets/atlas-open-data-sft-dataset/data/sft/train.parquet
    SOURCE_VAL_FILE=/workspace/data/datasets/atlas-open-data-sft-dataset/data/sft/validation.parquet
    DEFAULT_SYSTEM_PROMPT='Return exactly one executable ROOT command. Do not add explanation.'
    ;;
  qwen-root-command-chat-v1)
    SOURCE_TRAIN_FILE=/workspace/data/derived/qwen-root-command-chat-v1/train.parquet
    SOURCE_VAL_FILE=/workspace/data/derived/qwen-root-command-chat-v1/validation.parquet
    DEFAULT_SYSTEM_PROMPT="Return exactly one executable ROOT command on one line. The entire response must begin with root -l -b -q -e ' and end with one matching single quote. Inside it, print exactly one RESULT=<value> line and call gSystem->Exit(0). Do not add Markdown, backticks, explanations, XML, JSON, tool calls, or any other text."
    ;;
  *) echo "Unsupported DATASET_PROFILE=$DATASET_PROFILE" >&2; exit 2 ;;
esac
SYSTEM_PROMPT=${SYSTEM_PROMPT:-$DEFAULT_SYSTEM_PROMPT}
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ "${CONDA_DEFAULT_ENV:-}" == "$REPORT_ENV" && -n "${CONDA_PREFIX:-}" ]]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
  elif command -v conda >/dev/null 2>&1; then
    PYTHON_BIN="$(conda run -n "$REPORT_ENV" python -c 'import sys; print(sys.executable)')"
  else
    echo "Activate $REPORT_ENV first, or set PYTHON_BIN to its Python executable." >&2
    exit 2
  fi
fi
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Enter an interactive GPU allocation first.'; exit 2; }
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9_.-]+$ && "$RUN_NAME" != .* ]] || { echo 'Invalid RUN_NAME'; exit 2; }
nvidia-smi --query-gpu=name --format=csv,noheader
visible_gpus="$(nvidia-smi -L | awk 'END {print NR}')"
[[ "$visible_gpus" == "$NPROC_PER_NODE" ]] || {
  echo "Profile $QWEN35_MODEL_SIZE requires $NPROC_PER_NODE visible GPU(s); found $visible_gpus." >&2
  exit 2
}
if [[ "$RUN_ROOT_SCORING" == true ]]; then
  [[ -r /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh ]]
fi
if [[ -n "$RECOVERY_CHECKPOINT" ]]; then
  [[ "$RECOVERY_CHECKPOINT" == results/* && "$RECOVERY_CHECKPOINT" != *..* ]]
  [[ -f "$RECOVERY_CHECKPOINT/lora_train_meta.json" ]]
  RECOVERY_CHECKPOINT="/workspace/$RECOVERY_CHECKPOINT"
fi
result_group="root-command-$QWEN35_PROFILE_NAME"
if [[ "$DATASET_PROFILE" != canonical ]]; then
  result_group+="-$DATASET_PROFILE"
fi
relative="results/$result_group/$RUN_NAME"
mkdir -p "results/$result_group"
mkdir "$relative" # Refuse to overwrite any previous run.
out="$REPO_ROOT/$relative"
exec > >(tee "$out/job.log") 2>&1
trap 'code=$?; printf "exit_code=%s finished=%s\n" "$code" "$(date -u +%FT%TZ)" > "$out/status.txt"' EXIT
cp "${BASH_SOURCE[0]}" "$out/launcher.sh"
cp training/root_before_after.py inference/root_eval_pair.py "$out/"
git rev-parse HEAD > "$out/git-revision.txt"
# Reuse llm_env without installing or changing any packages.
# Check dependencies before expensive GPU work and record their versions.
export MPLCONFIGDIR="$out/.matplotlib"
export XDG_CACHE_HOME="$out/.cache"
"$PYTHON_BIN" -c 'import sys, matplotlib, pyarrow; print("python=" + sys.executable); print("matplotlib==" + matplotlib.__version__); print("pyarrow==" + pyarrow.__version__)' \
  | tee "$out/report-requirements.txt"
env_args=()
for key in QWEN35_MODEL_SIZE DATASET_PROFILE BASE_MODEL EVAL_SPLIT BASE_REVISION NPROC_PER_NODE SOURCE_TRAIN_FILE SOURCE_VAL_FILE SYSTEM_PROMPT LR TOTAL_EPOCHS TRAIN_BATCH_SIZE MICRO_BATCH_SIZE_PER_GPU MAX_LENGTH MAX_TOKEN_LEN_PER_GPU LORA_RANK LORA_ALPHA MAX_CKPT_TO_KEEP RECOVERY_CHECKPOINT; do
  printf '%s=%q\n' "$key" "${!key}" >> "$out/configuration.txt"
  env_args+=(-e "$key=${!key}")
done
cache="${PSCRATCH:-${SCRATCH:?Set PSCRATCH or SCRATCH}}/qwen35-hf-cache"
temporary="${PSCRATCH:-$SCRATCH}/qwen35-$RUN_NAME"
mkdir -p "$cache" "$temporary"
podman-hpc run --rm --gpu --ipc=host --entrypoint= \
  -v "$REPO_ROOT":/workspace -v "$cache":/hf_cache -v "$temporary":/tmp \
  -w /workspace -e CUDA_VISIBLE_DEVICES -e OMP_NUM_THREADS=8 \
  -e HF_HOME=/hf_cache -e HF_HUB_CACHE=/hf_cache/hub -e HF_HUB_DISABLE_XET=1 \
  -e UV_CACHE_DIR=/hf_cache/uv -e UV_PROJECT_ENVIRONMENT=/tmp/verl-venv \
  -e TMPDIR=/tmp -e PYTHONUNBUFFERED=1 "${env_args[@]}" "$IMAGE" bash -c '
    set -euo pipefail
    unset VIRTUAL_ENV
    cd /workspace/verl
    uv sync --frozen --extra fsdp --extra sglang
    cd /workspace
    uv run --project /workspace/verl --no-sync python "$1/root_before_after.py" "$1"
  ' bash "/workspace/$relative"
if [[ "$RUN_ROOT_SCORING" == true ]]; then
  for model in base lora; do
    PYTHON_BIN="$PYTHON_BIN" bash inference/score_atlas_benchmark.sh \
      "$out/$model/completions.jsonl" "$model" "$out/analysis/$model-score" \
      |& tee "$out/score-$model.log"
  done
fi
MPLCONFIGDIR="$out/.matplotlib" "$PYTHON_BIN" training/root_pipeline_report.py "$out" \
  |& tee "$out/report.log"
printf 'Pipeline completed. Logs, configuration, checkpoints, completions and scores: %s\n' "$out"
