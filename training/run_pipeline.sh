#!/usr/bin/env bash
# Config-driven pipeline: edit a copy of training/pipeline.conf.example, then
# run this from an interactive Perlmutter GPU allocation:
#
#   bash training/run_pipeline.sh training/pipeline.conf
#
# See training/pipeline.conf.example for every field and
# training/pipeline_runner.py for the stage logic this launches inside the
# container. Unlike training/run_root_pipeline.sh (which always runs all
# three stages against the fixed ATLAS-command dataset and produced the
# results cited in results/*.md), this script has independently toggleable
# stages and works against any train/validation/test files. It does not
# modify or replace run_root_pipeline.sh.
set -euo pipefail

CONFIG="${1:?Usage: run_pipeline.sh path/to/pipeline.conf}"
[[ -f "$CONFIG" ]] || { echo "Config file not found: $CONFIG" >&2; exit 2; }
set -a
# shellcheck source=/dev/null
source "$CONFIG"
set +a

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${RUN_NAME:?Set RUN_NAME in the config}"
: "${TASK_PROFILE:?Set TASK_PROFILE in the config}"
: "${QWEN35_MODEL_SIZE:?Set QWEN35_MODEL_SIZE in the config, or MODEL_PATH directly}"
RUN_BASELINE_INFERENCE="${RUN_BASELINE_INFERENCE:-false}"
RUN_TRAINING="${RUN_TRAINING:-false}"
RUN_POST_TRAINING_INFERENCE="${RUN_POST_TRAINING_INFERENCE:-false}"
RUN_ROOT_SCORING="${RUN_ROOT_SCORING:-false}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9_.-]+$ && "$RUN_NAME" != .* ]] || { echo 'Invalid RUN_NAME' >&2; exit 2; }

# Fail fast on config mistakes, before requesting any GPU work.
if [[ "$RUN_BASELINE_INFERENCE" != true && "$RUN_TRAINING" != true && "$RUN_POST_TRAINING_INFERENCE" != true ]]; then
  echo 'At least one of RUN_BASELINE_INFERENCE, RUN_TRAINING, RUN_POST_TRAINING_INFERENCE must be true' >&2
  exit 2
fi
if [[ "$RUN_TRAINING" == true ]]; then
  : "${TRAIN_FILE:?Set TRAIN_FILE in the config (RUN_TRAINING=true)}"
  : "${VAL_FILE:?Set VAL_FILE in the config (RUN_TRAINING=true)}"
fi
if [[ "$RUN_BASELINE_INFERENCE" == true || "$RUN_POST_TRAINING_INFERENCE" == true ]]; then
  : "${EVAL_FILE:?Set EVAL_FILE in the config}"
fi
if [[ "$RUN_POST_TRAINING_INFERENCE" == true && "$RUN_TRAINING" != true ]]; then
  [[ -n "${EXISTING_MODEL:-}" && -e "$EXISTING_MODEL" ]] || {
    echo 'RUN_POST_TRAINING_INFERENCE=true with RUN_TRAINING=false requires EXISTING_MODEL to point at an existing export' >&2
    exit 2
  }
fi
# Only these task profiles carry an expected_result + executable-command
# contract that inference/score_atlas_benchmark.sh can score. See
# training/pipeline_runner.py's EXECUTABLE_TASK_PROFILES (kept in sync here).
EXECUTABLE_TASK_PROFILES=(root_command)
if [[ "$RUN_ROOT_SCORING" == true ]]; then
  supported=false
  for profile in "${EXECUTABLE_TASK_PROFILES[@]}"; do
    [[ "$profile" == "$TASK_PROFILE" ]] && supported=true
  done
  [[ "$supported" == true ]] || {
    echo "RUN_ROOT_SCORING=true requires an executable task profile (${EXECUTABLE_TASK_PROFILES[*]}); got $TASK_PROFILE" >&2
    exit 2
  }
  [[ -r /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh ]] || {
    echo 'RUN_ROOT_SCORING=true but the CVMFS ATLAS release is not available on this node' >&2
    exit 2
  }
fi

# Paths in the config are relative to the repository root; make them
# absolute container paths (the repo is mounted at /workspace) so they
# resolve the same regardless of a subprocess's working directory inside it.
for key in TRAIN_FILE VAL_FILE EVAL_FILE EXISTING_MODEL; do
  value="${!key:-}"
  if [[ -n "$value" && "$value" != /* ]]; then
    printf -v "$key" '/workspace/%s' "$value"
  fi
done

[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Enter an interactive GPU allocation first.' >&2; exit 2; }
nvidia-smi --query-gpu=name --format=csv,noheader
visible_gpus="$(nvidia-smi -L | awk 'END {print NR}')"
[[ "$visible_gpus" == "$NPROC_PER_NODE" ]] || {
  echo "Config requests $NPROC_PER_NODE GPU(s); found $visible_gpus." >&2
  exit 2
}

relative="artifacts/pipeline-runs/$RUN_NAME"
mkdir -p artifacts/pipeline-runs
mkdir "$relative"  # Refuse to overwrite any previous run.
out="$REPO_ROOT/$relative"
cp "$CONFIG" "$out/pipeline.conf"
git rev-parse HEAD > "$out/git-revision.txt"
exec > >(tee "$out/job.log") 2>&1

# Same pinned Qwen3.5-capable image and cache-mount pattern as
# training/run_root_pipeline.sh / training/scripts/container.sh.
IMAGE=docker.io/verlai/verl@sha256:26b2b1de89333cfbddbf639be5f0ab9d2dbaed19ee5b4892765f59617671e915
cache="${PSCRATCH:-${SCRATCH:?Set PSCRATCH or SCRATCH}}/qwen35-hf-cache"
temporary="${PSCRATCH:-$SCRATCH}/qwen35-$RUN_NAME"
mkdir -p "$cache" "$temporary"

env_args=()
for key in RUN_BASELINE_INFERENCE RUN_TRAINING RUN_POST_TRAINING_INFERENCE RUN_ROOT_SCORING \
           QWEN35_MODEL_SIZE MODEL_PATH BASE_REVISION TASK_PROFILE EVAL_FILE \
           TRAIN_FILE VAL_FILE EXISTING_MODEL RUN_NAME \
           LR TOTAL_EPOCHS TRAIN_BATCH_SIZE MICRO_BATCH_SIZE_PER_GPU MAX_LENGTH \
           MAX_TOKEN_LEN_PER_GPU LORA_RANK LORA_ALPHA NPROC_PER_NODE; do
  [[ -v "$key" ]] || continue
  printf '%s=%q\n' "$key" "${!key}" >> "$out/configuration.txt"
  env_args+=(-e "$key=${!key}")
done

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
    uv run --project /workspace/verl --no-sync python training/pipeline_runner.py "$1"
  ' bash "/workspace/$relative"

if [[ "$RUN_ROOT_SCORING" == true ]]; then
  for name in baseline trained; do
    completions="$out/$name/completions.jsonl"
    [[ -f "$completions" ]] || continue
    bash inference/score_atlas_benchmark.sh "$completions" "$name" "$out/analysis/$name-score" \
      |& tee "$out/score-$name.log"
  done
fi

printf 'Pipeline completed. Logs, configuration, completions and scores: %s\n' "$out"
