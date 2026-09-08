#!/usr/bin/env bash
# Shared Qwen3.5 model selection for the SFT and RL launchers.
# Source this file from a launcher; it exports the selected model metadata.

QWEN35_MODEL_SIZE="${QWEN35_MODEL_SIZE:-0.8b}"

case "$QWEN35_MODEL_SIZE" in
  0.8b)
    QWEN35_MODEL_PATH="Qwen/Qwen3.5-0.8B"
    QWEN35_PROFILE_NAME="qwen35-0.8b"
    QWEN35_DEFAULT_GPUS=1
    ;;
  9b)
    QWEN35_MODEL_PATH="Qwen/Qwen3.5-9B"
    QWEN35_PROFILE_NAME="qwen35-9b"
    QWEN35_DEFAULT_GPUS=4
    ;;
  *)
    echo "Unsupported QWEN35_MODEL_SIZE=$QWEN35_MODEL_SIZE. Use 0.8b or 9b." >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

# MODEL_PATH remains available for a pinned revision or local snapshot, while
# QWEN35_MODEL_SIZE controls the resource profile.
MODEL_PATH="${MODEL_PATH:-$QWEN35_MODEL_PATH}"

export QWEN35_MODEL_SIZE QWEN35_MODEL_PATH QWEN35_PROFILE_NAME QWEN35_DEFAULT_GPUS MODEL_PATH
