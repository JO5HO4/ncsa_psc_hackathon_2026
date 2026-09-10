#!/usr/bin/env bash
# Source this inside the VERL container:
#   source training/scripts/setup.sh
#
# This prepares the current container session for training. It does not bake a
# new image; container changes are lost after exiting because training/scripts/container.sh uses
# podman-hpc run --rm.

set -euo pipefail

# Podman-HPC may inherit a host virtual environment. The pinned uv project
# environment below must take precedence for the Qwen3.5 runtime.
unset VIRTUAL_ENV

if [ ! -d /workspace/verl ]; then
  echo "Expected /workspace/verl. Start the container from the repo root with: bash training/scripts/container.sh" >&2
  return 1 2>/dev/null || exit 1
fi

cd /workspace/verl

# uv defaults to ~/.cache/uv, where Perlmutter's filesystem does not support
# the advisory locks used for concurrent cache access. Keep it beside the HF
# cache on the disk-backed bind mount instead.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/hf_cache/uv}"
mkdir -p "$UV_CACHE_DIR"

# The pinned uv image has an offline-warmed dependency cache. Keep its virtual
# environment off the mounted checkout, then select the FSDP + SGLang runtime
# needed by SFT and RL respectively.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/verl-venv}"
uv sync --frozen --extra fsdp --extra sglang

# The container launcher bind-mounts a disk-backed cache at /hf_cache. It keeps
# large checkpoints outside the writable layer and RAM-backed /tmp.
# The fallback also works for a manually started compatible container.
unset TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-$HF_HOME/assets}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_ASSETS_CACHE" "$HF_XET_CACHE"

# Clean up Ray state from interrupted attempts.
ray stop --force >/dev/null 2>&1 || true

cd /workspace

uv run --project /workspace/verl --no-sync python - <<'PY'
import importlib.util

import importlib.metadata

import torch
import transformers
import verl

if importlib.util.find_spec("transformers.models.qwen3_5") is None:
    raise RuntimeError(
        "This container cannot load Qwen3.5. Start the pinned image with "
        "bash training/scripts/container.sh (or set VERL_IMAGE to a compatible image)."
    )

print("verl:", verl.__file__)
print("main_ppo:", importlib.util.find_spec("verl.trainer.main_ppo").origin)
print("cuda:", torch.cuda.is_available(), torch.cuda.device_count())
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("sglang:", importlib.metadata.version("sglang"))
print("qwen3_5:", "available")
PY
