#!/usr/bin/env bash
# Source this inside the VERL container:
#   source training/setup.sh
#
# This prepares the current container session for training. It does not bake a
# new image; container changes are lost after exiting because training/container.sh uses
# podman-hpc run --rm.

set -euo pipefail

if [ ! -d /workspace/verl ]; then
  echo "Expected /workspace/verl. Start the container from the repo root with: source training/container.sh" >&2
  return 1 2>/dev/null || exit 1
fi

cd /workspace/verl

# Install the local VERL checkout into the current container environment.
python -m pip install --no-deps -e .

# Required by the current VERL checkout/container combination.
python -m pip install --no-cache-dir TransferQueue==0.1.8

# Use node-local cache to avoid CFS file-lock issues when downloading models.
unset TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-$HF_HOME/assets}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_ASSETS_CACHE" "$HF_XET_CACHE"

# Clean up Ray state from interrupted attempts.
ray stop --force >/dev/null 2>&1 || true

cd /workspace

python - <<'PY'
import importlib.util

import torch
import transfer_queue
import verl

print("verl:", verl.__file__)
print("main_ppo:", importlib.util.find_spec("verl.trainer.main_ppo").origin)
print("cuda:", torch.cuda.is_available(), torch.cuda.device_count())
print("transfer_queue:", transfer_queue.__file__)
PY
