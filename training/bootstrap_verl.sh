#!/usr/bin/env bash
# Initialize the linked verl and dataset repositories after a non-recursive clone.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_DIR="$REPO_ROOT/verl"

if [[ ! -f "$REPO_ROOT/.gitmodules" ]]; then
  echo "No .gitmodules file found in $REPO_ROOT" >&2
  exit 1
fi

git -C "$REPO_ROOT" submodule sync --recursive
git -C "$REPO_ROOT" submodule update --init --recursive

if [[ ! -d "$VERL_DIR" ]]; then
  echo "verl submodule was not initialized: $VERL_DIR" >&2
  exit 1
fi

echo "Submodules are ready."
git -C "$REPO_ROOT" submodule status
