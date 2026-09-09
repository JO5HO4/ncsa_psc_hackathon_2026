#!/usr/bin/env bash
# Run a command in the project's lightweight CVMFS ROOT runtime.
#
# This is intentionally not a Python virtualenv: a venv cannot provide ROOT's
# C++ binaries and shared libraries.  StatAnalysis from CVMFS provides the
# pinned ROOT runtime; the Python interpreter is captured before `asetup`
# because the release's bundled Python may require libcrypt.so.2 on Perlmutter.

set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 --python SCRIPT.py [args...] | $0 COMMAND [args...]" >&2
  exit 2
fi

host_python="$(command -v python3)"
export ATLAS_LOCAL_ROOT_BASE="${ATLAS_LOCAL_ROOT_BASE:-/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase}"
source "$ATLAS_LOCAL_ROOT_BASE/user/atlasLocalSetup.sh" --quiet
asetup StatAnalysis,0.5.1

if [[ $1 == "--python" ]]; then
  shift
  exec "$host_python" "$@"
fi
exec "$@"
