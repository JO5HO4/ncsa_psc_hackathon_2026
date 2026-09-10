#!/usr/bin/env bash
# Run a command in the project's lightweight CVMFS ROOT runtime.
#
# This is intentionally not a Python virtualenv: a venv cannot provide ROOT's
# C++ binaries and shared libraries.  StatAnalysis from CVMFS provides the
# pinned ROOT runtime; the Python interpreter is captured before `asetup`
# because the release's bundled Python may require libcrypt.so.2 on Perlmutter.

# atlasLocalSetup.sh is itself a sourced shell environment and accesses unset
# optional variables.  Do not enable errexit/nounset until after it completes.
set -o pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 [--current-root] --python SCRIPT.py [args...] | $0 [--current-root] COMMAND [args...]" >&2
  exit 2
fi

# Use an already configured ROOT installation, for example after
# `lsetup "root 6.30.02-x86_64-centos7-gcc11-opt"` in an EL7 container.
# This deliberately bypasses CVMFS StatAnalysis, which is an EL9 build.
if [[ $1 == "--current-root" ]]; then
  shift
  if [[ $# -eq 0 ]] || ! command -v root >/dev/null 2>&1; then
    echo "--current-root requires an already configured root command" >&2
    exit 2
  fi
  exec "$@"
fi

host_python="$(command -v python3)"
export ATLAS_LOCAL_ROOT_BASE="${ATLAS_LOCAL_ROOT_BASE:-/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase}"
source "$ATLAS_LOCAL_ROOT_BASE/user/atlasLocalSetup.sh" --quiet
asetup StatAnalysis,0.5.1
setup_status=$?
set -euo pipefail
if [[ $setup_status -ne 0 ]]; then
  echo "Failed to initialize StatAnalysis,0.5.1" >&2
  exit "$setup_status"
fi

if [[ $1 == "--python" ]]; then
  shift
  exec "$host_python" "$@"
fi
exec "$@"
