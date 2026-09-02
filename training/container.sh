REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHM_SIZE="${SHM_SIZE:-16g}"
IPC_MODE="${IPC_MODE:-host}"
IPC_ARGS=(--ipc="$IPC_MODE")

if [ "$IPC_MODE" != "host" ]; then
  IPC_ARGS+=(--shm-size="$SHM_SIZE")
fi

podman-hpc run --rm -it --gpu --entrypoint= \
  "${IPC_ARGS[@]}" \
  -v "$REPO_ROOT":/workspace \
  -w /workspace \
  docker.io/verlai/verl:sgl059.latest \
  /bin/bash
