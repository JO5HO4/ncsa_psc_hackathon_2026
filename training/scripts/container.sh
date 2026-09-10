REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHM_SIZE="${SHM_SIZE:-16g}"
IPC_MODE="${IPC_MODE:-host}"
# This digest resolves to the linux/amd64 image. It is intentionally pinned:
# Qwen3.5 requires a Transformers stack that recognizes `qwen3_5`.
VERL_IMAGE="${VERL_IMAGE:-docker.io/verlai/verl@sha256:26b2b1de89333cfbddbf639be5f0ab9d2dbaed19ee5b4892765f59617671e915}"
# Keep large model downloads outside both the container overlay and RAM-backed
# /tmp. On Perlmutter, PSCRATCH is disk-backed Lustre and supports HF locks.
CACHE_ROOT="${PSCRATCH:-${SCRATCH:-$REPO_ROOT/artifacts/hf_cache}}"
HF_CACHE_HOST="${HF_CACHE_HOST:-$CACHE_ROOT/qwen35-hf-cache}"
# Podman-HPC's default /tmp is a RAM-backed, job-private tmpfs. Bind it to
# disk as well: interrupted HF/model loads otherwise keep their pages charged
# to the Slurm cgroup until the interactive allocation ends.
CONTAINER_TMP_HOST="${CONTAINER_TMP_HOST:-$CACHE_ROOT/qwen35-container-tmp}"
mkdir -p "$HF_CACHE_HOST" "$CONTAINER_TMP_HOST"
IPC_ARGS=(--ipc="$IPC_MODE")

if [ "$IPC_MODE" != "host" ]; then
  IPC_ARGS+=(--shm-size="$SHM_SIZE")
fi

podman-hpc run --rm -it --gpu --entrypoint= \
  "${IPC_ARGS[@]}" \
  -v "$REPO_ROOT":/workspace \
  -v "$HF_CACHE_HOST":/hf_cache \
  -v "$CONTAINER_TMP_HOST":/tmp \
  -e HF_HOME=/hf_cache \
  -e HF_HUB_CACHE=/hf_cache/hub \
  -e HF_ASSETS_CACHE=/hf_cache/assets \
  -e HF_XET_CACHE=/hf_cache/xet \
  -e HF_HUB_DISABLE_XET=1 \
  -e UV_CACHE_DIR=/hf_cache/uv \
  -e TMPDIR=/tmp \
  -w /workspace \
  "$VERL_IMAGE" \
  /bin/bash
