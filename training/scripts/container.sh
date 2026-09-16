REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "$(uname -m)" == "aarch64" ]]; then
  # Delta-GH nodes are ARM64 GH200s. The regular Podman-HPC image below is
  # linux/amd64 and cannot run here; use the matching Apptainer image instead.
  DELTA_GH_CACHE_ROOT="${DELTA_GH_CACHE_ROOT:-$REPO_ROOT/artifacts/delta-gh-cache}"
  VERL_IMAGE="${VERL_IMAGE:-docker://verlai/verl@sha256:094943c1909590564feda8122bd090ba0a93aa42345245111a05225c1a7972f1}"
  VERL_IMAGE_SIF="${VERL_IMAGE_SIF:-$DELTA_GH_CACHE_ROOT/images/verl-uv-cu130-arm64.sif}"
  HF_CACHE_HOST="${HF_CACHE_HOST:-$DELTA_GH_CACHE_ROOT/hf-cache}"
  CONTAINER_TMP_HOST="${CONTAINER_TMP_HOST:-$DELTA_GH_CACHE_ROOT/tmp}"
  UV_CACHE_HOST="${UV_CACHE_HOST:-$DELTA_GH_CACHE_ROOT/uv-cache}"
  VERL_ENABLE_SGLANG="${VERL_ENABLE_SGLANG:-false}"

  mkdir -p "$HF_CACHE_HOST" "$CONTAINER_TMP_HOST" "$UV_CACHE_HOST" "$(dirname "$VERL_IMAGE_SIF")"
  if [[ ! -f "$VERL_IMAGE_SIF" ]]; then
    apptainer pull "$VERL_IMAGE_SIF" "$VERL_IMAGE"
  fi

  apptainer exec --nv --cleanenv \
    --bind "$REPO_ROOT:/workspace,$HF_CACHE_HOST:/hf_cache,$UV_CACHE_HOST:/uv_cache,$CONTAINER_TMP_HOST:/tmp" \
    --env HF_HOME=/hf_cache,HF_HUB_CACHE=/hf_cache/hub,HF_ASSETS_CACHE=/hf_cache/assets,HF_XET_CACHE=/hf_cache/xet,HF_HUB_DISABLE_XET=1,UV_CACHE_DIR=/uv_cache,TMPDIR=/tmp,VERL_ENABLE_SGLANG="$VERL_ENABLE_SGLANG" \
    --pwd /workspace \
    "$VERL_IMAGE_SIF" /bin/bash
fi

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
  -e HF_TOKEN \
  -e ROOT_SFT_IN_CONTAINER=1 \
  -e TREX_RL_REWARD_SOCKET \
  -e TREX_RL_REWARD_SECONDS \
  -e UV_CACHE_DIR=/hf_cache/uv \
  -e TMPDIR=/tmp \
  -w /workspace \
  "$VERL_IMAGE" \
  /bin/bash
