# Training with verl

This directory contains the runnable training side of the hackathon. It uses
[verl](https://github.com/volcengine/verl) without modifying verl itself.

## What is ready now

`scripts/run_verl_sft.sh` runs SFT on the included `hep-config-sft` reference
dataset by default. That dataset already has the `messages` and `tools` fields
expected by verl; `verl_dataset.py` decodes its stored tool definitions.

Hackathon datasets should use that same typed chat shape. Author one canonical
verified config task, render it for `codex`, `opencode`, and `direct_config`,
and retain `logical_task_id`, `modality`, `harness`, dataset family, fixture
revision, and verification metadata alongside `messages` and `tools`. A direct
record has `harness: none`, an empty `tools` list, and one assistant response
containing only the config snippet. Do not split or report the three renderings
as independent logical tasks.

The SFT script is a useful baseline and a template for the dataset that the
hackathon creates. It writes checkpoints under `artifacts/checkpoints/sft/`.
After training, it converts the final checkpoint to a `huggingface/` directory
that can be passed directly (or via its parent checkpoint directory) to the
scripts in [`inference/`](../inference/README.md).

Both launchers use Qwen3.5. The RL launcher still requires a separately
prepared RL task parquet and reward function based on TReX runner artifacts.

## ROOT runtime

On Perlmutter, `training/scripts/root_runtime.sh` starts the lightweight CVMFS
ROOT 6.34.02 environment without using its incompatible Python executable.
Use it for ROOT commands or for Python tools that need ROOT available:

```bash
training/scripts/root_runtime.sh root --version
training/scripts/root_runtime.sh --python data/datasets/atlas-open-data-sft-dataset/tools/check-data/validate_dataset.py
```

## ROOT task dataset

[`data/datasets/root-sft-dataset/`](../data/datasets/root-sft-dataset/) is an
HF dataset submodule containing the versioned `root.jsonl` source records.
Initialize it with:

```bash
git submodule update --init --recursive data/datasets/root-sft-dataset
```

It has fixed train, validation, and held-out test splits. Render its SFT rows
into the direct verl format with:

```bash
python3 training/prepare_root_sft.py \
  --source data/datasets/root-sft-dataset/root.jsonl \
  --output-dir data/datasets/root-sft-dataset/sft
```

This creates `sft/train.parquet` and `sft/validation.parquet` with the
`messages` and `tools` fields expected by the local verl adapter. Do not use
the held-out test records for training.

Run SFT with the rendered files:

```bash
TRAIN_FILE=/workspace/data/datasets/root-sft-dataset/sft/train.parquet \
VAL_FILE=/workspace/data/datasets/root-sft-dataset/sft/validation.parquet \
bash training/scripts/run_verl_sft.sh
```

## Run Qwen3.5 on a GPU node

Clone the repository with its submodules, or initialize them in an existing
clone:

```bash
bash training/bootstrap_verl.sh
bash data/fetch_reference_datasets.sh
```

From the repository root in a GPU session, start the supplied verl container:

```bash
bash training/scripts/container.sh
```

The launcher pins a Qwen3.5-capable verl image by digest. Its first start pulls
about 12 GB. To use a locally mirrored or tested image instead, set
`VERL_IMAGE` before invoking the launcher.

Inside the container, prepare the local verl checkout. This creates a
node-local uv environment from the image's warmed dependency cache, then
prints the active Torch, Transformers, SGLang, CUDA, and Qwen3.5 support
status. It fails early if the image cannot load Qwen3.5:

```bash
source training/scripts/setup.sh
```

Model downloads are cached at `/hf_cache`, a bind mount to
`$PSCRATCH/qwen35-hf-cache` (or `$SCRATCH` when `PSCRATCH` is unavailable).
This keeps a 19 GB checkpoint outside both the container writable layer and
Perlmutter's RAM-backed `/tmp`. An interrupted download resumes across
container and node restarts. Set `HF_CACHE_HOST=/path/on/disk` before
`scripts/container.sh` to override the host cache location.

The launcher disables the optional Xet transfer client (`HF_HUB_DISABLE_XET=1`)
because it can terminate Podman-HPC containers during large checkpoint
downloads. Downloads use resumable standard HTTP instead.

### SFT

Run a one-epoch 0.8B smoke test (the default profile):

```bash
QWEN35_MODEL_SIZE=0.8b \
TOTAL_EPOCHS=1 \
TRAIN_BATCH_SIZE=8 \
MICRO_BATCH_SIZE_PER_GPU=1 \
MAX_LENGTH=4096 \
MAX_TOKEN_LEN_PER_GPU=8192 \
RESUME_MODE=disable \
SAVE_DIR=/workspace/artifacts/checkpoints/sft-smoke \
bash training/scripts/run_verl_sft.sh
```

This trains `Qwen/Qwen3.5-0.8B` on the linked `hep-config-sft` dataset and
writes its checkpoint to
`artifacts/checkpoints/sft-smoke/`.

For `Qwen/Qwen3.5-9B`, request one four-GPU Perlmutter node and use the 9B
profile. Its default `NPROC_PER_NODE` is four; retain that value unless the
allocation intentionally differs:

```bash
QWEN35_MODEL_SIZE=9b \
TRAIN_FILE=/workspace/data/datasets/root-sft-dataset/sft/train.parquet \
VAL_FILE=/workspace/data/datasets/root-sft-dataset/sft/validation.parquet \
SAVE_DIR=/workspace/artifacts/checkpoints/qwen35-9b-sft \
bash training/scripts/run_verl_sft.sh
```

`QWEN35_MODEL_SIZE` accepts only `0.8b` and `9b`; it selects the matching
official post-trained model and resource defaults. `MODEL_PATH` may override
the resolved path for a pinned model revision or local snapshot. Both models
are multimodal, but the current SFT records are text-only, so no image fields
are required. The launchers use BF16 by default.

`TRAIN_FILE` and `VAL_FILE` must be absolute paths once inside the container,
or paths relative to the local `verl/` checkout.

### RL

RL uses the same profile interface and SGLang rollout runtime. It is ready to
launch only after a verl-format RL parquet and a Python reward module exist;
the ROOT SFT parquet is not an RL input. For a bounded 0.8B test, supply those
three artifacts explicitly:

```bash
QWEN35_MODEL_SIZE=0.8b \
TRAIN_FILE=/workspace/path/to/rl-train.parquet \
VAL_FILE=/workspace/path/to/rl-validation.parquet \
REWARD_FUNCTION=/workspace/path/to/reward.py \
TOTAL_TRAINING_STEPS=1 \
SAVE_DIR=/workspace/artifacts/checkpoints/qwen35-0.8b-rl-smoke \
bash training/scripts/run_verl_rl.sh
```

Use `QWEN35_MODEL_SIZE=9b` for the four-GPU 9B profile. Start with bounded
response lengths and inspect the smoke-run generations before enlarging a
Qwen3.5 RL experiment; the 0.8B model can otherwise overthink or loop.

## Files

- `bootstrap_verl.sh` initializes the linked verl repository for an existing
  clone.
- `scripts/container.sh` opens the GPU-enabled verl container with this repository at
  `/workspace`.
- `scripts/setup.sh` installs that local checkout and verifies Qwen3.5 support.
- `scripts/qwen35_profile.sh` selects the validated `0.8b` or `9b` model
  profile shared by SFT and RL.
- `scripts/run_verl_sft.sh` runs Qwen3.5 SFT.
- `scripts/run_verl_rl.sh` runs Qwen3.5 RL when `TRAIN_FILE`, `VAL_FILE`, and
  `REWARD_FUNCTION` are supplied.
