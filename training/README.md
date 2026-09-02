# Training with verl

This directory contains the runnable training side of the hackathon. It uses
[verl](https://github.com/volcengine/verl) without modifying verl itself.

## What is ready now

`scripts/run_verl_sft.sh` runs SFT on the included `hep-config-sft` reference
dataset by default. That dataset already has the `messages` and `tools` fields
expected by verl; `verl_dataset.py` decodes its stored tool definitions.

Hackathon datasets should use that same typed chat/tool shape. Author one
canonical verified episode, render it separately for `codex` and `opencode`,
and retain `logical_task_id`, `harness`, dataset family, fixture revision, and
verification metadata alongside `messages` and `tools`. Do not split or report
the two harness renderings as independent logical tasks.

The SFT script is a useful baseline and a template for the dataset that the
hackathon creates. It writes checkpoints under `artifacts/checkpoints/sft/`.
After training, it converts the final checkpoint to a `huggingface/` directory
that can be passed directly (or via its parent checkpoint directory) to the
scripts in [`inference/`](../inference/README.md).

The RL launcher is also retained, but is intentionally not runnable yet: it
needs the future RL task parquet and the reward function based on the TReX
runner artifacts.

## Run the SFT baseline on a GPU node

Clone the repository with its submodules, or initialize them in an existing
clone:

```bash
bash training/bootstrap_verl.sh
bash data/fetch_reference_datasets.sh
```

From the repository root in a GPU session, start the supplied verl container:

```bash
bash training/container.sh
```

Inside the container, run this one-epoch baseline:

```bash
source training/setup.sh && \
NPROC_PER_NODE=1 \
TOTAL_EPOCHS=1 \
TRAIN_BATCH_SIZE=8 \
MICRO_BATCH_SIZE_PER_GPU=1 \
MAX_LENGTH=4096 \
MAX_TOKEN_LEN_PER_GPU=8192 \
RESUME_MODE=disable \
SAVE_DIR=/workspace/artifacts/checkpoints/sft-smoke \
bash training/scripts/run_verl_sft.sh
```

This trains `Qwen/Qwen2.5-Coder-1.5B-Instruct` on the linked
`hep-config-sft` dataset and writes its checkpoint to
`artifacts/checkpoints/sft-smoke/`.

For a larger run, change the model, GPU count, output location, or input files
through environment variables:

```bash
MODEL_PATH=Qwen/Qwen2.5-Coder-1.5B-Instruct \
NPROC_PER_NODE=4 \
TRAIN_FILE=/workspace/data/trex_config/splits/train.parquet \
VAL_FILE=/workspace/data/trex_config/splits/validation.parquet \
SAVE_DIR=/workspace/artifacts/checkpoints/my-run \
bash training/scripts/run_verl_sft.sh
```

`TRAIN_FILE` and `VAL_FILE` must be absolute paths once inside the container,
or paths relative to the local `verl/` checkout. The future dataset-preparation
task will create these files from reviewed, verified dataset-family episodes.

## Files

- `bootstrap_verl.sh` initializes the linked verl repository for an existing
  clone.
- `container.sh` opens the GPU-enabled verl container with this repository at
  `/workspace`.
- `setup.sh` installs that local checkout into the container session.
- `scripts/run_verl_sft.sh` runs the ready SFT baseline.
- `scripts/run_verl_rl.sh` is the future RL entry point; it needs
  `TRAIN_FILE`, `VAL_FILE`, and `REWARD_FUNCTION` explicitly.
