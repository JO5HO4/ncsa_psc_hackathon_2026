# Paired Qwen3.5-9B ROOT-command inference

## All three stages in one allocation

For baseline inference, LoRA training/export, and post-training inference in
one allocation, use the four-GPU training profile:

```bash
mkdir -p results/slurm
sbatch training/root_before_after.sbatch
```

This requests one four-A100 node in `regular` QOS for up to six hours (a
configurable wall limit, not a measured runtime estimate). It uses one GPU
for each inference stage and all four for training. Default training is one
epoch, LoRA rank/alpha 16, learning rate 1e-5, batch size 16, microbatch 1 per
GPU, and maximum sequence length 2048 with truncation errors enabled.
For three epochs, explicitly submit:

```bash
TOTAL_EPOCHS=3 sbatch training/root_before_after.sbatch
```

The order is fixed: 560 baseline questions, SFT on 4,480 training examples
with the separate 560-example validation file, checkpoint export, then the
same 560 test prompts with the adapter. Training uses the exact baseline
model snapshot. No test answers are passed to training; no test results are
used for checkpoint selection. Only the latest epoch checkpoint is retained.

All experiment outputs live in
`results/root-command-qwen35-9b/pipeline-JOBID/`: `base/`, `lora/`, `data/`,
`checkpoints/`, `training-settings.json`, stage logs/status files, and the
final `pipeline-summary.json`. Stage failure stops subsequent work and
preserves earlier outputs. Resubmitting creates a new run, not an automatic
resume. Shared model/dependency caches and temporary files remain on scratch.

This script does not perform ROOT execution scoring. GPU training and the
combined pipeline still need end-to-end validation; only the earlier base
inference smoke has been GPU-tested. The one-GPU scripts below remain useful
for independent evaluations.

Submit from this repository root. Each job requests one A100 GPU via the
Perlmutter GPU/shared queue, 32 CPUs, and a two-hour wall limit. The pinned
container and scratch-backed model cache follow the successful smoke test.
No training or execution of generated commands occurs in these jobs.

## Base model

```bash
mkdir -p results/slurm
sbatch inference/perlmutter_root_eval.sbatch base
```

Outputs appear in `results/root-command-qwen35-9b/base-JOBID/`.
This evaluates all 560 held-out questions, with thinking disabled, greedy
generation, and 256 maximum new tokens. It saves prompt-only inputs, the exact
base snapshot revision, dataset and inference-code hashes, raw completions,
an inference summary, logs, and exit status. Generation settings deliberately
match the initial smoke test. Nonempty completions do not imply correctness.

## LoRA model, after training

```bash
sbatch inference/perlmutter_root_eval.sbatch lora \
  results/root-command-qwen35-9b/base-BASE_JOBID \
  /absolute/path/to/huggingface/lora_adapter
```

Replace the baseline job ID and adapter path. Pass the directory containing
`adapter_config.json` and adapter weights, not the raw FSDP checkpoint.
Direct PEFT exports may put those files in `huggingface/` itself. The adapter
configuration must declare `Qwen/Qwen3.5-9B` as its base model. Train using the
same base snapshot recorded in the baseline protocol, and exclude the test
split from training and checkpoint selection.

The baseline and adapter directories are mounted read-only at `/baseline`
and `/adapter`. The LoRA evaluation reuses the baseline prompts, model revision,
and generation settings, and rejects changed inference code. Its outputs go to
`results/root-command-qwen35-9b/lora-JOBID/`. Keep both completed directories.

Inspect `status.txt` (exit code), `inference-summary.json`, and
`completions.jsonl`. Early scheduler/container failures may only appear in
`results/slurm/`. Model downloads, dependency caches, and temporary runtime
files use scratch; experiment outputs and logs use `results/`.

## Scoring

ROOT scoring is a separate step using `inference/score_atlas_benchmark.sh`.
Run both outputs under the same ROOT release and dataset scorer revision, in
a disposable environment appropriate for executing model-generated commands.
Do not interpret the inference summary as a ROOT correctness score. GPU LoRA
evaluation has not yet been validated; the base smoke succeeded on one A100
40 GB, but adapter merging or longer generations can require more headroom.
