# Inference

`run_prompts.py` is the general inference entry point. It accepts any causal
language model available through Hugging Face, a local Hugging Face model
directory, or an exported checkpoint from this repository. It takes a list of
prompts and writes the raw completions. It does not apply patches, validate
configs, or run TRExFitter.

`run_tasks.py` remains a small optional wrapper for the later TRExFitter repair
task schema. It turns those structured records into prompts, but uses the same
model runtime.

## Checkpoint format

The SFT launcher saves verl's resumable state, then automatically converts its
final FSDP checkpoint to an inference-ready Hugging Face export. This is
important for the default LoRA training setup.

```text
artifacts/checkpoints/my-run/global_step_<N>/
└── huggingface/
    ├── config.json
    ├── model.safetensors
    └── tokenizer files
```

Pass either `global_step_<N>` or its `huggingface/` child to `--checkpoint`.
The scripts resolve the latter automatically. If the export includes a LoRA
adapter, the runtime loads and merges it before generation.

For a checkpoint made before this launcher update, create the export once:

```bash
bash inference/export_verl_checkpoint.sh \
  artifacts/checkpoints/sft-smoke/global_step_300
```

## Run on a GPU node

Use the same container session as training so the required PyTorch and
Transformers packages are already available:

```bash
bash training/container.sh
source training/setup.sh
```

## General prompt inference

Create a prompt file. JSON accepts a list of strings or records with `id` and
`prompt`; JSONL accepts one of those values per line:

```json
[
  {"id": "hello", "prompt": "Explain what a likelihood fit does in one sentence."},
  {"id": "config", "prompt": "Write a one-line TRExFitter config comment."}
]
```

Run a base model directly from the Hub:

```bash
python inference/run_prompts.py \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --prompts prompts.json \
  --output /workspace/artifacts/inference/base-qwen.jsonl \
  --format chat \
  --device cuda
```

For ordinary base language models, omit `--format chat`; this feeds the prompt
as literal text. For instruction-tuned models, `--format chat` uses the
model's Hugging Face chat template. Output JSONL has one metadata record then
one completion record per input prompt, preserving its `id`, prompt, raw output,
and generation time.

## Hugging Face Dataset input

The same runner can read prompts from a Hub dataset. The dataset only needs a
string prompt column; it does not need to use a TRExFitter-specific schema.
For example, a dataset whose `test` split has `instruction` and `task_id`
columns runs as:

```bash
python inference/run_prompts.py \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --dataset ho22joshua/my-prompt-dataset \
  --split test \
  --prompt-field instruction \
  --id-field task_id \
  --output /workspace/artifacts/inference/my-prompt-dataset.jsonl \
  --format chat \
  --device cuda
```

Use `--dataset-config NAME` for a configured dataset, `--revision COMMIT_OR_TAG`
for a reproducible dataset version, and `--streaming` for a large split. The
output metadata records all of those choices.

## This repository's trained checkpoints

First verify that a selected checkpoint loads and generates text:

```bash
python inference/smoke_test.py \
  --checkpoint /workspace/artifacts/checkpoints/sft-smoke/global_step_<N> \
  --device cuda
```

Use the same generic command with the `global_step_N` checkpoint path:

```bash
python inference/run_prompts.py \
  --model /workspace/artifacts/checkpoints/sft-smoke/global_step_<N> \
  --prompts prompts.json \
  --output /workspace/artifacts/inference/sft-prompts.jsonl \
  --format chat \
  --device cuda
```

For structured config tasks, run the optional wrapper. A `.json` file can hold
one task or a JSON list; `.jsonl` holds one task per line. The provided example
is schema-only, so use it only to check the interface, not as a physics result.

```bash
python inference/run_tasks.py \
  --checkpoint /workspace/artifacts/checkpoints/sft-smoke/global_step_<N> \
  --tasks data/trex_config/examples/repair-task.example.json \
  --output /workspace/artifacts/inference/example.jsonl \
  --device cuda \
  --temperature 0
```

Each output JSONL begins with one metadata record, followed by prediction
records containing the raw answer, an extracted `<patch>...</patch>` diff when
present, and generation time. Feed those records to the future evaluator; do
not give the model access to that evaluator or to TRExFitter.

Set `EXPORT_FOR_INFERENCE=false` when launching training only if you explicitly
want to skip the final conversion. In that case run
`export_verl_checkpoint.sh` before inference.
