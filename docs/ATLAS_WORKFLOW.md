# ATLAS Open Data workflow

This guide reproduces the Qwen command benchmark using the public ATLAS Open
Data SFT dataset. It requires a Perlmutter GPU allocation for inference and a
ROOT-enabled Perlmutter environment for command evaluation.

## Clone

HTTPS is the default and works for public read-only clones:

```bash
git clone --recurse-submodules https://github.com/JO5HO4/ncsa_psc_hackathon_2026.git
cd ncsa_psc_hackathon_2026
```

To use SSH for the Hugging Face dataset submodules after cloning, override only
your local configuration, then initialize them:

```bash
git config submodule.data/datasets/atlas-open-data-sft-dataset.url git@hf.co:datasets/ho22joshua/atlas-open-data-sft-dataset
git config submodule.data/datasets/root-sft-dataset.url git@hf.co:datasets/ho22joshua/root-sft-dataset
git submodule update --init --recursive
```

Validate the released dataset:

```bash
training/scripts/root_runtime.sh --python data/datasets/atlas-open-data-sft-dataset/tools/check-data/validate_dataset.py
```

## Generate held-out completions

On a GPU node, start the pinned verl environment:

```bash
bash training/scripts/container.sh
source training/scripts/setup.sh
```

Inside that container, export only the public held-out prompts and run Qwen:

```bash
python /workspace/data/datasets/atlas-open-data-sft-dataset/tools/make-data/export_parquet_prompts.py \
  --input /workspace/data/datasets/atlas-open-data-sft-dataset/data/sft/test.parquet \
  --output /workspace/artifacts/inference/atlas-test-prompts.jsonl

uv run --project /workspace/verl --no-sync python /workspace/inference/run_prompts.py \
  --model Qwen/Qwen3.5-0.8B \
  --prompts /workspace/artifacts/inference/atlas-test-prompts.jsonl \
  --prompt-field prompt --id-field id --format chat --no-enable-thinking \
  --system-prompt "Return exactly one executable ROOT command. Do not add explanation." \
  --device cuda --temperature 0 --max-new-tokens 256 \
  --output /workspace/artifacts/inference/atlas-test-qwen35-0.8b.jsonl
```

Use `Qwen/Qwen3.5-9B` for the larger model. Do not train on `test.parquet`.

For a LoRA-trained checkpoint, pass its `global_step_<N>` directory (or its
`huggingface/` child) to `--model`; inference automatically loads the Qwen base
model and applies `lora_adapter/`. To pass `lora_adapter/` directly, also add
`--base-model Qwen/Qwen3.5-0.8B`.

## Score the commands

From a ROOT-enabled shell, evaluate the completion file. This executes each
command and compares its `RESULT=` output with the committed expected result:

```bash
cd data/datasets/atlas-open-data-sft-dataset
root -l -b -q 'benchmark/evaluate_docs_completions.C("/workspace/artifacts/inference/atlas-test-qwen35-0.8b.jsonl", "benchmark/expected-results/test.jsonl", "/workspace/artifacts/inference/atlas-test-qwen35-0.8b-executed.jsonl")'

python tools/benchmark/build_report.py \
  --dataset data/sft/test.parquet \
  --expected benchmark/expected-results/test.jsonl \
  --completion qwen35_0_8b=/workspace/artifacts/inference/atlas-test-qwen35-0.8b-executed.jsonl \
  --output-json /workspace/artifacts/benchmarks/atlas-test-report.json \
  --output-csv /workspace/artifacts/benchmarks/atlas-test-report.csv \
  --summary-csv /workspace/artifacts/benchmarks/atlas-test-summary.csv
```

`tools/benchmark/plot_report.py` can render plots from the summary CSV. SFT
training setup will be documented separately.
