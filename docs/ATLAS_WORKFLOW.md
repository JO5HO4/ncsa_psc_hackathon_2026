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

For an SFT export, set `MODEL` to the `huggingface/` directory and use the
single inference entry point:

```bash
MODEL=/workspace/artifacts/my-atlas-run/sft-checkpoint/global_step_5600/huggingface \
  bash /workspace/inference/run_atlas_sft_inference.sh
```

## Score the commands

From a ROOT-enabled shell, evaluate completions and build the report in one
command. This executes each command and compares its `RESULT=` output with the
committed expected result:

```bash
bash inference/score_atlas_benchmark.sh \
  artifacts/atlas-sft-completions.jsonl \
  qwen35-0.8b-sft
```

The wrapper writes the executed completions, JSON report, CSV report, and
summary CSV under `artifacts/benchmarks/qwen35-0.8b-sft/`. Set `ROOT_RUNNER`
to an alternate ROOT launcher or `PYTHON_BIN` to a Python containing PyArrow
when your environment differs. `tools/benchmark/plot_report.py` can render
plots from the summary CSV.

If you have already run `lsetup "root 6.30.02-x86_64-centos7-gcc11-opt"`,
reuse that ROOT environment instead of requesting the CVMFS StatAnalysis
release:

```bash
ROOT_USE_CURRENT=1 bash inference/score_atlas_benchmark.sh \
  artifacts/atlas-open-data-sft/completions/qwen35-0.8b-sft-10epoch.jsonl \
  qwen35-0.8b-sft-10epoch \
  artifacts/atlas-open-data-sft/benchmarks/qwen35-0.8b-sft-10epoch
```
