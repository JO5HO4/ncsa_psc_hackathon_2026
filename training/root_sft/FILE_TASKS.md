# Executable ROOT-operation pilot

This adds practical task-selection SFT and deterministic execution evaluation to
the existing Qwen2.5/LoRA/verl workflow. No training framework changes or external
teacher-model calls are required. It is an explicitly limited pilot, not a claim
of parity with a frontier assistant.

## What is included

`data/root_io/file-tasks-v1/` contains 36 training examples, 12 validation tasks,
and 12 test tasks. Each split has equal counts across six operation families:
inventory, selection, jagged collections, weighted histograms/yields,
streaming/schema validation, and writing/reopening outputs.

There are **12 operation templates over five distinct synthetic fixture groups**,
not 60 unrelated skills. Train uses three groups; validation and test each use
one separate group with different branch names, thresholds and IDs. Templates
are intentionally shared: this measures argument/schema transfer, not unseen
algorithms. The small held-out sets are development pilots, not statistically
strong evidence of general ROOT competence. They do not replace the old sealed
knowledge test set.

Each prompt contains a schema, a user request and the operation API contract.
The answer is one JSON operation with arguments. The executor reads actual ROOT
files; it never executes model-produced Python, shell or expression strings.
The model does not choose filesystem paths. Writes use new temporary output
files and inputs are checked for changes. Tests use only trusted synthetic files.

Fixtures include empty collections, exact selection boundaries, signed/zero
weights, underflow/overflow values, repeated metadata cycles, and inconsistent
count branches. Reference outputs come from plain-list computations independent
of the Uproot executor. All 60 reference plans must execute correctly before
the builder writes a manifest, which records file hashes and library versions.

Implementation follows the [Uproot read/write API](https://uproot.readthedocs.io/en/latest/basic.html).
Training is supervised prediction of the requested operation, not yet RL or
multi-turn tool calling. The tools column remains `[]`; contracts are in the prompt.
The existing 12 executable-code Q&As remain separate and are not silently mixed in.

## Build and test

Requires Uproot 5, Awkward 2, NumPy and PyArrow, in addition to the existing setup.

```bash
python training/root_sft/file_tasks.py build --output data/root_io/file-tasks-v1
python -m unittest training.root_sft.test_file_tasks -v
```

The release is already built locally. Builders and evaluators refuse existing
output directories; use new version/run names instead of replacing prior results.

Local verification: all five evaluator tests passed. Tokenizing full conversations
with the saved Qwen2.5 tokenizer gave maximum lengths of 519 tokens for train,
519 for validation and 528 for test, below the existing 2048-token limit.
The actual verl assistant-loss-mask preflight and GPU smoke test are still pending.

## Establish a baseline before training

Inside the GPU container, with dependencies installed, from `/workspace`:

```bash
python inference/run_prompts.py \
  --model /path/to/your/sft/checkpoint/huggingface \
  --prompts data/root_io/file-tasks-v1/validation/prompts.jsonl \
  --output artifacts/root-filetasks-before.jsonl \
  --format chat --device cuda --temperature 0 --max-new-tokens 512 \
  --system-prompt 'Return one JSON object calling the documented ROOT operation. No prose, code fences or predicted results.'

python training/root_sft/file_tasks.py evaluate \
  --release data/root_io/file-tasks-v1 --split validation \
  --predictions artifacts/root-filetasks-before.jsonl \
  --output artifacts/root-filetasks-before-score
```

Also measure the original pinned Qwen snapshot under the same decoding settings.
The evaluator rejects missing/duplicate IDs, altered prompts and modified fixture
releases. It reports pass rate by category. A pass requires successful execution,
correct output, input preservation AND the requested operation/arguments. This
last requirement prevents accidental numerical equality from hiding a wrong plan
on a small fixture, but it means the benchmark does not accept arbitrary
alternative algorithms. Malformed JSON counts as failure, not an ignored example.
Do not interpret oracle/reference tests passing as model performance.

## Training integration, unchanged verl

The generated `train/sft.parquet` and `validation/sft.parquet` already have the
messages/tools schema understood by `training/verl_dataset.py`. Use the
existing generic `training/scripts/run_verl_sft.sh` launcher.

For a future two-step smoke run, after checking tokenizer lengths and assistant
loss masks with the actual model and adapter, the relevant overrides are:

```bash
MODEL_PATH=/path/to/pinned/local/Qwen-snapshot \
TRAIN_FILE=/workspace/data/root_io/file-tasks-v1/train/sft.parquet \
VAL_FILE=/workspace/data/root_io/file-tasks-v1/validation/sft.parquet \
SAVE_DIR=/workspace/artifacts/root-filetasks-smoke/checkpoints \
TRAIN_BATCH_SIZE=2 MICRO_BATCH_SIZE_PER_GPU=1 \
TOTAL_EPOCHS=1 LR=1e-5 RESUME_MODE=disable \
ENGINE_USE_TORCH_COMPILE=false EXPORT_FOR_INFERENCE=false \
bash training/scripts/run_verl_sft.sh \
  data.ignore_input_ids_mismatch=False trainer.total_training_steps=2
```

Replace the model placeholder with a real resolved snapshot and choose a fresh
save directory. No GPU training has been launched for this pack, and tokenization
preflight remains required. Use the existing checkpoint exporter separately;
do not rely on the generic launcher's newline-sensitive automatic export.

## Next expansion, after the pilot

Retain equal task-family quotas, audit token exposure as well as row counts, and
add independent scenarios rather than padding this pack with renamings. Important
missing capabilities include multi-file schema reconciliation, multi-turn
inspection before action, missing-object recovery, ownership in PyROOT,
compression/update policies, remote access and arbitrary code generation.
Those need new verified tasks and an isolated code-execution harness if generated
code is to be evaluated. Additional examples alone cannot guarantee improvement;
compare the same pre/post model on held-out execution tasks before making claims.
