# Translator: model-specific + task-specific SFT rendering

This generalizes the "renderer" idea from
[docs/DATASET_DESIGN.md](../../docs/DATASET_DESIGN.md)/[TOOL_CONTRACT.md](../../docs/TOOL_CONTRACT.md)
(there scoped to Codex/OpenCode tool-use rendering) to plain chat-model SFT
targets. It exists because getting a dataset's `messages`/`tools`/
`enable_thinking` shape right for a specific model turned out to matter a
lot: switching the ROOT-command dataset to an explicit output contract plus
the correct `tools`/`enable_thinking` values for Qwen3.5 took validation
execution success from 39.6% to 98.2%, with the *same* questions and answers.
That fix was two independent, reusable pieces of knowledge, previously
hardcoded together in one script. This package splits them and gives each a
registry, so adding a new dataset or a new target model is additive, not a
rediscovery.

## The two axes

- **`ModelProfile`** (`model_profiles.py`) — target-model rendering quirks:
  what to put in the `tools` and `enable_thinking` columns so the model's own
  chat template renders the intended branch. Nothing here is task content.
  Ships `"qwen3.5"`.
- **`TaskProfile`** (`task_profiles.py`) — a dataset family's required output
  contract: the system prompt that states it, and `validate_answer(text)`
  that checks a candidate answer against it. Nothing here is model-specific.
  Ships `"root_command"` (the ROOT one-line command wrapper), migrated
  verbatim from `training/prepare_qwen_root_command_sft.py`.

`translate_record(source, task, model)` in `translate.py` is the pure
function that combines them into one training row. It never touches a
tokenizer or a real model — that stays owned by `verl` and
`training/verl_dataset.py`, which already handle chat-template application
generically (confirmed by reading `verl/verl/utils/tokenizer/chat_template.py`).

## Authoring a new dataset

Write a JSONL file of:

```json
{"id": "unique-id", "question": "...", "answer": "...", "category": "optional-tag"}
```

No message-wrapping, `tools` field, or model awareness needed — that's the
whole point. Then:

```bash
python -m training.translator.cli \
  --source path/to/questions.jsonl --source-format simple-qa \
  --task-profile root_command --model-profile qwen3.5 \
  --split train --output-dir data/derived/my-new-dataset
```

This reads the file, validates it (`validate.py`'s `validate_source_records`
— catches duplicate/empty ids or questions before anything is written),
translates every record, re-validates every output row against the task's
contract (`validate_output_rows`), and writes `<split>.jsonl`,
`<split>.parquet`, and `<split>.manifest.json`. Run
`python -m training.translator.validate check-output <split>.jsonl
--task-profile root_command` any time afterward to re-check a release.

## Adding a new `TaskProfile`

Write the exact output contract (a system prompt) and a `validate_answer`
function, register it in `TASK_PROFILES`. See `docs/TOOL_CONTRACT.md`'s
`direct_config` modality for what a TRExFitter config-snippet contract would
need to say (config snippet only, no prose/diff fences) — a natural
fast-follow once that contract is settled, not built speculatively here.

## Adding a new `ModelProfile`

Read the target model's own chat-template/model-card documentation for its
quirks (does it branch on `tools`? does it support a thinking toggle? can it
hold a standalone system message, or does the calling code need to fold one
in?) *before* writing the profile — the Qwen3.5 profile here reflects
officially documented behavior, not guesswork. Add the entry to
`MODEL_PROFILES`; nothing else needs to change.

## Migration note

`training/prepare_qwen_root_command_sft.py` and
`training/build_qwen_root_command_parquet.py` are retired in favor of this
package. `test_regression_root_command_chat_v1.py` proves the replacement:
running `cli.run(..., task_name="root_command", model_name="qwen3.5")`
against the real `atlas-open-data-sft-dataset` source reproduces
`data/derived/qwen-root-command-chat-v1/{train,validation}.jsonl`
byte-for-byte and their `.parquet` files row-for-row. That committed dataset
itself is untouched by this change.

One deliberate improvement over the retired scripts: their manifests recorded
*absolute* machine paths (e.g. an NERSC `/global/cfs/cdirs/...` path), which
is why that path shows up baked into `data/derived/qwen-root-command-chat-v1/
manifest.json` today. This package's manifests record repository-relative
paths instead, so they stay reproducible across machines.
