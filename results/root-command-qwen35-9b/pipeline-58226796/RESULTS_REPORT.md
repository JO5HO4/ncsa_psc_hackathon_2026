# Qwen3.5-9B ROOT-command LoRA: training and paired evaluation

## Outcome

The base and trained models were evaluated on the same 560 held-out prompts with identical deterministic generation settings. The base model passed **0/560 (0.00%)** executable ROOT checks; the 10-epoch LoRA model passed **142/560 (25.36%)**, an absolute improvement of **25.36 percentage points**.

| Configuration | Nonempty answers | ROOT execution passes | Accuracy | Generation time |
| --- | ---: | ---: | ---: | ---: |
| Base `Qwen/Qwen3.5-9B` | 560/560 | 0/560 | 0.00% | 18.1 min |
| Step-2800 LoRA adapter | 560/560 | 142/560 | 25.36% | 21.7 min |

The base score is dominated by interface noncompliance: all 560 answers failed the required single-command wrapper format. LoRA learned that interface for 292 answers; 142 of all 560 then executed and matched the expected result. Nonempty completion count is an inference-integrity check, not an accuracy metric.

## Dataset and evaluation protocol

- Dataset: pinned `atlas-open-data-sft-dataset`; 4,480 train, 560 validation, and 560 test records.
- The three ID sets were checked as pairwise disjoint.
- Both models used the same saved test prompts, verified byte-for-byte.
- Base model: `Qwen/Qwen3.5-9B`, pinned Hugging Face snapshot `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Generation: greedy (`temperature=0`), thinking disabled, maximum 256 new tokens, and the same system prompt.
- Scoring: execution with ROOT 6.34.02 in fresh temporary working directories, requiring one `RESULT=` line matching the committed expected result.
- Test records were excluded from training and validation. Test performance was not used for checkpoint selection.

## Training configuration

| Hyperparameter | Value |
| --- | --- |
| Method | Supervised fine-tuning with LoRA |
| Epochs / optimizer steps | 10 / 2,800 (280 per epoch) |
| GPUs | 4 × NVIDIA A100 |
| Precision | BF16 |
| Optimizer / schedule | AdamW / constant learning rate |
| Learning rate / weight decay | `1e-5` / `0.01` |
| Global batch / microbatch | 16 / 1 per GPU |
| Maximum sequence length | 2,048 tokens (`truncation=error`) |
| Dynamic token budget | 8,192 tokens per GPU |
| LoRA rank / alpha | 16 / 16 |
| LoRA targets | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Seed | 1 |
| Checkpoint | `global_step_2800`; newest epoch retained |

## Loss curve

![Training loss](analysis/training-loss.png)

The per-step cross-entropy loss fell from **0.9493** at step 1 to **0.000477** at step 2,800. Its final 50-step rolling mean was **0.007137**; final validation loss was **0.011772**. Dashed red lines mark epoch boundaries. The rapid approach to a very low training loss indicates strong fitting of this structured command corpus; it should not be interpreted alone as general ROOT competence, hence the separate held-out execution score.

## Held-out accuracy by API family

![Accuracy by API family](analysis/accuracy-by-api-family.png)

| API family | Passed | Accuracy |
| --- | ---: | ---: |
| `cling` | 0/10 | 0.0% |
| `collections` | 4/20 | 20.0% |
| `directory_and_keys` | 20/40 | 50.0% |
| `fitting` | 10/50 | 20.0% |
| `geometry` | 2/10 | 20.0% |
| `graphics` | 0/20 | 0.0% |
| `graphs` | 6/20 | 30.0% |
| `histogram` | 8/90 | 8.9% |
| `linear_algebra` | 10/10 | 100.0% |
| `mathematics` | 10/10 | 100.0% |
| `multithreading` | 10/10 | 100.0% |
| `object_introspection` | 0/10 | 0.0% |
| `physics_vectors` | 0/10 | 0.0% |
| `random_numbers` | 10/10 | 100.0% |
| `rdataframe` | 2/60 | 3.3% |
| `roofit` | 10/10 | 100.0% |
| `system_utilities` | 0/10 | 0.0% |
| `tchain` | 0/40 | 0.0% |
| `tfile` | 12/40 | 30.0% |
| `timing` | 10/10 | 100.0% |
| `ttree` | 8/60 | 13.3% |
| `vectorized_operations` | 10/10 | 100.0% |

## Post-training failure breakdown

| Outcome/reason | Count |
| --- | ---: |
| `invalid_command_format` | 268 |
| `passed` | 142 |
| `nonzero_returncode` | 78 |
| `root_error_diagnostic` | 48 |
| `result_mismatch` | 24 |

`invalid_command_format` means the answer could not enter the ROOT execution gate. `nonzero_returncode`, `root_error_diagnostic`, and `result_mismatch` passed the command-format gate but failed execution or expected-output comparison.

## Artifacts

- Base completions: `base/completions.jsonl`
- LoRA completions: `lora/completions.jsonl`
- Exported adapter: `checkpoints/global_step_2800/huggingface/lora_adapter/`
- Per-step loss data: `analysis/training-loss.csv`
- Full executed-answer reports: `analysis/base-score/` and `analysis/lora-score/`
- Reproducible training settings: `training-settings.json`

The recovery shell initially ended with exit 127 after inference because its last assertion invoked an unavailable `python` command. The saved inference artifacts were independently validated as 560 unique, nonempty completions with identical prompt inputs. The wrapper has been corrected to use the pinned `uv` Python environment; the model inference itself did not fail and was not rerun.
