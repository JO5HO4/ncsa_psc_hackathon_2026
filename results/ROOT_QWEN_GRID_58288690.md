# Qwen3.5-9B LoRA validation grid: Slurm 58288690

All three Perlmutter array rows completed with exit code 0 on 2026-09-15 UTC. Each used four NVIDIA A100 40-GB GPUs and ran baseline inference, fresh LoRA SFT, adapter export, deterministic inference, and actual ROOT execution scoring. The selection metric is ROOT execution success on the fixed 560-example **validation** split; no test prompts or test expected results were used in this grid.

## Result

| Row | Learning rate | Epochs | Final checkpoint | Validation passes / 560 | Validation success | Final validation loss |
|---:|---:|---:|---|---:|---:|---:|
| 0 | 5e-6 | 10 | `global_step_2800` | 81 | 14.46% | 0.018905 |
| 1 | 1e-5 | 3 | `global_step_840` | 204 | 36.43% | 0.013972 |
| 2 | 1e-5 | 15 | `global_step_4200` | 222 | 39.64% | 0.013333 |

The selected configuration is **learning rate `1e-5`, 15 epochs**, with 222/560 validation passes. It exceeds the three-epoch run by 18 passes (3.21 percentage points) and the lower-learning-rate ten-epoch run by 141 passes (25.18 percentage points). This gives evidence that, with the fixed batch size and LoRA configuration, `1e-5` trains much more effectively than `5e-6`, and extending `1e-5` from three to fifteen epochs still improves validation execution success.

![Validation-grid comparison](ROOT_QWEN_MODEL_COMPARISON.png)

## Qwen3.5-0.8B comparison

One 0.8B run used the same ROOT-command dataset, the same 560 validation prompts, greedy decoding, the same LoRA rank/alpha, and the same `1e-5` learning rate for 15 epochs. It used one A100 rather than four, as appropriate for the smaller profile.

| Model | Learning rate | Epochs | Validation passes / 560 | Validation success | Final validation loss |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 1e-5 | 15 | 0 | 0.00% | 0.015910 |
| Qwen3.5-9B | 1e-5 | 15 | 222 | 39.64% | 0.013333 |

The 0.8B LoRA model generated 560 unique, nonempty answers but passed none of the executable ROOT checks. Of its 560 outputs, 532 failed the exact command format and 28 reached execution but returned a nonzero code. The result shows that, under this training and decoding setup, reducing the model from 9B to 0.8B removes the benchmark improvement. It is not a valid candidate for test evaluation.

0.8B artifacts: [run directory](root-command-qwen35-0.8b/qwen35-0.8b-lr1e-5-e15-validation/), [run report](root-command-qwen35-0.8b/qwen35-0.8b-lr1e-5-e15-validation/RESULTS_REPORT.md), adapter `root-command-qwen35-0.8b/qwen35-0.8b-lr1e-5-e15-validation/checkpoints/global_step_4200/huggingface/lora_adapter/`, and score report `root-command-qwen35-0.8b/qwen35-0.8b-lr1e-5-e15-validation/analysis/lora-score/lora-report.json`.

This is a single-seed grid. It identifies the best tested configuration, not a statistical optimum. The final validation loss for the fifteen-epoch run was slightly above its lowest observed value, 0.012271 at step 1,960 (epoch 7), while its final checkpoint had the best measured execution score. Loss alone is therefore not a sufficient checkpoint-selection metric for this benchmark.

## Failure outcomes

| Row | Passed | Invalid command format | Nonzero return code | ROOT diagnostic error | Result mismatch |
|---:|---:|---:|---:|---:|---:|
| 5e-6 × 10 | 81 | 332 | 94 | 22 | 31 |
| 1e-5 × 3 | 204 | 176 | 82 | 52 | 46 |
| 1e-5 × 15 | 222 | 150 | 100 | 60 | 28 |

The 15-epoch run reduced format failures from 176 to 150 relative to the three-epoch `1e-5` run and improved matched ROOT executions. It also produced more execution-stage failures; format compliance and executable-command correctness should be inspected separately in follow-up work.

The baseline model passed 0/560 in every grid row, with all answers failing the exact command-wrapper format. All trained models generated 560 unique, nonempty answers. These checks establish generation integrity but are not correctness scores.

## Fixed configuration

- Dataset: 4,480 train, 560 validation, and 560 test ROOT-command examples from `data/datasets/atlas-open-data-sft-dataset/data/sft/`. The benchmark expected-result IDs match the validation prompts one-to-one.
- Base model: `Qwen/Qwen3.5-9B`, pinned revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Training: BF16 FSDP; four GPUs; global batch 16; microbatch 1 per GPU; maximum length 2,048; maximum token budget 8,192 per GPU; LoRA rank/alpha 16/16; same target modules in every row; fresh adapter initialization; AdamW with constant learning rate and weight decay 0.01.
- Inference: greedy decoding, temperature 0, thinking disabled, 256 maximum generated tokens, and byte-identical prompts for base and LoRA within each row.
- Scoring: the repository CVMFS ROOT runtime runs the generated commands and requires an expected `RESULT=` output.

## Artifact locations

| Row | Run directory | Full run report | Adapter | ROOT score report |
|---:|---|---|---|---|
| 0 | [run](root-command-qwen35-9b/grid-58288690-row-0-lr5e-6-e10/) | [RESULTS_REPORT.md](root-command-qwen35-9b/grid-58288690-row-0-lr5e-6-e10/RESULTS_REPORT.md) | `checkpoints/global_step_2800/huggingface/lora_adapter/` | `analysis/lora-score/lora-report.json` |
| 1 | [run](root-command-qwen35-9b/grid-58288690-row-1-lr1e-5-e3/) | [RESULTS_REPORT.md](root-command-qwen35-9b/grid-58288690-row-1-lr1e-5-e3/RESULTS_REPORT.md) | `checkpoints/global_step_840/huggingface/lora_adapter/` | `analysis/lora-score/lora-report.json` |
| 2 | [run](root-command-qwen35-9b/grid-58288690-row-2-lr1e-5-e15/) | [RESULTS_REPORT.md](root-command-qwen35-9b/grid-58288690-row-2-lr1e-5-e15/RESULTS_REPORT.md) | `checkpoints/global_step_4200/huggingface/lora_adapter/` | `analysis/lora-score/lora-report.json` |

Each run directory includes `configuration.txt`, `training-settings.json`, `status.txt`, `job.log`, per-stage logs/status JSON, frozen train/validation data hashes, base and LoRA completions, ROOT execution records, CSV/JSON scores by API family, and a loss plot. Slurm stdout logs are `slurm/root-grid-58288690_0.out`, `_1.out`, and `_2.out`; their `.err` files are empty.

## Recommended next action

Evaluate the selected 15-epoch adapter once on the held-out test set, using the existing recovery-capable pipeline. Do not use its test result to choose another hyperparameter. The validation grid already shows that additional epochs can improve execution success; a later experiment should preserve every epoch adapter or export/evaluate selected intermediate checkpoints on validation, because validation loss reached its minimum around epoch 7 while the measured final execution score was highest at epoch 15.
