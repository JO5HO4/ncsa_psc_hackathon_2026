# ROOT-command Qwen3.5-9B campaign

Updated 2026-09-14 UTC. Repository on Perlmutter:
`/global/cfs/cdirs/atlas/dwkim/agentic-ai-hep/ncsa_psc_hackathon_2026`.
Links below are relative to this report, so collaborators can follow them in the shared directory.

## Hyperparameter-grid update: 2026-09-15

Slurm array 58288690 completed all three validation-scored grid rows successfully. The selected setting is **learning rate `1e-5`, 15 epochs**, with **222/560 (39.64%)** ROOT execution passes on validation. The companion report [ROOT_QWEN_GRID_58288690.md](ROOT_QWEN_GRID_58288690.md) gives the complete comparison, failure outcomes, loss behavior, and artifact locations.

This selection uses validation only. Its held-out test score has not yet been measured.

Qwen3.5-0.8B was also trained for 15 epochs at `1e-5` and scored **0/560** validation ROOT passes, despite 560 nonempty outputs. It is included in [ROOT_QWEN_GRID_58288690.md](ROOT_QWEN_GRID_58288690.md) and the combined comparison plot.

The five-run, directly comparable validation summary is [ROOT_QWEN_FIVE_RUNS.md](ROOT_QWEN_FIVE_RUNS.md).

## Completed model evaluations

| Model | Learning rate | Epochs | Final validation loss | Test passed / 560 | Test success |
|---|---:|---:|---:|---:|---:|
| Base Qwen3.5-9B | — | — | — | 0 | 0.00% |
| LoRA A | 1e-5 | 10 | 0.011772 | 142 | 25.36% |
| LoRA B | 5e-6 | 3 | 0.029423 | 116 | 20.71% |

The base model was evaluated twice, with 0/560 both times. Both base and both trained-model inference runs produced 560 unique, nonempty answers. The decoding protocols of the two experiments matched. LoRA B scored 26 fewer passes (4.64 percentage points lower) than LoRA A. These are single runs, not seed-averaged estimates.

The baseline's 560 failures were all `invalid_command_format`. Its zero score therefore measures success under the benchmark's exact executable-command contract, not general ROOT knowledge.

| ROOT scorer outcome | LoRA A: 10 epochs | LoRA B: 3 epochs |
|---|---:|---:|
| Passed | 142 | 116 |
| Invalid command format | 268 | 258 |
| Nonzero return code | 78 | 122 |
| ROOT diagnostic error | 48 | 38 |
| Incorrect result | 24 | 26 |

## Data and fixed settings

- Dataset: `data/datasets/atlas-open-data-sft-dataset/data/sft/`: 4,480 training, 560 validation, 560 test examples across 22 ROOT API families. This is the 5,600 ROOT-command dataset.
- Base model: `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Training: four A100 40-GB GPUs, BF16 FSDP, batch 16, microbatch 1 per GPU, maximum sequence length 2,048, token budget per GPU 8,192. LoRA rank/alpha 16/16, targeting q/k/v/o projections and gate/up/down projections. AdamW, constant learning rate, weight decay 0.01.
- Inference: one visible GPU, greedy decoding (temperature 0), thinking disabled, maximum 256 generated tokens. Base and adapter use identical prompts.
- Inference system prompt: `Return exactly one executable ROOT command. Do not add explanation.`
- Training system prompt: `Return exactly one executable command requested by the user. Do not execute it or add explanation, Markdown, or a guessed result.` This mismatch is a separate candidate for validation experiments.
- ROOT execution scoring uses the dataset's expected results and repository CVMFS ROOT runtime. Nonempty completion counts are not correctness scores.

## Training curves

LoRA A completed 2,800 steps. Mean training losses per epoch were 0.280262, 0.016833, 0.011563, 0.010171, 0.009500, 0.009016, 0.008785, 0.008537, 0.008365, 0.008231.

| LoRA B epoch | Step | Mean training loss | Validation loss |
|---|---:|---:|---:|
| 1 | 280 | 0.504211 | 0.185426 |
| 2 | 560 | 0.103402 | 0.048839 |
| 3 | 840 | 0.035409 | 0.029423 |

LoRA B's validation loss continued falling through epoch 3. This motivates a longer run at the same learning rate but does not establish that ROOT success will improve.

## Files and run history

### LoRA A: learning rate 1e-5, 10 epochs

- [Full report](root-command-qwen35-9b/pipeline-58226796/RESULTS_REPORT.md)
- [Run directory](root-command-qwen35-9b/pipeline-58226796/): baseline, training logs/settings, recovered inference logs, and adapter artifacts.
- [Scores and plots](root-command-qwen35-9b/pipeline-58226796/analysis/): `base-score/`, `lora-score/`, training loss PNG/PDF/CSV, and API-family comparison.
- Adapter: `root-command-qwen35-9b/pipeline-58226796/checkpoints/global_step_2800/huggingface/lora_adapter/`.
- Original Slurm job 58226796 completed training but stopped before adapter export. The checkpoint tracker lacked a final newline; Bash `read` returned 1 under `set -e`. The export was subsequently recovered. Recovery inference under 58271681 produced 560 answers, then the reporting wrapper hit `python: command not found`; completed outputs were retained and scored separately. Both wrapper issues have been fixed.

### LoRA B: learning rate 5e-6, 3 epochs

- [Training log](root-command-qwen35-9b/lr5e6-epoch3/training.log)
- [Checkpoints](root-command-qwen35-9b/lr5e6-epoch3/checkpoints/): steps 280, 560, 840; the final adapter is in `global_step_840/huggingface/lora_adapter/`.
- [Full evaluation report](root-command-qwen35-9b/pipeline-20260914T034258Z-58282405/RESULTS_REPORT.md)
- [Evaluation directory](root-command-qwen35-9b/pipeline-20260914T034258Z-58282405/): `base/` and `lora/` hold completions, prompts, protocols, and timings; `analysis/` holds ROOT execution records, JSON/CSV scores by API family, and loss plots with dashed red epoch boundaries.
- Evaluation pipeline finished with exit code 0 at 2026-09-14 04:32:17 UTC. No retraining occurred during checkpoint recovery.
- [Earlier failed launch](root-command-qwen35-9b/pipeline-20260914T034001Z-58282405/): package installation failed before model evaluation; excluded from the score table. The launcher now loads the Python module and activates the existing `llm_env` instead of installing reporting packages.

### Smoke test

- [10-prompt smoke test](root-command-qwen35-9b-smoke-20260911/), Slurm 58224548: 10/10 nonempty completions, one A100; 54.35 seconds generation. This checked runtime operation, not benchmark accuracy, and is excluded from the score table.

## Next three configurations

No optimal setting can be inferred from two runs that changed both learning rate and epoch count. The following is a controlled exploratory grid, not a claim that any setting is optimal.

| Array row | Learning rate | Epochs | Expected steps | Purpose |
|---|---:|---:|---:|---|
| 0 | 5e-6 | 10 | 2,800 | Test longer training at the lower LR; compare equal epochs to LoRA A. |
| 1 | 1e-5 | 3 | 840 | Complete the LR/epoch cross-comparison; test whether higher LR learns faster. |
| 2 | 1e-5 | 15 | 4,200 | Test whether extending the current best configuration helps. |

All start from the same base checkpoint with fresh adapters, not resumed optimizer states. All other training and inference settings remain fixed. The grid scores **560 validation questions**, including actual ROOT execution. Select by validation execution success; use format failure rate and validation loss as supporting diagnostics. Then evaluate the selected model once on the existing test set. Historical test results have already influenced this exploration, so the test set is no longer an untouched final holdout; an additional unseen set would give a stronger final estimate.

The historical models have test scores, not validation execution scores. Direct claims about improvement over them require evaluating their saved adapters on validation too. Epoch 10's lower loss alone does not establish an optimal stopping point. Retaining the last three checkpoints supports follow-up inspection, but this grid evaluates the final checkpoint of each run automatically.

## Perlmutter submission

Prepared script: [training/root_hyperparameter_grid.sbatch](../training/root_hyperparameter_grid.sbatch). Not submitted as part of report preparation.

```bash
cd /global/cfs/cdirs/atlas/dwkim/agentic-ai-hep/ncsa_psc_hackathon_2026
mkdir -p results/slurm
sbatch training/root_hyperparameter_grid.sbatch
```

Account `atlas_g`, regular GPU queue, one node/four GPUs per task, six-hour time limit per task, array `0-2%1` (one task at a time). Maximum requested GPU walltime is 72 GPU-hours across all three tasks; this is a limit, not a runtime estimate. Models execute locally, without CBORG API usage.

Each task runs environment setup, baseline inference, training/export, trained inference, ROOT scoring, and Markdown/plot generation. Logs: `results/slurm/root-grid-<array-id>_<row>.out/.err`. Artifacts: `results/root-command-qwen35-9b/grid-<array-id>-row-<row>-lr<LR>-e<epochs>/`. Start with each run's `RESULTS_REPORT.md`, `pipeline-summary.json`, and `status.txt`. This campaign report records the current results and proposed grid; new grid scores should be added after completion.
