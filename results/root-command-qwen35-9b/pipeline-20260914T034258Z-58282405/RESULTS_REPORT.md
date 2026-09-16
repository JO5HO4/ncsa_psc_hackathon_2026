# ROOT-command Qwen3.5-9B results

Both models answer the same 560 held-out test prompts with the same decoding protocol.

| Model | Passed | Total | Success rate |
|---|---:|---:|---:|
| base | 0 | 560 | 0.00% |
| lora | 116 | 560 | 20.71% |

Recovered checkpoint: `/workspace/results/root-command-qwen35-9b/lr5e6-epoch3/checkpoints/global_step_840`. No training was performed in this run.
The configuration file records launcher settings; consult the source training log for actual recovered-model hyperparameters.

![Training loss](analysis/training-loss.png)

## Artifacts

- `configuration.txt`: launcher parameters; `training-settings.json`: training settings.
- `base/` and `lora/`: prompts, decoding protocol, completions, and inference timing.
- `analysis/base-score/` and `analysis/lora-score/`: ROOT execution records and JSON/CSV reports, including results by API family.
- `checkpoints/`: saved training states (recovery runs reference the original checkpoint).
- `job.log`, stage logs/status JSON, `score-*.log`, and `status.txt`: execution and errors.

Use validation questions to select changes; reserve test scores for reporting.
