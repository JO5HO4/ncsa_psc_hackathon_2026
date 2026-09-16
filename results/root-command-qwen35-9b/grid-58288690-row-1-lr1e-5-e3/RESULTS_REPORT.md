# ROOT-command Qwen3.5-9B results

Both models answer the same 560 validation prompts with the same decoding protocol.

| Model | Passed | Total | Success rate |
|---|---:|---:|---:|
| base | 0 | 560 | 0.00% |
| lora | 204 | 560 | 36.43% |

![Training loss](analysis/training-loss.png)

## Artifacts

- `configuration.txt`: launcher parameters; `training-settings.json`: training settings.
- `base/` and `lora/`: prompts, decoding protocol, completions, and inference timing.
- `analysis/base-score/` and `analysis/lora-score/`: ROOT execution records and JSON/CSV reports, including results by API family.
- `checkpoints/`: saved training states (recovery runs reference the original checkpoint).
- `job.log`, stage logs/status JSON, `score-*.log`, and `status.txt`: execution and errors.

Use validation questions to select changes; reserve test scores for reporting.
