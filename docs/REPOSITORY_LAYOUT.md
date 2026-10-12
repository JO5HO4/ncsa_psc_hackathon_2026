# Repository layout and artifact retention

This repository contains the code, datasets, reproducibility metadata, and
small final results for the ROOT Dataset and ROOT + TRExFitter Dataset. Keep
mutable runtime state outside the repository so that source review and
experiment reproduction remain manageable.

## Ownership

| Location | Purpose | Retention |
| --- | --- | --- |
| `data/` | Versioned dataset definitions, fixtures, and dataset submodules | Durable |
| `training/` | SFT/RL dataset preparation, agent loops, rewards, configs, and launch helpers | Durable source |
| `inference/` | Generation, ROOT verification, diagnostics, and plotting code | Durable source |
| `training/scripts/` and `inference/*.sbatch` | Perlmutter job entrypoints | Durable source; prefer parameterized launchers over condition-specific copies |
| `artifacts/<study>/` | Final JSONL completions, verification reports, plots, compact adapters, manifests, and concise logs | Durable results |
| `/pscratch/sd/j/joshuaho/<study>/` | Caches, container layers, active checkpoints, resume JSONLs, executor workspaces, and temporary logs | Mutable runtime state |
| `/global/cfs/cdirs/atlas/joshua/model-hubs/` | Independent Hugging Face publishing/check-out repositories | Separate Git repositories |

`verl/` is an upstream submodule. Treat it as frozen for study work: adapt it
from `training/` and `inference/`, then record any intentional submodule bump
as an isolated change.

## Study artifact contract

Each new study should use one stable directory under `artifacts/`, for example
`artifacts/inference/root-execution-one-shot-YYYYMMDD/` or
`artifacts/rl/root-execution-repair-rl-YYYYMMDD/`. Its durable handoff should
contain:

- a configuration or manifest recording model, adapter, prompt, seed, and
  decoding settings;
- validated completion JSONL files (metadata plus unique prompt IDs);
- ROOT 6.40.02 verification reports and diagnostics;
- analysis tables and plots;
- compact, load-tested LoRA adapters when a model is produced.

Do not retain Hugging Face caches, container overlay data, temporary executor
workspaces, or full resumable FSDP checkpoints in `artifacts/` after their
handoff is validated. Full checkpoints belong on scratch while they are needed
for resume. A compact adapter is sufficient for inference, publishing, and as
the starting policy for a new LoRA RL run.

## Source organization target

Keep new code grouped by responsibility rather than by one-off experiment:

```text
training/
  configs/                 # declarative SFT/RL settings
  rewards/                 # reward functions and contracts
  scripts/                 # reusable container and scheduler entrypoints
  <domain>_agent_loop.py   # domain-specific rollout loops

inference/
  prompts/                 # canonical prompt assets
  run_*.sh / run_prompts.py
  verify_*.sh / score_*.py
  diagnose_*.py / plot_*.py
```

The next structural pass should consolidate condition-specific Slurm files into
parameterized launchers only after completed jobs no longer refer to them. Do
not move source used by an active job.

## Release and archive checklist

1. Validate a final adapter by loading it with its declared base model.
2. Copy the adapter and tokenizer/chat-template files to the model-hub
   repository; verify hashes after push.
3. Retain the study manifest, final completions, verifier report, and plots.
4. Keep a resumable FSDP checkpoint on scratch only while continuation is a
   real requirement. Otherwise archive or remove it after the adapter check.
5. Keep nested external repositories outside this worktree so their Git state
   cannot be mistaken for untracked project code.
