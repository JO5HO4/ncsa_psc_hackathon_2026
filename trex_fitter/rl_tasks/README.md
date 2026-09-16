# TRExFitter RL tasks

`v3-histfit-001` is the first replayable, native TRExFitter RL task. It uses
the compact shared histogram inputs and the pinned StatAnalysis 0.8.2
environment (TRExFitter v1.10.0, ROOT 6.40.04).

Prepare an isolated workspace inside this repository:

```bash
python -m trex_fitter.rl_tasks.v3_histfit_001 prepare \
  --workspace artifacts/trex_fitter/rl-workspaces/v3-histfit-001/demo
```

The agent may read `TASK.md`, inspect the read-only inputs, and edit only
`analysis.config`. Score an intermediate state without executing a fit:

```bash
python -m trex_fitter.rl_tasks.v3_histfit_001 evaluate \
  --workspace artifacts/trex_fitter/rl-workspaces/v3-histfit-001/demo
```

Score a terminal candidate with the native `hwfs` chain:

```bash
python -m trex_fitter.rl_tasks.v3_histfit_001 evaluate \
  --workspace artifacts/trex_fitter/rl-workspaces/v3-histfit-001/demo \
  --execute --timeout 1800
```

The evaluator reports structured reward components. The harness, not the task
script, must enforce the editable-path allowlist, network isolation, resource
limits, and maximum agent steps.

For the initial VERL GRPO stage, use `training/prepare_trex_v3_rl.py` and
`training/rewards/trex_v3_histfit_reward.py`. That stage scores a final config
only; a true inspect/run/edit trajectory requires the separate tool-rollout
environment still to be built.
