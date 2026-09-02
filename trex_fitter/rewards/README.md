# Rewards

This folder is where we will turn one completed TRExFitter run into one
training reward. It belongs next to the runner, not in `training/`: verl only
needs a final number, while this code knows how to read TRExFitter output.

The model never calls this code. The evaluator applies the model's patch, runs
TRExFitter, gathers the artifacts, and then writes `reward.json` next to that
evaluation's logs.

## Input and output

The future scorer receives:

1. The evaluator result: whether the command finished, its return code, and
   parsed significance.
2. A private copy of the artifacts from that same run: fit logs, workspace,
   fit results, uncertainty breakdown, pulls, correlations, and plots when
   available.
3. The task's physics goal and allowed-change limits.

It writes one `reward.json` using
[`reward.schema.json`](reward.schema.json). The stable parts are:

- `total`: the single number sent to RL.
- `components`: separate, explainable parts of the score.
- `metrics`: raw numbers read from TRExFitter; these are for review and later
  reward design, not necessarily all used for training.
- `artifacts`: paths to the files from which the metrics were read.

## First component names

These names are reserved now. Their weights and exact definitions are still
hackathon tasks.

| Component | What it should measure |
| --- | --- |
| `execution` | Did the config pass checks and complete the requested run? |
| `fit_health` | Did minimization converge with a healthy fit status? |
| `physics_goal` | Did the result meet the task's stated physics goal? |
| `stability` | Are uncertainties, pulls, and correlations reasonable? |
| `complexity` | Did the patch avoid unnecessary config changes? |

## Required runner work

Before implementing a scorer, update the evaluator so that each run owns a
private artifact directory. The H→γγ baseline currently writes a shared
`hyy/` output folder, which is fine for a manual check but unsafe for parallel
RL evaluations. The scorer must never read artifacts from another run.

Do not choose component weights or train with this format until known-good and
known-bad configs have been reviewed.
