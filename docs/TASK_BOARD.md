# Task board

Keep each dataset small and useful on its own; we will join them into longer
tasks later.

## Status key for assigned tasks

- ⚪ Not started
- 🟢 Complete
- 🔴 Blocked

Only assigned tasks use these circles. Unassigned tasks stay as empty
checkboxes until someone takes them.

## 1. Make the small datasets

| Owner | Status | Task |
| --- | --- | --- |
| Chengxi | ⚪ | Turn the TRExFitter documentation into H→γγ operational tasks. |
| Joshua | ⚪ | Turn ATLAS Open Data documentation into small tasks. |
| Dongwon | ⚪ | Make ROOT-file tasks: inspect objects and variables, then describe them in plain language. |

- [ ] Choose the first small part of the TRExFitter config language to support.
- [ ] Make simple config tasks: a physics goal, a starting config, a known good answer, and broken examples.
- [ ] Make small tasks for running TRExFitter and reading its results.
- [ ] Keep training, validation, and final test tasks separate from the start.
- [ ] Find a faster way to histogram data: evaluate a new backend and already-histogrammed input data.

## 2. Make one output format and two verifiers

| Owners | Status | Task |
| --- | --- | --- |
| Joshua, Chengxi | ⚪ | Agree on one basic record format for every dataset: task ID, dataset name, split, starting files, available tools, tool calls and their results, final answer, and check result. |

- [ ] From each reviewed task, make a Codex version and an OpenCode version. Give both versions the same logical task ID and keep them in the same split.
- [ ] Build a verifier that checks whether the model's tool-use output has valid Codex and OpenCode syntax.
- [ ] Build a verifier that checks whether a TRExFitter config is valid and can run when needed. Save a clear pass/fail result, error message, time limit, log, and—when applicable—significance.
- [ ] Test both verifiers and the two output versions on a few known good and bad tasks before publishing data.

## 3. Release, join, and train datasets

| Owner | Status | Task |
| --- | --- | --- |
| Joshua | ⚪ | Convert the checked records into the files verl needs and provide one simple training command for Qwen 1.5B and Qwen 7B. |
| Joshua | ⚪ | Compare untrained Qwen 1.5B/7B, trained Qwen 1.5B/7B, and strong reference models on held-out tasks. Report success separately for each dataset and for Codex versus OpenCode. |

- [ ] Release each checked one-turn dataset separately.
- [ ] Join released datasets into a separate long task: read a ROOT file → write a config → run TRExFitter → explain the result. Keep the source dataset and split recorded for every step.
- [ ] Add RL only after SFT works, using the final result from the verifier.

## 4. Score physics results

- [ ] Decide which fit outputs show a healthy result: run status, significance, yields, uncertainties, pulls, and correlations.
- [ ] Turn those outputs into one simple score that rewards a healthy fit meeting the goal and penalizes failures or unstable results.
- [ ] Test the score on known good and bad configs, and save the full report before sending its final score to RL.

## 5. Later: Harbor data

- [ ] Add the Harbor export under `data/harbor/`, choose useful examples, and keep final test examples out of training.
