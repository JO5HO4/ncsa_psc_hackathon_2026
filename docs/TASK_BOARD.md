# Task board

Pick one area and put an owner next to each task.

## Config tasks and physics

- [ ] Review the working H→γγ config in `data/trex_config/fixtures/hyy/hyy.config` and choose the first small part of the `.config` language we will support.
- [ ] Write five simple physics goals.
- [ ] Make a starting config and a correct fixed config for each goal.
- [ ] Make a few broken versions: missing sample, bad region, bad setting, and so on.
- [ ] Split tasks into training, validation, and final test sets.

## Checking and running configs

- [ ] Decide what “valid config” means for the first version.
- [ ] Build a local checker for the supported config fields.
- [ ] Build a real TRExFitter check using the container and fixture inputs.
- [ ] Make one command that takes a finished config and returns whether it worked and its significance.
- [ ] Add clear error messages, time limits, and saved logs.

## Physics rewards from a fit

A reward is the number used to tell the model whether one finished config is better than another. It should use more than significance alone.

- [ ] List the useful files TRExFitter writes after a run: fit status, yields, uncertainties, pulls, correlations, plots, and significance.
- [ ] Decide which numbers show that a fit is healthy and which numbers show a bad or unphysical fit.
- [ ] Define a first reward made from several simple parts: successful run, fit quality, sensible uncertainties, reasonable nuisance-parameter pulls, and the physics goal.
- [ ] Add penalties for failed fits, missing output, unstable fits, or changes that make the config much more complicated without helping the result.
- [ ] Update the evaluator so every run keeps a private copy of its TRExFitter artifacts before a reward is read.
- [ ] Write code that reads the chosen output files and saves these numbers in one simple result file.
- [ ] Fill the standard `trex_fitter/rewards/reward.schema.json` report; send only its `total` field to RL.
- [ ] Test the reward on a small set of known good and bad configs to make sure it prefers the good ones for the right reasons.
- [ ] Decide which reward parts are used for training and which are only reported to people.

## Training and testing models

- [ ] Turn the JSON task records into the format verl needs.
- [ ] Make one simple command to train a small model with SFT.
- [ ] Test the model on the held-out tasks.
- [ ] Report patch success, config validity, and TRExFitter run success.
- [ ] Only after SFT works, add RL scoring based on the final config result.

## Training benchmarking

- [ ] Test untrained small language model without any context
- [ ] Test untrained small language model with some additional context
- [ ] Test small language model after RL/SFT
- [ ] Test SOTA model without any context
- [ ] Test SOTA models with additional context

## Harbor data — later

- [ ] Add the Harbor export under `data/harbor/`.
- [ ] Decide which Harbor examples are useful for training.
- [ ] Keep any final test examples out of the training data.
