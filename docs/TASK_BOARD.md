# Task board

Pick one area and put an owner next to each task.

## Config tasks and physics

- [ ] Review the working H→γγ config in `data/trex_config/fixtures/hyy/hyy.config` and choose the first small part of the `.config` language we will support.
- [ ] Write five simple physics goals.
- [ ] Make a starting config and a correct fixed config for each goal.
- [ ] Make a few broken versions: missing sample, bad region, bad setting, and so on.
- [ ] Split tasks into training, validation, and final test sets.

## TRExFitter Histogramming

- [ ] TRExFitter histogramming is very slow; need some other backend to handle histogramming
- [ ] Start with a dataset that is already histogrammed
- [ ] Modify TRExFitter to return a success/fail based on usability of the config file

## Checking and running configs

- [ ] Decide what “valid config” means for the first version.
- [ ] Build a local checker for the supported config fields.
- [ ] Build a real TRExFitter check using the container and fixture inputs.
- [ ] Make one command that takes a finished config and returns whether it worked and its significance.
- [ ] Add clear error messages, time limits, and saved logs.

## SFT Dataset
- [ ] (Joshua + Chengxi) Decide on a standardized format of data and required fields
- [ ] (Chengxi) Generate SFT dataset from TRExFitter documentation, frame it as operational question/answers pairs in the context of Hyy https://trexfitter-docs.web.cern.ch/trexfitter-docs/latest/settings/
- [ ] (Joshua) Open Data documentation https://opendata.atlas.cern/docs/data/for_education/13TeV25_details
- [ ] (Dongwon) Opening root files and interacting with the objects inside (understand what is inside, the variables, summarize into natural language output)
- [ ] Create some sort of verifier for the dataset
- [ ] Trajectory dataset (e.g. asking model to run trex fitter as a tool call, ask model to extract the significance given some trex fitter artifacts)
- [ ] Long horizon tasks as a combination of all other datasets (start from reading root file, write config, run and execute config, interpret results) --> eventually use in RL

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
(Joshua)
- [ ] Make sure that the model outputs are usable by an agentic harness (codex, opencode)
- [ ] Turn the JSON task records into the format verl needs.
- [ ] Make one simple command to train a small model with SFT.
- [ ] Test the model on the held-out tasks.
- [ ] Report patch success, config validity, and TRExFitter run success.
- [ ] Only after SFT works, add RL scoring based on the final config result.

## Training benchmarking 
(Joshua)
- [ ] Test untrained small language model without any context
- [ ] Test untrained small language model with some additional context
- [ ] Test small language model after RL/SFT
- [ ] Test SOTA model without any context
- [ ] Test SOTA models with additional context

## Harbor data — later

- [ ] Add the Harbor export under `data/harbor/`.
- [ ] Decide which Harbor examples are useful for training.
- [ ] Keep any final test examples out of the training data.
