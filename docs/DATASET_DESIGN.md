# Dataset design

## First task type: fix a config

The first task is simple:

```text
physics goal + starting .config file + error message
                         ->
                    config patch
```

The model returns a unified diff inside `<patch>...</patch>`. We apply the patch, check the result, and run TRExFitter. The model does not get to run TRExFitter or inspect its output.

Start with `data/trex_config/fixtures/hyy/hyy.config`, our working H→γγ config. Make the first tasks by copying it and introducing small, well-understood mistakes.

Later, we can add tasks where the model writes a small config from scratch or makes several edits in a row.

## What each task needs

- A task ID and a split: `train`, `validation`, or `test`.
- A short physics goal.
- The starting `.config` text.
- Limits on what may be changed.
- A clear error message when the starting config is broken.
- The correct patch for training examples.
- A way to tell whether the finished config worked.

The JSON schema in `data/trex_config/schema/` lists the exact fields.

## Three meanings of “works”

1. **Basic check:** the config has the right shape and uses allowed fields.
2. **TRExFitter check:** the real TRExFitter container can run the config with its input files.
3. **Physics result:** the finished fit meets the task goal, such as reaching a target significance.

The mock runner is only for testing our software. Its score is fake and must not be used as a physics result.

## Reward for later RL

For RL, we want a score that reflects a useful physics result, not only a large significance. After TRExFitter runs, the runner should read its output files and save a small set of numbers, such as whether the fit finished, the significance, yields, uncertainties, nuisance-parameter pulls, and correlations.

The first reward should be simple and easy to explain:

- Give credit when the config runs and the fit is healthy.
- Give credit for meeting the physics goal.
- Penalize failed or unstable fits, missing output, and needlessly complicated changes.

The exact physics choices need to be agreed on and tested against known good and bad configs before using the reward for training.

The fixed output format lives in `trex_fitter/rewards/reward.schema.json`.
The runner owns reward code because it owns the fit artifacts; training only
receives the final `total` value.

## Training data

We will keep task records as JSON because they are easy to read and review. Later, training code will turn them into the Parquet files that verl expects. More detail is in [TRAINING_INTERFACE.md](TRAINING_INTERFACE.md).
