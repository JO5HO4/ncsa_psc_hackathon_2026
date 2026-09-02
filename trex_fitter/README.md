# TRExFitter runner

This folder will hold the code that checks a finished `.config` file and runs TRExFitter.

The model gives us a patch. We apply the patch ourselves, then this runner checks and runs the finished config. The model never runs TRExFitter directly.

`scripts/trex.py` uses the CERN StatAnalysis container through `podman-hpc`. It needs the input links in `inputs/` to work on the machine where it runs.

To download the 2025 diphoton files needed by the working H→γγ config, run:

```bash
bash trex_fitter/scripts/fetch_hyy_inputs.sh
```

This creates local `inputs/Data/` and `inputs/MC/` folders. They are ignored
by git and contain only the 16 data and 10 MC ROOT files named in the config.

## Test the H→γγ config

Run this from the repository root on a host compute node, not inside the verl
container. It checks the available StatAnalysis container, input files, and
the complete H→γγ workflow:

```bash
python3 trex_fitter/scripts/trex.py ignored.config --check
```

```bash
python3 trex_fitter/scripts/evaluate_config.py \
  data/trex_config/fixtures/hyy/hyy.config \
  --actions n w f s \
  --timeout 7200 \
  --log-dir trex_fitter/workspaces/evaluations/hyy-reference
```

The second command makes histograms, builds the workspace, fits it, and
calculates significance. Logs and the JSON result are written to
`trex_fitter/workspaces/evaluations/hyy-reference/`.

## Reward format for later RL

After a completed run, the evaluator will write one explainable reward report
next to its logs. The report format and the place to implement its reader are
in [rewards/](rewards/README.md). This is evaluator code, not a tool for the
model and not training-framework code.

The starting configs live with the data, not in this folder. The first one is `../data/trex_config/fixtures/hyy/hyy.config`. The interface copies a supplied config into its own work folder before it runs it.

`scripts/mock_trex.py` is only a fast fake runner for testing software. It does not check configs or do physics.

The first real interface should return:

- Whether the config ran successfully.
- The significance, if TRExFitter produced one.
- Useful error messages and log locations.

See the shared [agent tool contract](../docs/TOOL_CONTRACT.md) for the boundary between the model and the runner.
