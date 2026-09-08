# TRExFitter inputs

This directory separates shared TRExFitter example samples from inputs specific
to the local H→γγ configuration.

| Path | Purpose | Versioned |
| --- | --- | --- |
| `examples/` | Symlink to the shared, read-only TRExFitter example sample collection. | No |
| `hyy/Data/` | H→γγ Open Data ROOT files. | No |
| `hyy/MC/` | H→γγ simulated ROOT files. | No |

The canonical test-sample path in this workspace is
`data/samples/examples/`. It resolves to the shared project copy, which
is readable by all collaborators. Do not duplicate those files in this
repository.

Large inputs are intentionally ignored by Git. Fetch or stage the H→γγ inputs
with `bash trex_fitter/scripts/fetch_hyy_inputs.sh` when that script is
available for the checked-out workflow.
