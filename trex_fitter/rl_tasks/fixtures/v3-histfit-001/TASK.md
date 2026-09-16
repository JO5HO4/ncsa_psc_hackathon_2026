# V3 native TRExFitter histogram-fit task

Create `analysis.config` as a runnable `ReadFrom: HIST` TRExFitter analysis.

Available input ROOT histograms are in `/workdir/inputs/examples/Histo`, mounted
read-only in the pinned StatAnalysis container. You may inspect them before
writing the config. Build a signal-plus-background fit using the available data,
background, and signal histograms.

The config must contain at least one each of:

- `Job` with `ReadFrom: HIST`, a `POI`, and `HistoPath`;
- `Fit` with `FitType: SPLUSB` and `FitRegion: CRSR`;
- `Region` with a `HistoName`;
- a DATA sample, a BACKGROUND sample, and a SIGNAL sample, each with a
  `HistoFile`;
- a SIGNAL sample `NormFactor` declaration for `SigXsecOverSM`.

Use `h w f s` when you run TRExFitter from a fresh workspace. The terminal
evaluation requires a successful native run, histogram/workspace/fit artifacts,
zero bad fits, and a finite expected significance near the reference baseline.

Allowed mutation: `analysis.config` only. Do not modify inputs, evaluator code,
or task files.
