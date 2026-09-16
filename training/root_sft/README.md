# ROOT SFT: executable file-operation pilot

This directory holds the executable ROOT-operation pilot: structured-operation
SFT data over real synthetic ROOT files, plus a deterministic execution
evaluator. It uses the repository's unchanged `training/scripts/run_verl_sft.sh`,
dataset adapter, model exporter, and inference runner; it does not introduce
another trainer or change verl.

See [FILE_TASKS.md](FILE_TASKS.md) for what is included, how to build and test
the release, how to establish a baseline, and how to run training with the
existing generic verl launcher.

`requirements.txt` covers dataset preparation and checks only (`pyarrow`); GPU
training dependencies remain owned by the existing verl container and
`training/scripts/setup.sh`.
