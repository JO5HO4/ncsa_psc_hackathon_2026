# TRExFitter hackathon

We are building a starting point for teaching a small language model to operate a high-energy-physics analysis environment through agentic tools.

We will release several small, verified one-turn datasets: TRExFitter config work, ROOT-file inspection and modification, ATLAS Open Data knowledge, and execution/result interpretation. Each supported config task has three aligned renderings: Codex and OpenCode tool-use harnesses, plus a direct natural-language-to-config response for the model endpoint. The direct response contains only an insertable config snippet; the agent versions use bounded tools to inspect, repair, validate, and run. All sibling renderings share a logical task ID and split. Compatible family releases are then composed into a separate long-horizon dataset.

## ROOT runtime

`training/root_runtime.sh` is the lightweight ROOT runtime for Perlmutter. It
activates CVMFS `StatAnalysis,0.5.1` (ROOT 6.34.02) and avoids using that
release's Python executable, which can fail on Perlmutter due to
`libcrypt.so.2`. It is not a Python virtual environment: ROOT's C++ runtime is
provided by CVMFS.

```bash
training/root_runtime.sh root --version
training/root_runtime.sh --python data/datasets/atlas-open-data-sft-dataset/scripts/validate_dataset.py
```

Our first starting point is the working H→γγ config at [data/configs/examples/hyy.config](data/configs/examples/hyy.config).

## Clone it

`verl` is a Git submodule. Clone it with the repository:

```bash
git clone --recurse-submodules git@github.com:JO5HO4/ncsa_psc_hackathon_2026.git
```

For an existing clone, initialize the dependency:

```bash
git submodule update --init --recursive
```

`bash training/bootstrap_verl.sh` remains available when only the training
dependency needs to be initialized. Download the reference training Parquet
files directly from Hugging Face when needed:

```bash
bash data/fetch_reference_datasets.sh
```

## Main folders

| Folder | What it is for |
| --- | --- |
| [data/](data/README.md) | Dataset families, fixtures, and publishing conventions |
| [trex_fitter/](trex_fitter/README.md) | The code that checks and runs configs |
| [training/](training/README.md) | Training with verl |
| [inference/](inference/README.md) | Testing a trained model |
| [docs/](docs/HACKATHON.md) | The plan and task list |

Start with the [hackathon plan](docs/HACKATHON.md), then pick a task from the [task board](docs/TASK_BOARD.md).
