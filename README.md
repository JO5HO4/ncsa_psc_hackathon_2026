# TRExFitter hackathon

We are building a starting point for teaching a small language model to fix TRExFitter `.config` files.

The model receives a physics goal and a starting config. It returns a patch. Our software then checks the patch and runs TRExFitter. The model does not run commands or TRExFitter itself.

Our first starting point is the working H→γγ config at [data/trex_config/fixtures/hyy/hyy.config](data/trex_config/fixtures/hyy/hyy.config).

## Clone it

`verl` is a Git submodule. Clone it with the repository:

```bash
git clone --recurse-submodules git@github.com:JO5HO4/ncsa_psc_hackathon_2026.git
```

For an existing clone, run `bash training/bootstrap_verl.sh`. Download the
reference training Parquet files directly from Hugging Face when needed:

```bash
bash data/fetch_reference_datasets.sh
```

## Main folders

| Folder | What it is for |
| --- | --- |
| [data/](data/README.md) | Datasets and new config tasks |
| [trex_fitter/](trex_fitter/README.md) | The code that checks and runs configs |
| [training/](training/README.md) | Training with verl |
| [inference/](inference/README.md) | Testing a trained model |
| [docs/](docs/HACKATHON.md) | The plan and task list |

Start with the [hackathon plan](docs/HACKATHON.md), then pick a task from the [task board](docs/TASK_BOARD.md).
