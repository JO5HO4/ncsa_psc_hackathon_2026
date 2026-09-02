# Data

`reference/` holds the two reference datasets after they are downloaded from
Hugging Face. They are not stored in this repository. Fetch the needed Parquet
files with:

```bash
bash data/fetch_reference_datasets.sh
```

`trex_config/` is where we will build the new TRExFitter config dataset. Each task describes a physics goal, a starting `.config` file, and the expected fix.

The first starting config is `trex_config/fixtures/hyy/hyy.config`. It is a working H→γγ TRExFitter config and should be the base for our first example tasks.

`harbor/` is reserved for Harbor data when it arrives.

Write new tasks as JSON records using `trex_config/schema/`. Do not create training Parquet files by hand; the training code will create them later.

## ATLAS Open Data

Use `fetch_atlas_opendata.sh` to download complete Open Data skims directly
from `ho22joshua/atlas_opendata`. Files are stored under `atlas_opendata/`,
preserving the Hub layout, and are ignored by git.

```bash
# One 2025 skim
bash data/fetch_atlas_opendata.sh 2to4lep

# The 2025 diphoton collection
bash data/fetch_atlas_opendata.sh GamGam

# Every available ROOT file; this is a very large download
bash data/fetch_atlas_opendata.sh all
```

The H→γγ fixture needs only a small diphoton subset. Use
`bash trex_fitter/scripts/fetch_hyy_inputs.sh` for that instead of downloading
the full GamGam skim.
