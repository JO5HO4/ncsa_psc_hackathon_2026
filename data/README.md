# Data

`reference/` holds the two reference datasets after they are downloaded from
Hugging Face. They are not stored in this repository. Fetch the needed Parquet
files with:

```bash
bash data/fetch_reference_datasets.sh
```

`trex_config/` is the first of several small dataset families. Each supported config task is rendered as a Codex agent episode, an OpenCode agent episode, and a direct natural-language-to-config example for the prompt endpoint. The direct target is only an insertable config snippet and has an empty `tools` list. Other families will cover ROOT-file work, ATLAS Open Data knowledge, TRExFitter execution, and fit-result interpretation. Each family owns small reviewed source-task records, fixtures, and a verifier; family releases are later combined into a separately versioned long-horizon dataset.

The first starting config is `trex_config/fixtures/hyy/hyy.config`. It is a working H→γγ TRExFitter config and should be the base for our first example tasks.

`harbor/` is reserved for Harbor data when it arrives.

Write new TRExFitter-config tasks as JSON records using `trex_config/schema/`. Do not create training Parquet files by hand. A dataset builder will turn each verified source task into an agent episode rendered for `codex` and `opencode`, plus a `direct_config` rendering. All rendered rows share a `logical_task_id` and split; they must never be separated across train, validation, or test. Verify a direct target by inserting its config snippet into the documented fixture/template.

All agentic records use the v1 [native-tool contract](../docs/TOOL_CONTRACT.md). Do not create one-off tools, MCP servers, or custom wrappers for individual dataset families.

## Dataset publishing convention

Treat a Hugging Face dataset repository as the canonical home for every
reviewed dataset we create: individual task-family releases, Codex/OpenCode
agent and direct-config SFT renderings, RL prompts, held-out evaluation tasks,
merged long-horizon tasks, and inference prompt sets. Keep only schemas, small examples, and download
scripts in this Git repository. Large or generated dataset files should be
published to the Hub and fetched by dataset ID and revision.

For inference, a Hub dataset can use any string prompt column; pass its name to
`inference/run_prompts.py --prompt-field`. The dataset's README should state
the family, schema, agent harness/template versions, direct prompt-template
version, tool-manifest revision, splits, source fixture version, verifier,
intended use, and immutable commit or tag to use for evaluations.

When a reviewed local split is ready, publish it with a logged-in Hugging Face
account instead of committing it here:

```bash
python data/publish_dataset.py \
  --source data/trex_config/splits/train.jsonl \
  --repo-id ho22joshua/trex-config-tasks \
  --split train
```

Run the command once for each split. It accepts `.json`, `.jsonl`, and
`.parquet`; use `--config-name` when one dataset repository hosts multiple
schemas.

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
