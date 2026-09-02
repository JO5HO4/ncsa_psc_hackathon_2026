# Dataset design

## Dataset programme

We are building a collection of small, independently reviewable **one-turn agent datasets**, not one monolithic TRExFitter-config dataset. Initial families include:

- TRExFitter configuration repair, synthesis, and explanation.
- ROOT-file inspection and modification.
- ATLAS Open Data knowledge and analysis tasks.
- TRExFitter execution, artifact inspection, and result interpretation.

Here, *one turn* means one user task and one bounded agent episode. An episode may contain several tool calls, their results, and a final answer; it is not restricted to one assistant message. Each family has its own source-task schema, fixtures, expected outcome, and train/validation/test split, and uses the two shared project verifiers. Keep final test tasks out of every training family and preserve the split when families are combined.

Later, compose compatible one-turn episodes into long-horizon tasks—for example, inspect a ROOT file, write a config, run TRExFitter, and interpret the fit. Those composed tasks are a distinct dataset with their own held-out evaluation, not a replacement for the component datasets. They are the eventual basis for long-horizon SFT and RL.

## Agentic harness target

The trained model must operate in both Codex and OpenCode agent harnesses. A trainable example therefore represents an agent interaction, not a prose question/answer pair. It contains:

- a system instruction that establishes the task boundary and tool-use rules;
- the user request and any supplied workspace context;
- assistant tool-call turns, with stable call IDs and JSON arguments;
- tool-result turns that refer to their call ID; and
- a concise final assistant answer, or a justified no-action answer.

Every record also carries a JSON-schema tool manifest. Tool names, parameter schemas, call IDs, and the observable tool results are part of the task contract. Do not train on invented tool output or on a hidden verifier result. The local `hep-config-sft` reference dataset demonstrates this `messages` plus `tools` structure.

### Canonical source, two harness renderings

Author and review one canonical episode, then render it twice:

| Field | Codex rendering | OpenCode rendering |
| --- | --- | --- |
| `logical_task_id` | same source task ID | same source task ID |
| `harness` | `codex` | `opencode` |
| `messages` | Codex-native assistant tool-call serialization | OpenCode-native assistant tool-call serialization |
| `tools` | the same semantic JSON-schema tool manifest | the same semantic JSON-schema tool manifest |
| tool-call IDs and tool results | preserved | preserved |
| split, fixture revision, verifier outcome | preserved | preserved |

The renderer owns protocol-specific tokens and message-template details. Do not put a literal Codex or OpenCode wrapper in the canonical source episode, and do not hand-maintain two divergent answers. The two renderings may double the number of training rows, but they are **not independent examples**: they must share `logical_task_id`, source revision, difficulty, provenance, and split. Deduplicate and split by `logical_task_id`, never by rendered row ID, so that a Codex rendering cannot appear in training while its OpenCode sibling appears in validation or test.

Before publishing a renderer, replay its calls in the corresponding harness against the fixture and compare its observable final state and verifier result with the canonical episode. Store the harness/version/template used for the render so the result can be reproduced.

## Common dataset shape

All published family datasets and the eventual merged dataset should expose the following common columns, in addition to family-specific fields:

```text
id                  unique rendered-record ID, e.g. trex-repair-001--codex
logical_task_id     ID shared by all renderings of one authored task
dataset_family      trex_config | root_io | open_data | trex_execution | ...
harness             codex | opencode
split               train | validation | test
messages             typed chat/tool trajectory
tools                JSON-encoded tool manifest
fixture              immutable input/environment revision
verification         verifier name, status, and non-secret evidence
provenance           source, authoring/review, and renderer metadata
```

Use typed chat records: assistant tool calls have a function name and a JSON argument string; a tool result has the matching `tool_call_id`. This is the shape already consumed by the verl dataset loader and used by `hep-config-sft`. Family-specific columns such as `initial_config`, `expected_config`, ROOT object inventory, or fit artifacts may remain alongside these common columns.

## First family: TRExFitter configuration repair

The first task is:

```text
physics goal + starting .config file + diagnostic
                         ->
                 bounded agent repair episode
```

Start with `data/trex_config/fixtures/hyy/hyy.config`, our working H→γγ config. Make small, well-understood mistakes and define a limited tool set (for example: read the config, inspect allowed fields, apply a patch, validate, run, and inspect results). The agent should use the supplied tools rather than emit an out-of-band `<patch>...</patch>` answer. A rendered trajectory can still use a unified diff as an argument to an `apply_patch` tool when that is the appropriate interface.

The existing JSON schema in `data/trex_config/schema/` remains the reviewed source-task schema for this family. A dataset builder adds the common agentic columns and produces the Codex and OpenCode renderings. Later config synthesis and multi-edit repair fit this same family; ROOT and Open Data tasks should get small family-specific source schemas rather than being forced into a TRExFitter-config record.

## Verification and rewards

Each task must state its fixture/environment revision, permitted actions, and
expected observable outcome. The project has exactly two verifiers:

1. a harness-syntax verifier, which checks that the tool-use transcript has
   valid Codex and OpenCode syntax; and
2. a TRExFitter-config verifier, which checks that a changed config is valid
   and runs when the task requires a run.

ROOT inventories, documentation answers, and fit results are expected task
outcomes saved as evidence; they do not create separate verifier projects.

For fit tasks there are three increasingly strong checks:

1. **Basic check:** the config has the right shape and uses allowed fields.
2. **TRExFitter check:** the real container runs it with the fixture inputs.
3. **Physics result:** the fit meets its task goal, such as a target significance.

The mock runner is only for software tests; its score is not a physics result. For later RL, reward a healthy successful fit and the stated physics goal, and penalize failures, instability, missing output, or needless complexity. The runner owns the reward because it owns the fit artifacts; training receives the final `total` from `trex_fitter/rewards/reward.schema.json`.

## Publishing and merging

Keep reviewed source records as JSON and publish generated, versioned dataset splits to the Hub. Do not hand-edit Parquet. A dataset README must name its family, source schema, harness renderer versions, tool manifest revision, fixture revision, split policy, verifier, and intended use.

The merge step takes selected released family versions, validates that their common agentic columns and tool contracts are compatible, balances families and harnesses deliberately, and writes a manifest of every source dataset and revision. It must retain family, `logical_task_id`, split, provenance, and verification fields so contamination checks and per-family evaluation remain possible.
