# Dataset design

## Dataset programme

We are building a collection of small, independently reviewable **one-turn datasets**, not one monolithic TRExFitter-config dataset. Each supported configuration task is published in three aligned modalities: a Codex tool-using agent episode, an OpenCode tool-using agent episode, and a direct natural-language-to-config response for the model endpoint. Initial families include:

- TRExFitter configuration repair, synthesis, and explanation.
- ROOT-file inspection and modification.
- ATLAS Open Data knowledge and analysis tasks.
- TRExFitter execution, artifact inspection, and result interpretation.

Here, *one turn* means one user task and one bounded response. An agent response may contain several tool calls, their results, and a final answer; a direct response is a single config snippet. Each family has its own source-task schema, fixtures, expected outcome, and train/validation/test split, and uses the two shared project verifiers. Keep final test tasks out of every training family and preserve the split when families are combined.

Later, compose compatible one-turn episodes into long-horizon tasks—for example, inspect a ROOT file, write a config, run TRExFitter, and interpret the fit. Those composed tasks are a distinct dataset with their own held-out evaluation, not a replacement for the component datasets. They are the eventual basis for long-horizon SFT and RL.

## Three target modalities

The trained model has two complementary uses: agentic operation in Codex and OpenCode, and a prompt endpoint that turns a human-readable request into a TRExFitter config snippet.

An agentic record contains:

- a system instruction that establishes the task boundary and tool-use rules;
- the user request and any supplied workspace context;
- assistant tool-call turns, with stable call IDs and JSON arguments;
- tool-result turns that refer to their call ID; and
- a concise final assistant answer, or a justified no-action answer.

Every agentic record carries the native, versioned tool manifest defined in [TOOL_CONTRACT.md](TOOL_CONTRACT.md). Codex records use `Bash` and `apply_patch`; OpenCode records use its native `bash`, `read`, `edit`, `write`, and `apply_patch` tools as needed. Tool names, parameter schemas, call IDs, and observable tool results are part of the harness-specific task contract. Do not train on invented tool output or a hidden verifier result.

The direct `direct_config` modality is not an abbreviated agent transcript. Its prompt states the physics goal, supported config subset, constraints, and necessary local context. Its assistant target contains only the reviewed, self-contained config snippet: no tools, diff fences, explanation, or verifier output. The `messages` field uses the same typed chat structure as agent rows and `tools` is an empty list (`[]`). A consumer can insert the snippet at the documented location in a config template.

### Canonical source, three aligned renderings

Author and review one canonical config task, then render it three ways:

| Field | Codex agent | OpenCode agent | Direct config |
| --- | --- | --- | --- |
| `logical_task_id` | same source task ID | same source task ID | same source task ID |
| `modality` | `codex_agent` | `opencode_agent` | `direct_config` |
| `harness` | `codex` | `opencode` | `none` |
| `messages` | Codex-native tool-call serialization | OpenCode-native tool-call serialization | natural-language request followed by one config-only answer |
| `tools` | semantic JSON-schema tool manifest | same semantic tool manifest | `[]` |
| tool-call IDs and results | preserved | preserved | absent |
| split, fixture revision, config verification | preserved | preserved | preserved |

The renderer owns protocol-specific tokens and message-template details. Do not put a literal Codex or OpenCode wrapper in the canonical source task, and do not hand-maintain divergent config content between renderings. The three renderings may triple the number of training rows, but are **not independent examples**: they share `logical_task_id`, source revision, difficulty, provenance, fixture, and split. Deduplicate and split by `logical_task_id`, never by rendered row ID, so no sibling rendering can occur in a different split.

Before publishing an agent renderer, replay its calls in the corresponding harness against the fixture and compare its observable final state and verifier result with the canonical task. Before publishing a direct renderer, insert its target snippet into the documented fixture/template and run the config verifier. Store the harness/version/template, or direct prompt-template version, so the result can be reproduced.

## Common dataset shape

All published family datasets and the eventual merged dataset should expose the following common columns, in addition to family-specific fields:

```text
id                  unique rendered-record ID, e.g. trex-repair-001--codex-agent
logical_task_id     ID shared by all renderings of one authored task
dataset_family      trex_config | root_io | open_data | trex_execution | ...
modality            codex_agent | opencode_agent | direct_config
harness             codex | opencode | none
split               train | validation | test
messages             typed chat/tool trajectory
tools                JSON-encoded tool manifest; [] for direct_config
fixture              immutable input/environment revision
verification         verifier name, status, and non-secret evidence
provenance           source, authoring/review, and renderer metadata
```

Use typed chat records: assistant tool calls have a function name and a JSON argument string; a tool result has the matching `tool_call_id`. In a `direct_config` record, the assistant has one final message containing only the config snippet and no tool-call messages. This shape is consumed by the verl dataset loader and used by `hep-config-sft`. Family-specific columns such as `initial_config`, `snippet_target`, `snippet_insertion_context`, ROOT object inventory, or fit artifacts may remain alongside these common columns.

## First family: TRExFitter configuration repair

The first task is:

```text
physics goal + starting .config file + diagnostic
                         ->
                 bounded agent repair episode
```

Start with `data/trex_config/fixtures/hyy/hyy.config`, our working H→γγ config. Make small, well-understood mistakes and use native tools: Codex reads through bounded `Bash` commands and changes files with `apply_patch`; OpenCode uses `read`, `edit`/`write`/`apply_patch`, and `bash`. Both harnesses invoke task-provided validation, TRExFitter, and result-inspection commands only in the sandbox. The agent should use supplied tools rather than emit an out-of-band `<patch>...</patch>` answer.

For every supported repair or synthesis task, also create the direct rendering. Phrase its request for a human, provide only necessary local context, and target the smallest valid config block—not a patch or the complete fixture file.

The existing JSON schema in `data/trex_config/schema/` remains the reviewed source-task schema for this family. A dataset builder adds the common columns and produces all three renderings. Later config synthesis and multi-edit repair fit this same family; ROOT and Open Data tasks should get small family-specific source schemas rather than being forced into a TRExFitter-config record.

## Verification and rewards

Each task must state its fixture/environment revision, permitted actions, and
expected observable outcome. The project has exactly two verifiers:

1. a harness-syntax verifier, which checks that each agentic tool-use transcript has
   valid Codex or OpenCode syntax; and
2. a TRExFitter-config verifier, which checks that a changed config is valid
   and runs when the task requires a run. For a direct rendering, it first
   inserts the snippet into its documented template.

ROOT inventories, documentation answers, and fit results are expected task
outcomes saved as evidence; they do not create separate verifier projects.

For fit tasks there are three increasingly strong checks:

1. **Basic check:** the config has the right shape and uses allowed fields.
2. **TRExFitter check:** the real container runs it with the fixture inputs.
3. **Physics result:** the fit meets its task goal, such as a target significance.

The mock runner is only for software tests; its score is not a physics result. For later RL, reward a healthy successful fit and the stated physics goal, and penalize failures, instability, missing output, or needless complexity. The runner owns the reward because it owns the fit artifacts; training receives the final `total` from `trex_fitter/rewards/reward.schema.json`.

## Publishing and merging

Keep reviewed source records as JSON and publish generated, versioned dataset splits to the Hub. Do not hand-edit Parquet. A dataset README must name its family, source schema, agent harness renderer versions, direct prompt-template version, tool manifest revision, fixture revision, split policy, verifier, and intended use.

The merge step takes selected released family versions, validates that their common columns and modality contracts are compatible, balances families and modalities deliberately, and writes a manifest of every source dataset and revision. It must retain family, `logical_task_id`, modality, split, provenance, and verification fields so contamination checks and per-family evaluation remain possible.
