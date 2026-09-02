# Training and testing plan

## What we will train

We will train two small Qwen models on the datasets made in this project:

- Qwen 1.5B
- Qwen 7B

Each supported config task has three aligned versions: Codex agent, OpenCode
agent, and direct natural-language-to-config. The direct version teaches the
prompt endpoint to return only a config snippet, without tools. All three
versions of a task must stay in the same data split.

## What we will compare

Use the same held-out tasks to compare three groups:

| Group | Models |
| --- | --- |
| Untrained baseline | Qwen 1.5B and Qwen 7B before training on our data |
| Trained models | The same Qwen 1.5B and Qwen 7B after training on our data |
| Strong reference models | Available state-of-the-art models, tested without training on our data |

For every model, report results separately for each dataset family and
modality: Codex agent, OpenCode agent, and direct config. Do not use any
held-out task, or any of its sibling renderings, during training.

## What the training data needs

The dataset builder turns each reviewed task into the chat format used by the
training code. Agent records include the user request, available tools, tool
calls and their results, and the final answer. A direct-config record includes
a human-readable user request and one assistant answer containing only the
reviewed config snippet; its `tools` column is `[]`. The builder makes one
Codex record, one OpenCode record, and one direct-config record for the same
logical task.

People review the source tasks and the results from the two verifiers. The
training code creates the Parquet files it needs; do not edit those files by
hand.

## What to save

For each run, save the model name and size, dataset version, training command,
and test results. For each test task, save whether an agent output has valid
Codex/OpenCode syntax and, for config tasks in every modality, whether the
TRExFitter config is valid and runs when required. Validate a direct response
by inserting the snippet into its documented template.

Reinforcement learning is a later step only if this supervised-training
comparison works first.
