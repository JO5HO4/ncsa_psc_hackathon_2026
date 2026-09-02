# Training and testing plan

## What we will train

We will train two small Qwen models on the datasets made in this project:

- Qwen 1.5B
- Qwen 7B

Each training dataset includes both the Codex and OpenCode versions of the
same tasks. The two versions of a task must stay in the same data split.

## What we will compare

Use the same held-out tasks to compare three groups:

| Group | Models |
| --- | --- |
| Untrained baseline | Qwen 1.5B and Qwen 7B before training on our data |
| Trained models | The same Qwen 1.5B and Qwen 7B after training on our data |
| Strong reference models | Available state-of-the-art models, tested without training on our data |

For every model, report results separately for each dataset family and for
Codex versus OpenCode. Do not use any held-out task, or its Codex/OpenCode
partner, during training.

## What the training data needs

The dataset builder turns each reviewed task into the chat and tool-call format
used by the training code. Each record includes the user request, available
tools, tool calls and their results, and the final answer. The builder makes
one Codex record and one OpenCode record for the same task.

People review the source tasks and the results from the two verifiers. The
training code creates the Parquet files it needs; do not edit those files by
hand.

## What to save

For each run, save the model name and size, dataset version, training command,
and test results. For each test task, save whether the output has valid
Codex/OpenCode syntax and, for config tasks, whether the TRExFitter config is
valid and runs when required.

Reinforcement learning is a later step only if this supervised-training
comparison works first.
