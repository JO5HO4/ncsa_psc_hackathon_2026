# Hackathon plan

## Goal

Train a small language model that can help with common high-energy-physics
analysis work. A user should be able to give it a new prompt, such as asking
what is inside a ROOT file, how to change a TRExFitter config, how to run a
fit, or how to explain a result.

The final model must work in both of these ways:

- through a simple model endpoint or command that accepts user prompts; and
- as the model inside both Codex and OpenCode tool-using agents.

For the agent version, the model must make tool calls in a form that the chosen
harness accepts. The tools—not the model—inspect files, change configs, run
TRExFitter, and return results.

## What we will build

We will first make several small, checked datasets. Each example is one user
request and the agent's complete response to it, including any tool calls and
tool results. The first datasets cover:

- changing and checking TRExFitter configs;
- looking inside ROOT files and describing their contents;
- answering questions about ATLAS Open Data;
- running TRExFitter and explaining its output.

We will make a Codex version and an OpenCode version of every reviewed task.
They are two formats for the same task, not two independent tasks, so they
must always stay in the same train, validation, or test split.

Once the small datasets work, we will combine them into longer tasks, such as:

```text
inspect a ROOT file → write a config → run TRExFitter → explain the fit
```

## Checks before training

Every example must pass the checks that apply to it. We need two checkers:

1. A harness-syntax verifier: confirms that tool-use output has valid Codex and
   OpenCode syntax.
2. A TRExFitter-config verifier: confirms that a changed config is valid and,
   when needed, that TRExFitter can run it.

The second checker saves a clear pass/fail result, errors, logs, time used, and
the significance when a fit is run. We will test both checkers with known good
and bad examples before using a dataset for training.

## Work plan

1. Pick a small part of the TRExFitter config language and create simple
   correct and broken examples from the H→γγ fixture.
2. Build the ROOT-file, Open Data, TRExFitter-running, and result-explanation
   datasets.
3. Find a faster histogramming path, either through another backend or by
   using data that is already histogrammed.
4. Define one shared record format and make Codex and OpenCode versions.
5. Build and test the two checkers.
6. Train Qwen 1.5B and Qwen 7B with the checked datasets using verl.
7. Test held-out tasks with untrained Qwen, trained Qwen, and strong reference
   models.
8. Package the trained model for prompt-based use and for Codex/OpenCode use.
9. Only after supervised training works, try reinforcement learning using the
   final checker result as the score.

The detailed, assigned work is in [TASK_BOARD.md](TASK_BOARD.md).

## Final deliverable

The final demo should include:

- a trained small model exported in a reusable format;
- a simple way to send it different prompts (a local command or endpoint);
- working Codex and OpenCode integrations that can use the same model for tool
  calls;
- a small held-out test report showing which tasks it can complete; and
- the dataset and model versions, training command, and checker results needed
  to repeat the demo.

## What counts as success

For a held-out task, success means the model uses the allowed tools correctly,
produces output accepted by the target harness, and reaches the requested
result. For a config task, that also means the config is valid and TRExFitter
runs when the task requires it. For a fit task, report whether the physics goal
was met as well as whether the run succeeded.
