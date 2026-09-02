# What the model does and what the runner does

## The model

The model gets a physics goal, a starting `.config` file, and any useful error message. It returns one patch inside `<patch>...</patch>`.

The patch may only change allowed `.config` files. It may not contain shell commands, absolute paths, or `..` paths.

## The runner

After the model is finished, the runner:

1. Copies the starting files to a private work folder.
2. Applies the patch.
3. Checks the config.
4. Runs TRExFitter.
5. Returns whether it worked and the significance when available.

The runner decides which TRExFitter steps to use and how long they may run. It never executes a command written by the model.
