# Hackathon plan

## What we want to show

Given a physics goal and a broken TRExFitter `.config` file, a small model can suggest a correct fix. We then check the config and run TRExFitter ourselves to see whether the fix works.

## The plan

1. Pick a small, well-defined part of the TRExFitter config language.
2. Start from the working H→γγ config in `data/trex_config/fixtures/hyy/hyy.config` and make a few good example tasks: goal, broken config, and correct fix.
3. Build a checker that can tell us whether a config is valid and whether TRExFitter can run it.
4. Train a small model on the examples using verl.
5. Test the trained model on examples it has not seen before.
6. If the basic training works, try reinforcement learning (RL), where better working configs get better scores.

## What counts as a good result

For a held-out task, the model should produce a patch that:

- Applies cleanly to the starting config.
- Makes the config pass our checks.
- Lets TRExFitter run.
- Meets the physics goal when possible.

Keep test tasks separate from training tasks. Save enough information to repeat a run later: dataset version, model, command, and result.
