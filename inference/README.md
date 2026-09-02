# Testing a trained model

This folder will hold the code used to test a trained model.

For each unseen task, the model receives the goal and starting config and returns a patch. The TRExFitter runner then checks and runs the finished config. We record the patch, whether it worked, and the final significance when available.

The model must not receive the correct patch for a test task.
