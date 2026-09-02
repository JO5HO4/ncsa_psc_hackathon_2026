# Training data format

This is a plan for later. There is no one-command training tool yet.

People making physics tasks write JSON records in `data/trex_config/`. The training code will read those records and make the files verl needs.

For supervised fine-tuning (SFT), each training example will look like a chat:

```text
system: You may only edit the supplied .config file.
user:   Physics goal, starting config, and error message.
assistant: <patch>...the correct diff...</patch>
```

For reinforcement learning (RL), the model sees the same task and returns a patch. Our checker applies the patch and runs TRExFitter. The result becomes the score for that answer.

Important rules:

- People review JSON task records, not Parquet files.
- Test answers must never be used for training.
- A task keeps the same ID and starting config once it is published.
- Every training run should save the model, data version, command, and result.
